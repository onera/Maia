import numpy as np

import Pypdm.Pypdm as PDM

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.algo     import indexing
from maia.transfer import protocols as EP
from maia.utils    import par_utils, np_utils
from maia.utils    import vstride as vs

def PDM_dfacecell_to_dcellface(comm, face_distri, cell_distri, face_cell):
  _face_distri = np_utils.safe_int_cast(face_distri, PDM.npy_pdm_gnum_dtype)
  _cell_distri = np_utils.safe_int_cast(cell_distri, PDM.npy_pdm_gnum_dtype)
  _dface_cell  = np_utils.safe_int_cast(face_cell, PDM.npy_pdm_gnum_dtype)
  _cell_face_idx, _cell_face = PDM.dfacecell_to_dcellface(comm, _face_distri, _cell_distri, _dface_cell)
  cell_face_idx = np_utils.safe_int_cast(_cell_face_idx, face_cell.dtype)
  cell_face     = np_utils.safe_int_cast(_cell_face, face_cell.dtype)
  return vs.from_displs(cell_face_idx, cell_face)

def PDM_dcellface_to_dfacecell(comm, face_distri, cell_distri, cell_face:vs.VStrideArray):
  _face_distri   = np_utils.safe_int_cast(face_distri, PDM.npy_pdm_gnum_dtype)
  _cell_distri   = np_utils.safe_int_cast(cell_distri, PDM.npy_pdm_gnum_dtype)
  _cell_face_idx = np_utils.safe_int_cast(cell_face.displs, np.int32)
  _cell_face     = np_utils.safe_int_cast(cell_face.values, PDM.npy_pdm_gnum_dtype)
  _face_cell = PDM.dcellface_to_dfacecell(comm, _face_distri, _cell_distri, _cell_face_idx, _cell_face)
  return np_utils.safe_int_cast(_face_cell, cell_face.dtype)

def PDM_dfacevtx_from_face_and_edge(comm, face_distri, edge_distri, face_edge:vs.VStrideArray, edge_vtx):
  _face_distri   = np_utils.safe_int_cast(face_distri, PDM.npy_pdm_gnum_dtype)
  _edge_distri   = np_utils.safe_int_cast(edge_distri, PDM.npy_pdm_gnum_dtype)
  _face_edge_idx = np_utils.safe_int_cast(face_edge.displs, np.int32)
  _face_edge     = np_utils.safe_int_cast(face_edge.values, PDM.npy_pdm_gnum_dtype)
  _edge_vtx      = np_utils.safe_int_cast(edge_vtx, PDM.npy_pdm_gnum_dtype)
  _face_vtx = PDM.compute_dfacevtx_from_face_and_edge(comm, _face_distri, _edge_distri, _face_edge_idx, _face_edge, _edge_vtx)
  face_vtx  = np_utils.safe_int_cast(_face_vtx, face_edge.dtype)
  return vs.from_displs(face_edge.displs, face_vtx) # Same displs


def pe_to_nface(zone, comm, remove_PE=False):
  """Create a NFace node from a NGon node with ParentElements.

  NGon range is supposed to start at 1.
  Input tree is modified inplace.

  Args:
    zone       (CGNSTree): Distributed zone
    comm       (MPIComm) : MPI communicator
    remove_PE  (bool, optional): If True, remove the ParentElements node.
      Defaults to False.
  """
  ngon_node = PT.Zone.NGonNode(zone)
  nface_distri = MT.getDistribution(zone, 'Cell')[1]
  ngon_distri  = MT.getDistribution(ngon_node, 'Element')[1]
  face_distri = par_utils.partial_to_full_distribution(ngon_distri, comm)
  cell_distri = par_utils.partial_to_full_distribution(nface_distri, comm)
  assert PT.Element.Range(ngon_node)[0] == 1
  local_pe = indexing.get_pe_local(ngon_node).reshape(-1, order='C')

  cell_face = PDM_dfacecell_to_dcellface(comm, face_distri, cell_distri, local_pe)
  cell_face_range  = np.array([1, PT.Zone.n_cell(zone)], zone[1].dtype) + PT.Zone.n_face(zone)
  nface_ec_distr_f = par_utils.gather_and_shift(cell_face.dsize, comm)
  nface_ec_distri  = par_utils.full_to_partial_distribution(nface_ec_distr_f, comm)
  nface_ec_distri  = np_utils.safe_int_cast(nface_ec_distri, nface_distri.dtype)
  eso = cell_face.displs + nface_ec_distri[0]

  nface = PT.new_NFaceElements(erange=cell_face_range, eso=eso, ec=cell_face.values, parent=zone)
  MT.newDistribution({"Element" : nface_distri, "ElementConnectivity" : nface_ec_distri}, nface)

  if remove_PE:
    PT.rm_children_from_name(ngon_node, "ParentElements")

def nface_to_pe(zone, comm, remove_NFace=False):
  """Create a ParentElements node in the NGon node from a NFace node.

  Input tree is modified inplace.

  Args:
    zone         (CGNSTree): Distributed zone
    comm         (MPIComm) : MPI communicator
    remove_NFace (bool, optional): If True, remove the NFace node.
      Defaults to False.
  """
  ngon_node  = PT.Zone.NGonNode(zone)
  nface_node = PT.Zone.NFaceNode(zone)
  ngon_distri    = MT.getDistribution(ngon_node , 'Element')[1]
  nface_distri   = MT.getDistribution(nface_node, 'Element')[1]

  face_distri = par_utils.partial_to_full_distribution(ngon_distri, comm)
  cell_distri = par_utils.partial_to_full_distribution(nface_distri, comm)
  
  cell_face = MT.Element.connectivity(nface_node)

  # If NFace are before NGon, then face ids must be shifted
  if PT.Element.Range(ngon_node)[0] == 1:
    pass
  else:
    _cell_face_sign = vs.sign(cell_face)
    _cell_face = abs(cell_face) - PT.Element.Size(nface_node)
    cell_face = _cell_face * _cell_face_sign

  face_cell = PDM_dcellface_to_dfacecell(comm, face_distri, cell_distri, cell_face)
  # Strangely PDM can return negative indices if face has only a right parent
  face_cell = abs(face_cell)
  np_utils.shift_nonzeros(face_cell, PT.Element.Range(nface_node)[0]-1) # Refer to NFace global ids

  pe = np.empty((ngon_distri[1] - ngon_distri[0], 2), dtype=face_cell.dtype, order='F')
  pe[:,0] = face_cell[0::2]
  pe[:,1] = face_cell[1::2]

  PT.new_DataArray('ParentElements', pe, parent=ngon_node)
  if remove_NFace:
    PT.rm_child(zone, nface_node)


def ngon_to_edge_pe(zone, comm, remove_NGon=False):
  """Create a ParentElements node in the EdgeElements node from a NGon node.

  Note that EdgeElement is supposed to exists and define all (including internal)
  edges. This function retrieve the link between these edges and the NGon node.

  Input tree is modified inplace.

  Args:
    zone         (CGNSTree): Distributed zone
    comm         (MPIComm) : MPI communicator
    remove_NGon (bool, optional): If True, remove the NGon node.
      Defaults to False.
  """

  # EDGE Data
  edge_node  = MT.Zone.EdgeNode(zone)
  dedge_vtx = PT.get_child_from_name(edge_node, 'ElementConnectivity')[1]
  key_from_edge = dedge_vtx[0::2] + dedge_vtx[1::2] - 1 # (GlobalIndexer starts at 0)

  # NGON Data
  ngon_node = PT.Zone.NGonNode(zone)
  distri_face = MT.getDistribution(ngon_node, 'Element')[1]
  face_vtx = MT.Element.connectivity(ngon_node)

  first_vtx  = face_vtx
  second_vtx = vs.roll(first_vtx, -1, vs.INNER_AXIS)
  key_from_face = first_vtx.values + second_vtx.values
  start_gnum = distri_face[0] + PT.Element.Range(ngon_node)[0]
  face_gnum = np_utils.repeated_arange(face_vtx.counts, start_gnum, dtype=face_vtx.dtype)

  # Now do the search in // using key
  # First : gather data from face into a block vision
  distri = par_utils.distribution_from_gnum([key_from_face], comm, full=True)
  GI = EP.GlobalIndexer(distri, key_from_face-1, comm)
  stride_one = np.ones(key_from_face.size, np.int32)

  stride, data1 = GI.Put_v((stride_one, face_gnum), append=True)
  stride, data2 = GI.Put_v((stride_one, first_vtx.values), append=True)
  # We don't need to exchange second vertex because we know key and vtx1 (vtx1 + vtx2 == key)
  dist_data = {'FaceGnum' : data1, 'FirstVtx' : data2}

  # Second : get data from block, for each edge
  recv_stride, recv_data = EP.block_to_part_strided(stride, dist_data, distri, key_from_edge, comm)
  first_vtx  = recv_data['FirstVtx']
  face_gnum  = recv_data['FaceGnum']


  # Third: post treat (solving conflits) for fill edge_face
  dn_edge = dedge_vtx.size // 2
  edge_face = np.zeros((dn_edge, 2), order='F', dtype=dedge_vtx.dtype)

  # Id of edge, with repetitions eg. if stride == [1,1,2,1], iedge == [0,1,2,2,3]
  iedge_extended = np_utils.repeated_arange(recv_stride)

  # Test if vertex of each edges is equal to recv face_first_vtx, because we can same
  # key for several edges pairs
  first_vtx_match  = dedge_vtx[2*iedge_extended  ] == first_vtx
  second_vtx_match = dedge_vtx[2*iedge_extended+1] == first_vtx
  # Fill edge_face with matching face_gnum
  edge_face[iedge_extended[first_vtx_match],  0] = face_gnum[first_vtx_match]
  edge_face[iedge_extended[second_vtx_match], 1] = face_gnum[second_vtx_match]


  PT.new_DataArray('ParentElements', edge_face, parent=edge_node)
  if remove_NGon:
    PT.rm_child(zone, ngon_node)

def edge_pe_to_ngon(zone, comm, remove_PE=False):
  """Create a NGon node from a Edge node with ParentElements.

  Edge range is supposed to start at 1.
  Input tree is modified inplace.

  Args:
    zone       (CGNSTree): Distributed zone
    comm       (MPIComm) : MPI communicator
    remove_PE  (bool, optional): If True, remove the ParentElements node.
      Defaults to False.
  """

  edge_node = MT.Zone.EdgeNode(zone)
  edge_distri = MT.getDistribution(edge_node, 'Element')[1]
  edge_distri = par_utils.partial_to_full_distribution(edge_distri, comm)
  ngon_distri = MT.getDistribution(zone, 'Cell')[1] # ngon = face = cell in tree
  face_distri = par_utils.partial_to_full_distribution(ngon_distri, comm)
  assert PT.Element.Range(edge_node)[0] == 1
  local_pe = indexing.get_pe_local(edge_node).reshape(-1, order='C')
  edge_vtx = PT.get_child_from_name(edge_node, 'ElementConnectivity')[1]

  face_edge = PDM_dfacecell_to_dcellface(comm, edge_distri, face_distri, local_pe)
  face_vtx = PDM_dfacevtx_from_face_and_edge(comm, face_distri, edge_distri, face_edge, edge_vtx)
  face_vtx_range  = np.array([1, PT.Zone.n_cell(zone)], zone[1].dtype) + PT.Element.Range(edge_node)[1] #n_cell = n_face
  ngon_ec_distr_f = par_utils.gather_and_shift(face_edge.dsize, comm)
  ngon_ec_distri  = par_utils.full_to_partial_distribution(ngon_ec_distr_f, comm)
  ngon_ec_distri  = np_utils.safe_int_cast(ngon_ec_distri, ngon_distri.dtype)
  eso = face_edge.displs + ngon_ec_distri[0]

  ngon = PT.new_NGonElements(erange=face_vtx_range, eso=eso, ec=face_vtx.values, parent=zone)
  MT.newDistribution({"Element" : ngon_distri, "ElementConnectivity" : ngon_ec_distri}, ngon)

  if remove_PE:
    PT.rm_children_from_name(edge_node, "ParentElements")
