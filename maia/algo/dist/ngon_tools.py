import numpy as np

import Pypdm.Pypdm as PDM

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.algo     import indexing
from maia.transfer import protocols as EP
from maia.utils    import par_utils, np_utils

def PDM_dfacecell_to_dcellface(comm, face_distri, cell_distri, face_cell):
  _face_distri = np_utils.safe_int_cast(face_distri, PDM.npy_pdm_gnum_dtype)
  _cell_distri = np_utils.safe_int_cast(cell_distri, PDM.npy_pdm_gnum_dtype)
  _dface_cell  = np_utils.safe_int_cast(face_cell, PDM.npy_pdm_gnum_dtype)
  _cell_face_idx, _cell_face = PDM.dfacecell_to_dcellface(comm, _face_distri, _cell_distri, _dface_cell)
  cell_face_idx = np_utils.safe_int_cast(_cell_face_idx, face_cell.dtype)
  cell_face     = np_utils.safe_int_cast(_cell_face, face_cell.dtype)
  return cell_face_idx, cell_face

def PDM_dcellface_to_dfacecell(comm, face_distri, cell_distri, cell_face_idx, cell_face):
  _face_distri   = np_utils.safe_int_cast(face_distri, PDM.npy_pdm_gnum_dtype)
  _cell_distri   = np_utils.safe_int_cast(cell_distri, PDM.npy_pdm_gnum_dtype)
  _cell_face_idx = np_utils.safe_int_cast(cell_face_idx, np.int32)
  _cell_face     = np_utils.safe_int_cast(cell_face, PDM.npy_pdm_gnum_dtype)
  _face_cell = PDM.dcellface_to_dfacecell(comm, _face_distri, _cell_distri, _cell_face_idx, _cell_face)
  return np_utils.safe_int_cast(_face_cell, cell_face.dtype)


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
  local_pe = indexing.get_ngon_pe_local(ngon_node).reshape(-1, order='C')

  cell_face_idx, cell_face = PDM_dfacecell_to_dcellface(comm, face_distri, cell_distri, local_pe)
  cell_face_range  = np.array([1, PT.Zone.n_cell(zone)], zone[1].dtype) + PT.Zone.n_face(zone)
  nface_ec_distr_f = par_utils.gather_and_shift(cell_face_idx[-1], comm)
  nface_ec_distri  = par_utils.full_to_partial_distribution(nface_ec_distr_f, comm)
  nface_ec_distri  = np_utils.safe_int_cast(nface_ec_distri, nface_distri.dtype)
  eso = cell_face_idx + nface_ec_distri[0]

  nface = PT.new_NFaceElements(erange=cell_face_range, eso=eso, ec=cell_face, parent=zone)
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
  nface_distri_c = MT.getDistribution(nface_node, 'ElementConnectivity')[1]

  face_distri = par_utils.partial_to_full_distribution(ngon_distri, comm)
  cell_distri = par_utils.partial_to_full_distribution(nface_distri, comm)
  cell_face_idx = PT.get_child_from_name(nface_node, "ElementStartOffset")[1]
  cell_face     = PT.get_child_from_name(nface_node, "ElementConnectivity")[1]
  
  # If NFace are before NGon, then face ids must be shifted
  if PT.Element.Range(ngon_node)[0] == 1:
    _cell_face = cell_face
  else:
    _cell_face_sign = np.sign(cell_face)
    _cell_face = np.abs(cell_face) - PT.Element.Size(nface_node)
    _cell_face = _cell_face * _cell_face_sign
  _cell_face_idx = cell_face_idx - nface_distri_c[0] #Go to local idx

  face_cell = PDM_dcellface_to_dfacecell(comm, face_distri, cell_distri, _cell_face_idx, _cell_face)
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
  key_from_edge = dedge_vtx[0::2] + dedge_vtx[1::2]

  # NGON Data
  ngon_node = PT.Zone.NGonNode(zone)
  distri_face = MT.getDistribution(ngon_node, 'Element')[1]
  face_vtx     = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
  face_vtx_idx = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
  face_vtx_idx = face_vtx_idx - face_vtx_idx[0]

  first_vtx  = face_vtx
  second_vtx = np_utils.roll_by_stride(face_vtx_idx, face_vtx)
  key_from_face = first_vtx + second_vtx
  start_gnum = distri_face[0] + PT.Element.Range(ngon_node)[0]
  end_gnum   = distri_face[1] + PT.Element.Range(ngon_node)[0]
  face_gnum = np.repeat(np.arange(start_gnum, end_gnum, dtype=face_vtx.dtype), np.diff(face_vtx_idx))

  # Now do the search in // using key
  # First : gather data from face into a block vision
  ptb = EP.PartToBlock(None, [key_from_face], comm, keep_multiple=True)
  stride_one = np.ones(key_from_face.size, np.int32)

  stride, data2 = ptb.exchange_field([first_vtx], [stride_one])
  stride, data3 = ptb.exchange_field([second_vtx], [stride_one])
  stride, data1 = ptb.exchange_field([face_gnum], [stride_one])
  dist_data = {'FaceGnum' : data1, 'FirstEdge' : data2, 'SecondEdge' : data3}
  ptb_distri = ptb.getDistributionCopy()
  # Adapt to full block, since some gnum does not appear
  fstride = np.zeros(ptb_distri[comm.rank+1] - ptb_distri[comm.rank], np.int32)
  fstride[ptb.getBlockGnumCopy() - ptb_distri[comm.rank] - 1] = stride
  
  # Second : get data from block, for each edge 
  recv_stride, recv_data = EP.block_to_part_strided(fstride, dist_data, ptb_distri, [key_from_edge], comm)
  recv_stride = recv_stride[0]
  first_vtx  = recv_data['FirstEdge'][0]
  second_vtx = recv_data['SecondEdge'][0]
  face_gnum  = recv_data['FaceGnum'][0]

  # Third: post treat (solving conflits) for fill edge_face
  dn_edge = dedge_vtx.size // 2
  edge_face = np.zeros((dn_edge, 2), order='F', dtype=dedge_vtx.dtype)
  read_idx = 0
  for iedge in range(dn_edge):
    for _ in range(recv_stride[iedge]):
      if dedge_vtx[2*iedge] == first_vtx[read_idx] and dedge_vtx[2*iedge+1] == second_vtx[read_idx]:
        edge_face[iedge,0] = face_gnum[read_idx]
      elif dedge_vtx[2*iedge] == second_vtx[read_idx] and dedge_vtx[2*iedge+1] == first_vtx[read_idx]:
        edge_face[iedge,1] = face_gnum[read_idx]
      else:
        pass # Fake positive
      read_idx += 1
  assert read_idx == recv_stride.sum()

  PT.new_DataArray('ParentElements', edge_face, parent=edge_node)
  if remove_NGon:
    PT.rm_child(zone, ngon_node)