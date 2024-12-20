import numpy as np

import Pypdm.Pypdm as PDM

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.algo      import indexing
from maia.transfer  import protocols  as EP
from maia.utils     import np_utils, par_utils, s_numbering, as_pdm_gnum

from .ngon_tools    import PDM_dfacecell_to_dcellface


def combine_face_edge_and_edge_vtx(face_edge_idx, face_edge, edge_distrib, edge_vtx, comm):
  """
  Compute face_vtx connectivity from face_edge and edge_vtx connectivities in
  a distributed context

  Args:
    face_edge_idx (array)  : Face to edge connectivity index
    face_edge     (array)  : Face to edge connectivity
    edge_distrib  (array)  : Edge distribution
    edge_vtx      (array)  : Edge to vertex connectivity
    comm          (MPIComm): MPI communicator
  """
  face_edge_idx = np_utils.safe_int_cast(face_edge_idx - face_edge_idx[0], np.int32)
  
  edge_distrib_f = par_utils.partial_to_full_distribution(edge_distrib, comm)
  GI = EP.GlobalIndexer(edge_distrib_f, np.abs(face_edge)-1, comm)
  global_edge_vtx = GI.Take(edge_vtx, count=2)
  
  # Convert in local numbering to be allowed to use part algo in dist context
  # We basically redefine all edges (internal edges are defined twice) instead of creating 
  # a proper subnumbering, it is ok because then we extract the resulting vertices
  uniq, idx, inv = np.unique(global_edge_vtx, return_index=True, return_inverse=True)
  local_edge_vtx = np_utils.safe_int_cast(inv+1, np.int32)
  
  local_face_edge = np.sign(face_edge, dtype=np.int32) * np.arange(1, face_edge.size+1, dtype=np.int32)
  
  local_face_vtx = PDM.compute_face_vtx_from_face_and_edge(face_edge_idx,
                                                           local_face_edge,
                                                           local_edge_vtx)
  
  # Return in the global numbering
  global_face_vtx = global_edge_vtx[idx][local_face_vtx-1]
  
  return global_face_vtx

def cell_vtx_connectivity_S(zone_S, dim):
  # NB this is not factorised with part.connectivity_utils because arrays layout seems different
  # Maybe we could merge it 
  vertex_size = PT.Zone.VertexSize(zone_S)
  cell_distri = MT.getDistribution(zone_S, 'Cell')[1]

  cell_idx = np.arange(cell_distri[0]+1, cell_distri[1]+1, dtype=zone_S[1].dtype) # Distributed view of cells, as idx  
  dn_cell  = cell_idx.size

  if dim == 2:
    cell_i, cell_j = s_numbering.index_to_ij(cell_idx, PT.Zone.CellSize(zone_S))
    cell_vtx = np.zeros(4*dn_cell, zone_S[1].dtype)
    cell_vtx_idx = 4*np.arange(0, dn_cell+1, dtype=np.int32)
    cell_vtx[0::4] = s_numbering.ij_to_index(cell_i,   cell_j,   vertex_size).flatten()
    cell_vtx[1::4] = s_numbering.ij_to_index(cell_i+1, cell_j,   vertex_size).flatten()
    cell_vtx[2::4] = s_numbering.ij_to_index(cell_i+1, cell_j+1, vertex_size).flatten()
    cell_vtx[3::4] = s_numbering.ij_to_index(cell_i,   cell_j+1, vertex_size).flatten()
  elif dim == 3:
    cell_i, cell_j, cell_k = s_numbering.index_to_ijk(cell_idx, PT.Zone.CellSize(zone_S))
    cell_vtx = np.zeros(8*dn_cell, zone_S[1].dtype)
    cell_vtx_idx = 8*np.arange(0, dn_cell+1, dtype=np.int32)
    cell_vtx[0::8] = s_numbering.ijk_to_index(cell_i,   cell_j,   cell_k,   vertex_size).flatten()
    cell_vtx[1::8] = s_numbering.ijk_to_index(cell_i+1, cell_j,   cell_k,   vertex_size).flatten()
    cell_vtx[2::8] = s_numbering.ijk_to_index(cell_i+1, cell_j+1, cell_k,   vertex_size).flatten()
    cell_vtx[3::8] = s_numbering.ijk_to_index(cell_i,   cell_j+1, cell_k,   vertex_size).flatten()
    cell_vtx[4::8] = s_numbering.ijk_to_index(cell_i,   cell_j,   cell_k+1, vertex_size).flatten()
    cell_vtx[5::8] = s_numbering.ijk_to_index(cell_i+1, cell_j,   cell_k+1, vertex_size).flatten()
    cell_vtx[6::8] = s_numbering.ijk_to_index(cell_i+1, cell_j+1, cell_k+1, vertex_size).flatten()
    cell_vtx[7::8] = s_numbering.ijk_to_index(cell_i,   cell_j+1, cell_k+1, vertex_size).flatten()

  return cell_vtx_idx, cell_vtx

def cell_vtx_connectivity_ngon(zone, comm):
  """
  Return cell_vtx connectivity for an input NGON Zone
  """
  assert PT.Zone.Type(zone) == "Unstructured" and PT.Zone.CellDimension(zone) == 3
  if PT.Zone.has_ngon_elements(zone):
    ngon_node = PT.Zone.NGonNode(zone)
    face_vtx      = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
    face_vtx_idx  = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
    face_distri   = MT.get_distribution(ngon_node, 'Element')[1]
    _face_distri  = par_utils.partial_to_full_distribution(face_distri, comm)
    _face_vtx_idx = np.empty(face_vtx_idx.size, np.int32)
    np.subtract(face_vtx_idx, face_vtx_idx[0], out=_face_vtx_idx)
    if PT.Zone.has_nface_elements(zone):
      nface_node = PT.Zone.NFaceNode(zone)
      cell_face      = PT.get_child_from_name(nface_node, 'ElementConnectivity')[1]
      cell_distri    = MT.get_distribution(nface_node, 'Element')[1]
      _cell_distri   = par_utils.partial_to_full_distribution(cell_distri, comm)
      cell_face_idx  = PT.get_child_from_name(nface_node, 'ElementStartOffset')[1]
      _cell_face_idx = np.empty(cell_face_idx.size, np.int32)
      np.subtract(cell_face_idx, cell_face_idx[0], out=_cell_face_idx)

    else:
      assert PT.Element.Range(ngon_node)[0] == 1
      local_pe = indexing.get_pe_local(ngon_node).reshape(-1, order='C')
      cell_distri   = MT.get_distribution(zone, 'Cell')[1]
      _cell_distri  = par_utils.partial_to_full_distribution(cell_distri, comm)
      _cell_face_idx, cell_face = PDM_dfacecell_to_dcellface(comm, _face_distri, _cell_distri, local_pe)
      _cell_face_idx = np_utils.safe_int_cast(_cell_face_idx, np.int32)

    cell_vtx_idx, cell_vtx = PDM.dconnectivity_combine(comm, 
                                                      as_pdm_gnum(_cell_distri),
                                                      as_pdm_gnum(_face_distri),
                                                      _cell_face_idx,
                                                      as_pdm_gnum(cell_face),
                                                      _face_vtx_idx,
                                                      as_pdm_gnum(face_vtx),
                                                      False)
  else:
    raise NotImplementedError("Only NGON zones are managed")

  return cell_vtx_idx, cell_vtx


def entity_vtx_connectivity_elt(zone, comm, dim, distri_global):
  """
  Exchange vtx ids to compute the cell_vtx table for a given dimension.
  All elements of same dim are concatenated in output.
  If distrib_global is True, this cell_vtx connectivity is redistributed to match
  the global distribution of all elements of the requested dim
  Otherwise, we just concatenate the data of each section
  Exemple : if distri TETRA = [0,5,9], distri PRISM = [0,3,7] and global distri CELL = [0,8,16]
  with local mode rank 0 get 5 tetra and 3 prism, rank 1 get 4 tetra and 4 prism
  with global mode rank 0 get 8 tetra and rank 1 get 1 tetra and 7 prism
  """
  all_cell_vtx_n = []
  all_cell_vtx = []

  if distri_global:
    assert PT.Zone.CellDimension(zone) == dim, "Redispatch only supported for native cell dimension"
    distri_cell = MT.getDistribution(zone, 'Cell')[1]
    start = 0

  for elt in PT.Zone.get_ordered_elements_per_dim(zone)[dim]:
    distri = MT.get_distribution(elt, 'Element')[1]
    ec = PT.get_child_from_name(elt, 'ElementConnectivity')[1]
    
    if distri_global:
      end = start + PT.Element.Size(elt)
      distri_out = distri.copy()
      # Here we restrict the total cell distribution to ElementRange (ignoring low order elts), 
      # then we shift it to make it start a 0
      distri_out[0] = max(min(distri_cell[0], end), start) - start
      distri_out[1] = max(min(distri_cell[1], end), start) - start
      btb = EP.BlockToBlock(distri, distri_out, comm)
      ec = btb.exchange(ec, PT.Element.NVtx(elt))
      ec_idx = PT.Element.NVtx(elt) * np.ones(distri_out[1] - distri_out[0], np.int32)
      start = end
    else:
      ec_idx = PT.Element.NVtx(elt) * np.ones(distri[1] - distri[0], np.int32)

    all_cell_vtx.append(ec)
    all_cell_vtx_n.append(ec_idx)

  cell_vtx_n = np.concatenate(all_cell_vtx_n, dtype=np.int32)
  cell_vtx = np.concatenate(all_cell_vtx)
  cell_vtx_idx = np_utils.sizes_to_indices(cell_vtx_n)

  return cell_vtx_idx, cell_vtx