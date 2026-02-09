import numpy as np

import Pypdm.Pypdm as PDM

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia import npy_pdm_gnum_dtype as pdm_dtype

from maia.algo      import indexing
from maia.transfer  import protocols  as EP
from maia.utils     import np_utils, par_utils, s_numbering
from maia.utils     import vstride as vs

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
  
  GI = EP.GlobalIndexer(edge_distrib, np.abs(face_edge)-1, comm)
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

def cell_vtx_connectivity_S(zone_S, dim, cell_subset=None):
  # NB this is not factorised with part.connectivity_utils because arrays layout seems different
  # Maybe we could merge it 
  vertex_size = PT.Zone.VertexSize(zone_S)
  cell_distri = MT.Zone.cell_distribution(zone_S)

  if cell_subset is not None:
    # cell_i, cell_j and cell_k are provided
    if dim == 1:
      cell_i = cell_subset
    elif dim == 2:
      cell_i, cell_j = cell_subset
    elif dim == 3:
      cell_i, cell_j, cell_k = cell_subset
  else:
    # Compute cell_i, cell_j, cell_k for all cells of the mesh (distributed)
    cell_idx = np.arange(cell_distri[0]+1, cell_distri[1]+1, dtype=zone_S[1].dtype) # Distributed view of cells, as idx  
    if dim == 1:
      cell_i = cell_idx
    elif dim == 2:
      cell_i, cell_j = s_numbering.index_to_ij(cell_idx, PT.Zone.CellSize(zone_S))
    elif dim == 3:
      cell_i, cell_j, cell_k = s_numbering.index_to_ijk(cell_idx, PT.Zone.CellSize(zone_S))

  dn_cell  = cell_i.size

  if dim == 1:
    cell_vtx = np.zeros(2*dn_cell, zone_S[1].dtype)
    cell_vtx_idx = 2*np.arange(0, dn_cell+1, dtype=np.int32)
    cell_vtx[0::2] = cell_i
    cell_vtx[1::2] = cell_i+1
  elif dim == 2:
    cell_vtx = np.zeros(4*dn_cell, zone_S[1].dtype)
    cell_vtx_idx = 4*np.arange(0, dn_cell+1, dtype=np.int32)
    cell_vtx[0::4] = s_numbering.ij_to_index(cell_i,   cell_j,   vertex_size).flatten()
    cell_vtx[1::4] = s_numbering.ij_to_index(cell_i+1, cell_j,   vertex_size).flatten()
    cell_vtx[2::4] = s_numbering.ij_to_index(cell_i+1, cell_j+1, vertex_size).flatten()
    cell_vtx[3::4] = s_numbering.ij_to_index(cell_i,   cell_j+1, vertex_size).flatten()
  elif dim == 3:
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

  return vs.from_displs(cell_vtx_idx, cell_vtx)

def combine_dconnectivity(distri1, distri2, cnt1, cnt2, keep_sign, comm):
  """
  Compute strided connectivity A->C from A->B + B->C
  If keep sign is True, sign of A->B is reported on output
  """
  # 1. For values of A->B, get corresponding values in B->C
  #    so we have A->C but with reps. and without sign
  cnt3 = EP.block_to_part(cnt2, distri2, np.abs(cnt1.values)-1, comm)

  # 2. Report sign of A->C if needed. The sign extends to all the 
  #   'C' elements coming from a same 'B' elt
  if keep_sign:
    cnt3 *= np.sign(cnt1.values)

  # 3. Compute new index of A->C by summing the counts of items coming from each B
  idx = vs.from_displs(cnt1.displs, cnt3.counts).reduce(vs.ReduceOp.SUM)
  # 4. Make A->C elts unique
  out = vs.unique(vs.from_counts(idx, cnt3.values), vs.INNER_AXIS)
  return out


def cell_vtx_connectivity_ngon(zone, comm, cell_subset=None):
  """
  Return cell_vtx connectivity for an input NGON Zone
  """
  assert PT.Zone.Type(zone) == "Unstructured" and PT.Zone.CellDimension(zone) == 3
  if PT.Zone.has_ngon_elements(zone):
    ngon_node = PT.Zone.NGonNode(zone)
    face_distri   = MT.Element.distribution(ngon_node)
    _face_distri  = par_utils.partial_to_full_distribution(face_distri, comm)
    face_vtx = MT.Element.connectivity(ngon_node)
    if PT.Zone.has_nface_elements(zone):
      nface_node = PT.Zone.NFaceNode(zone)
      cell_face = MT.Element.connectivity(nface_node)
      cell_distri    = MT.Element.distribution(nface_node)
      _cell_distri   = par_utils.partial_to_full_distribution(cell_distri, comm)

    else:
      assert PT.Element.Range(ngon_node)[0] == 1
      local_pe = indexing.get_pe_local(ngon_node).reshape(-1, order='C')
      cell_distri   = MT.Zone.cell_distribution(zone)
      _cell_distri  = par_utils.partial_to_full_distribution(cell_distri, comm)
      cell_face = PDM_dfacecell_to_dcellface(comm, _face_distri, _cell_distri, local_pe)

    cell_vtx = combine_dconnectivity(_cell_distri, _face_distri, cell_face, face_vtx, False, comm)

    if cell_subset is not None:
      _cell_subset = cell_subset - PT.Zone.get_elt_range_per_dim(zone)[3][0]
      cell_vtx = EP.block_to_part(cell_vtx, _cell_distri, _cell_subset, comm)
  else:
    raise NotImplementedError("Only NGON zones are managed")

  return cell_vtx


def entity_vtx_connectivity_elt(zone, comm, dim, distri_global, elts_subset=None):
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
  all_cell_vtx_vs = []
  all_elt_gnum = []

  if elts_subset is not None:
    distri_global = False

  if distri_global:
    assert PT.Zone.CellDimension(zone) == dim, "Redispatch only supported for native cell dimension"
    distri_cell = MT.Zone.cell_distribution(zone)
    start = 0

  for elt in PT.Zone.get_ordered_elements_per_dim(zone)[dim]:
    distri = MT.Element.distribution(elt)
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
      start = end

    all_cell_vtx_vs.append(vs.from_counts(np.int32(PT.Element.NVtx(elt)), ec))
    if elts_subset is not None:
      all_elt_gnum.append(np.arange(distri[0], distri[1], dtype=pdm_dtype) + PT.Element.Range(elt)[0])

  if elts_subset is not None:
    cell_vtx_n, cell_vtx = EP.part_to_part_strided([a.counts for a in all_cell_vtx_vs], [a.values for a in all_cell_vtx_vs], all_elt_gnum, [elts_subset], comm)
    cell_vtx = vs.from_counts(cell_vtx_n[0], cell_vtx[0])
  elif len(all_cell_vtx_vs):
    cell_vtx = vs.concatenate(all_cell_vtx_vs, vs.OUTER_AXIS)
  else: 
    cell_vtx = vs.from_displs(np.array([0],dtype=np.int32), np.array([],np.int32))

  return cell_vtx
