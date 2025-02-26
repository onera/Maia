import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.utils                import vstride as vs
from maia.algo.part.ngon_tools import pe_to_nface
from maia.utils import s_numbering

import Pypdm.Pypdm as PDM

def PDM_combine_connectivity(first:vs.VStrideArray, second:vs.VStrideArray):
  return vs.from_displs(*PDM.combine_connectivity(first.displs, first.values,
                                                  second.displs, second.values))

def PDM_connectivity_transpose(n_opp:int, connec:vs.VStrideArray):                                             
  return vs.from_displs(*PDM.connectivity_transpose(int(n_opp), connec.displs, connec.values))


def cell_vtx_connectivity_S(zone_S, dim) :
    n_cell = PT.Zone.n_cell(zone_S)
    vertex_size = PT.Zone.VertexSize(zone_S)
    
    if dim == 2:
      cell_vtx_idx = 4*np.arange(0, n_cell+1, dtype=np.int32)
      cell_vtx = np.zeros(4*n_cell, dtype=np.int32)
      i = np.arange(1, vertex_size[0])
      j = np.arange(1, vertex_size[1]).reshape(-1,1)
      cell_vtx[0::4] = s_numbering.ij_to_index(i,   j,   vertex_size).flatten()
      cell_vtx[1::4] = s_numbering.ij_to_index(i+1, j,   vertex_size).flatten()
      cell_vtx[2::4] = s_numbering.ij_to_index(i+1, j+1, vertex_size).flatten()
      cell_vtx[3::4] = s_numbering.ij_to_index(i,   j+1, vertex_size).flatten()
    elif dim == 3:
      cell_vtx_idx = 8*np.arange(0, n_cell+1, dtype=np.int32)
      cell_vtx = np.zeros(8*n_cell, dtype=np.int32)
      i = np.arange(1, vertex_size[0])
      j = np.arange(1, vertex_size[1]).reshape(-1,1)
      k = np.arange(1, vertex_size[2]).reshape(-1,1,1)
      cell_vtx[0::8] = s_numbering.ijk_to_index(i,   j,   k,   vertex_size).flatten()
      cell_vtx[1::8] = s_numbering.ijk_to_index(i+1, j,   k,   vertex_size).flatten()
      cell_vtx[2::8] = s_numbering.ijk_to_index(i+1, j+1, k,   vertex_size).flatten()
      cell_vtx[3::8] = s_numbering.ijk_to_index(i,   j+1, k,   vertex_size).flatten()
      cell_vtx[4::8] = s_numbering.ijk_to_index(i,   j,   k+1, vertex_size).flatten()
      cell_vtx[5::8] = s_numbering.ijk_to_index(i+1, j,   k+1, vertex_size).flatten()
      cell_vtx[6::8] = s_numbering.ijk_to_index(i+1, j+1, k+1, vertex_size).flatten()
      cell_vtx[7::8] = s_numbering.ijk_to_index(i,   j+1, k+1, vertex_size).flatten()
    else:
      raise NotImplementedError("Unsupported dimension")

    return vs.from_displs(cell_vtx_idx, cell_vtx)

def cell_vtx_connectivity_ngon(zone, dim):
  if dim==1:
    raise NotImplementedError("U-NGON meshes doesn't support dimension 1 elements")

  ngon_node = PT.Zone.NGonNode(zone)
  face_vtx = MT.Element.connectivity(ngon_node)
  
  if dim==2:
    return face_vtx

  if not PT.Zone.has_nface_elements(zone):
    pe_to_nface(zone)

  nface_node = PT.Zone.NFaceNode(zone)
  cell_face = MT.Element.connectivity(nface_node)

  return PDM_combine_connectivity(cell_face, face_vtx)

def cell_vtx_connectivity_elts(zone, dim):
  ordered_elts = PT.Zone.get_ordered_elements_per_dim(zone)
  connectivities = [PT.get_child_from_name(e, 'ElementConnectivity')[1] for e in ordered_elts[dim]]
  n_elts = sum([PT.Element.Size(e) for e in ordered_elts[dim]])
  cell_vtx = vs.array(connectivities, dtype=np.int32) # For now, strides are the size of sections
  cell_vtx_idx = np.empty(n_elts+1, np.int32)
  cell_vtx_idx[0] = 0

  cur = 1
  for elt in ordered_elts[dim]:
    cell_vtx_idx[cur:cur+PT.Element.Size(elt)] = \
      PT.Element.NVtx(elt) * np.arange(1, PT.Element.Size(elt)+1, dtype=np.int32) + cell_vtx_idx[cur-1]
    cur += PT.Element.Size(elt)
  assert cur == n_elts +1

  cell_vtx.restride(displs=cell_vtx_idx) # Registrer true cell_vtx_idx

  return cell_vtx

def cell_vtx_connectivity(zone, dim=3, elts_subset=None):
  """
  Compute and return the cell->vtx connectivity on a partitioned zone

  If elts_subset is None, cell_vtx connectivity is computed for all elements of the zone.
  Otherwise, a 2d numpy array of element indices (in absolute numbering) must be provided;
  cell_vtx connectivity for the requested indices (only U meshes)
  """
  assert dim in [1,2,3]
  assert PT.Zone.Type(zone) in ['Structured', 'Unstructured']
  
  if PT.Zone.Type(zone) == 'Structured':
    cell_vtx = cell_vtx_connectivity_S(zone, dim)
  else:
    if PT.Zone.has_ngon_elements(zone):
      if dim == 1:
        cell_vtx = cell_vtx_connectivity_elts(zone, dim)
      else:
        cell_vtx = cell_vtx_connectivity_ngon(zone, dim)
    else: # zone has standard elements
      cell_vtx = cell_vtx_connectivity_elts(zone, dim)
  
  if elts_subset is not None:
    assert PT.Zone.Type(zone) == 'Unstructured'
    offset = PT.Zone.get_elt_range_per_dim(zone)[dim][0]
    _elts_ids = elts_subset[0] - offset
    cell_vtx = vs.take(cell_vtx, _elts_ids)

  return cell_vtx
