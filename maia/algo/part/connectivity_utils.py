import numpy as np

import maia.pytree as PT

from maia.utils                import np_utils
from maia.algo.part.ngon_tools import pe_to_nface
from maia.utils import s_numbering

import Pypdm.Pypdm as PDM

def cell_vtx_connect_2D(zone_S) :

    assert PT.Zone.Type(zone_S)=='Structured'

    n_cell = PT.Zone.n_cell(zone_S)
    vertex_size = PT.Zone.VertexSize(zone_S)
    cell_vtx_idx = 4*np.arange(0, n_cell+1, dtype=np.int32)
    cell_vtx = np.zeros(4*n_cell, dtype=np.int32)
    i = np.arange(1, vertex_size[0])
    j = np.arange(1, vertex_size[1]).reshape(-1,1)
    cell_vtx[0::4] = s_numbering.ijk_to_index(i,   j,   1, vertex_size).flatten()
    cell_vtx[1::4] = s_numbering.ijk_to_index(i+1, j,   1, vertex_size).flatten()
    cell_vtx[2::4] = s_numbering.ijk_to_index(i+1, j+1, 1, vertex_size).flatten()
    cell_vtx[3::4] = s_numbering.ijk_to_index(i,   j+1, 1, vertex_size).flatten()

    return cell_vtx_idx, cell_vtx

def cell_vtx_connect_3D(zone_S) :

    assert PT.Zone.Type(zone_S)=='Structured'

    n_cell = PT.Zone.n_cell(zone_S)
    vertex_size = PT.Zone.VertexSize(zone_S)
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

    return cell_vtx_idx, cell_vtx


def cell_vtx_connectivity(zone, dim=3):
  """
  Compute and return the cell->vtx connectivity on a partitioned zone
  """
  assert dim in [1,2,3]
  
  if PT.Zone.Type(zone) == 'Structured':
    if dim == 2:
      cell_vtx_idx, cell_vtx = cell_vtx_connect_2D(zone)
    elif dim == 3:
      cell_vtx_idx, cell_vtx = cell_vtx_connect_3D(zone)
    else:
      raise NotImplementedError("Unsupported dimension")
  else:
    if PT.Zone.has_ngon_elements(zone):
      if dim==1:
        raise NotImplementedError("U-NGON meshes doesn't support dimension 1 elements")

      ngon_node = PT.Zone.NGonNode(zone)
      ngon_eso = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
      ngon_ec = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
      
      if dim==2:
        return ngon_eso, ngon_ec

      try:
        nface_node = PT.Zone.NFaceNode(zone)
      except RuntimeError:
        pe_to_nface(zone)

      nface_node = PT.Zone.NFaceNode(zone)
      nface_eso = PT.get_child_from_name(nface_node, 'ElementStartOffset')[1]
      nface_ec = PT.get_child_from_name(nface_node, 'ElementConnectivity')[1]

      cell_vtx_idx, cell_vtx = PDM.combine_connectivity(nface_eso, nface_ec, ngon_eso, ngon_ec)

    else: # zone has standard elements
      ordered_elts = PT.Zone.get_ordered_elements_per_dim(zone)
      connectivities = [PT.get_child_from_name(e, 'ElementConnectivity')[1] for e in ordered_elts[dim]]
      n_elts = sum([PT.Element.Size(e) for e in ordered_elts[dim]])
      _, cell_vtx = np_utils.concatenate_np_arrays(connectivities, dtype=np.int32)
      cell_vtx_idx = np.empty(n_elts+1, np.int32)
      cell_vtx_idx[0] = 0

      cur = 1
      for i, elt in enumerate(ordered_elts[dim]):
        cell_vtx_idx[cur:cur+PT.Element.Size(elt)] = \
          PT.Element.NVtx(elt) * np.arange(1, PT.Element.Size(elt)+1, dtype=np.int32) + cell_vtx_idx[cur-1]
        cur += PT.Element.Size(elt)
      assert cur == n_elts +1

  return cell_vtx_idx, cell_vtx

