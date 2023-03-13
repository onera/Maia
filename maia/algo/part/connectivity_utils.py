import numpy as np

import maia.pytree as PT

from maia.utils                import np_utils
from maia.algo.part.ngon_tools import pe_to_nface

import Pypdm.Pypdm as PDM

def cell_vtx_connectivity(zone, dim=3):
  """
  Compute and return the cell->vtx connectivity on a partitioned zone
  """
  if PT.Zone.Type(zone) == 'Structured':
    raise NotImplementedError("Structured zones are not supported")
  else:
    if PT.Zone.has_ngon_elements(zone):
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
      _, cell_vtx = np_utils.concatenate_np_arrays(connectivities)
      cell_vtx_idx = np.empty(n_elts+1, np.int32)

      cur = 0
      for i, elt in enumerate(ordered_elts[dim]):
        if i == 0:
          cell_vtx_idx[0:PT.Element.Size(elt)+1] = \
            PT.Element.NVtx(elt) * np.arange(PT.Element.Size(elt)+1, dtype=np.int32)
        else:
          cell_vtx_idx[cur:cur+PT.Element.Size(elt)] = \
            PT.Element.NVtx(elt) * np.arange(1, PT.Element.Size(elt)+1, dtype=np.int32) + cell_vtx_idx[cur-1]
        cur += PT.Element.Size(elt) + (i==0) #One more for first elt
      assert cur == n_elts+1

  return cell_vtx_idx, cell_vtx

