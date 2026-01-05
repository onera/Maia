import maia.pytree      as PT
import maia.pytree.maia as MT

import maia
import numpy as np
from maia.transfer import protocols as EP
from maia.typing import *
from maia.utils import vstride as vs
from maia.utils import par_utils

from .renumber import renumber_vertices, renumber_edges, renumber_faces


def flag_to_old2new(flag, comm, dtype):
  # Compute old to new table, putting elts for which flag is True first
  true_n  = flag.sum()
  false_n = flag.size - true_n
  true_offset  = par_utils.gather_and_shift(true_n, comm)
  false_offset = par_utils.gather_and_shift(false_n, comm)

  old_to_new = np.empty(flag.size, dtype)
  old_to_new[ flag] = np.arange(true_n)  + true_offset[comm.rank]
  old_to_new[~flag] = np.arange(false_n) + false_offset[comm.rank] + true_offset[comm.size]

  return old_to_new

def put_boundary_first(t:CGNSDistTree, comm:MPIComm):
  for zone_path in PT.predicates_to_paths(t, 'CGNSBase_t/Zone_t'):
    zone  = PT.find_node_from_path(t, zone_path)
    ztype = PT.get_np_value(zone).dtype

    if PT.Zone.Type(zone) != 'Unstructured':
      continue # Nothing to do for unstructured zones

    assert (zone_dim := PT.Zone.CellDimension(zone)) >= 2
    needs_vtx_ordering = PT.Zone.n_vtx_bnd(zone) == 0

    assert PT.Zone.CellDimension(zone) >= 2
    is_poly3d = PT.pred.IS_POLY3D_ZONE(zone)
    is_poly2d = PT.pred.IS_POLY2D_ZONE(zone)
    is_poly = is_poly2d or is_poly3d

    if is_poly:
      # Boundary elts are : face (ngon) if 3D, edges (bar) if 2D
      # If we have an indication of boundary elements in elt node, use it;
      # Otherwise, detect it with ParentElements
      node_fn = MT.Zone.EdgeNode        if is_poly2d else PT.Zone.NGonNode
      pe_fn = maia.algo.ngon_to_edge_pe if is_poly2d else maia.algo.nface_to_pe
      renum_fn = renumber_edges         if is_poly2d else renumber_faces

      node = node_fn(zone)
      elt_vtx = MT.Element.connectivity(node)

      if (n_bnd_elts:=PT.get_np_value(node)[1]) != 0:
        elt_distri = MT.distribution_value(node, 'Element')
        end = max(min(elt_distri[1], n_bnd_elts) - elt_distri[0], 0) # Get local bound
        elt_vtx.displs[end]
        bnd_vertices = elt_vtx.values[:elt_vtx.displs[end]]
      else:
        pe_fn(zone, comm) # Ensure presence of ParentElements
        pe = PT.get_np_value(PT.find_child_from_name(node, 'ParentElements'))
        is_bnd_elt = np.zeros(MT.Element.dn_elt(node), bool)
        is_bnd_elt[pe.min(axis=1)==0] = True
        bnd_elt_vtx = vs.take(elt_vtx, np.flatnonzero(is_bnd_elt))
        bnd_vertices = np.unique(bnd_elt_vtx.values)

        # Reorder faces
        new_face_id = flag_to_old2new(is_bnd_elt, comm, ztype)
        renum_fn(t, zone_path, new_face_id, comm)
        PT.get_np_value(node)[1] = comm.allreduce(is_bnd_elt.sum())

    else:
      # STD elts case
      bnd_vertices = list()
      elts_2d = PT.Zone.get_ordered_elements_per_dim(zone)[zone_dim-1]
      for elt in elts_2d:
        ec = PT.get_np_value(PT.find_child_from_name(elt, 'ElementConnectivity'))
        # If we have an indication of boundary elements in elt node, use it;
        # Otherwise, we assume that only BND faces are described in 2D elt sections
        # Better approch could be:
        #   - decompose volumic elts into faces and search faces appearing twice (~elt->ng)
        #   - search faces appearing in BC/GC
        if (n_bnd_elts:=PT.get_np_value(elt)[1]) != 0:
          elt_distri = MT.distribution_value(elt, 'Element')
          end = max(min(elt_distri[1], n_bnd_elts) - elt_distri[0], 0) # Get local bound
          bnd_vertices.append(ec[0:end*PT.Element.NVtx(elt)]) # BND elements are first
        else:
          bnd_vertices.append(ec) # All faces considered BND

    if needs_vtx_ordering:
      # Reorder vertices
      vtx_distri = MT.distribution_value(zone, 'Vertex')
      GI = EP.GlobalIndexer(vtx_distri, bnd_vertices, comm, gnum_offset=1)
      is_bnd_vtx = GI.access_counts > 0
      new_vtx_id = flag_to_old2new(is_bnd_vtx, comm, ztype)
      renumber_vertices(t, zone_path, new_vtx_id, comm)

      # Update zone val
      zval = PT.get_np_value(zone)
      zval[0,2] = comm.allreduce(is_bnd_vtx.sum())
