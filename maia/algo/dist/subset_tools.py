import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

import maia
from maia.transfer import protocols as EP
from maia.utils    import np_utils, par_utils

from maia.utils.parallel import algo as par_algo

from maia.typing import *

def sort_dist_pointlist(subset, comm):
    pl_n = PT.get_child_from_name(subset, 'PointList')
    pld_n = PT.get_child_from_name(subset, 'PointListDonor')

    sorter = par_algo.DistSorter(pl_n[1][0], comm)

    dist_pl = sorter.sort(pl_n[1][0])
    PT.update_child(subset, 'PointList', value=dist_pl.reshape((1,-1), order='F'))

    if pld_n is not None:
      dist_pld = sorter.sort(pld_n[1][0])
      PT.update_child(subset, 'PointListDonor', value=dist_pld.reshape((1,-1), order='F'))

    new_distri = par_utils.dn_to_distribution(dist_pl.size, comm)
    MT.new_Distribution({'Index' : new_distri}, subset)

def vtx_ids_to_face_ids(vtx_ids, elt_n, comm, elt_full):
  """
  From an array of vertex ids, search in the distributed NGon node
  the id of faces constituted by these vertices.
  If elt_full is True, only faces having all their vertices in vtx_ids
  are returned.
  Otherwise, faces having at least one vertex in vtx_ids are returned.
  """
  elt_distri = MT.Element.distribution(elt_n)
  delt_vtx   = PT.get_child_from_name(elt_n, 'ElementConnectivity')[1]
  if PT.Element.Type(elt_n)=='NGON_n':
    delt_vtx_idx = PT.get_child_from_name(elt_n, 'ElementStartOffset')[1]
  else:
    elt_size     = PT.Element.NVtx(elt_n)
    delt_vtx_idx = np.arange(elt_distri[0]*elt_size,(elt_distri[1]+1)*elt_size,elt_size, dtype=np.int32)

  # > Building PTP object, the graph between vtx indices and elt connectivity is what we need
  delt_vtx_tag = par_algo.gnum_isin(delt_vtx, vtx_ids, comm)

  # Then reduce : select face if all its vertices have flag set to 1
  ufunc = np.logical_and if elt_full==True else np.logical_or
  delt_vtx_idx_loc = delt_vtx_idx - delt_vtx_idx[0]
  delt_vtx_tag = ufunc.reduceat(delt_vtx_tag, delt_vtx_idx_loc[:-1])
  face_ids = np.where(delt_vtx_tag)[0] + elt_distri[0] + 1

  return np_utils.safe_int_cast(face_ids, vtx_ids.dtype)

def convert_subset_as_facelist(dist_tree:CGNSDistTree, subset_path:CGNSPath, comm:MPIComm, only_bnd:bool=False):
  """
  Transform Vertex located subsets (as BCs) to Face or Edge located
  subsets, depending on input dim
  """
  node = PT.find_node_from_path(dist_tree, subset_path)
  zone_path = PT.utils.path_head(subset_path, 2)
  zone = PT.find_node_from_path(dist_tree, zone_path)
  assert (zonedim:= PT.Zone.CellDimension(zone)) >= 2
  loc = 'FaceCenter' if zonedim == 3 else 'EdgeCenter'

  if PT.Subset.GridLocation(node) == 'Vertex':
    
    low_dim_elt = PT.Zone.NGonNode(zone) if zonedim == 3 else MT.Zone.EdgeNode(zone)
    pl_vtx = PT.get_np_value(PT.find_child_from_name(node, 'PointList'))[0]
    face_list = vtx_ids_to_face_ids(pl_vtx, low_dim_elt, comm, True)

    if only_bnd:
      # Exclude internal faces (see #73, #208)
      maia.algo.nface_to_pe(zone, comm) if zonedim == 3 else maia.algo.ngon_to_edge_pe(zone, comm)
      offset = MT.Element.distribution(low_dim_elt)[0] + 1
      pe = PT.get_np_value(PT.find_child_from_name(low_dim_elt, 'ParentElements'))
      is_boundary = pe[face_list-offset, 1] == 0
      face_list = face_list[is_boundary]

    PT.update_child(node, 'GridLocation', 'GridLocation_t', value=loc)
    PT.update_child(node, 'PointList', value=face_list.reshape((1,-1), order='F'))
    MT.new_Distribution({'Index' : par_utils.dn_to_distribution(face_list.size, comm)}, node)
  elif PT.Subset.GridLocation(node) != loc:
      raise ValueError(f"Unsupported location for subset {subset_path}")

