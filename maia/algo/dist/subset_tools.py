import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.transfer import protocols as EP
from maia.utils    import np_utils, par_utils

from maia.utils.parallel import algo as par_algo

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
    MT.newDistribution({'Index' : new_distri}, subset)

def vtx_ids_to_face_ids(vtx_ids, elt_n, comm, elt_full):
  """
  From an array of vertex ids, search in the distributed NGon node
  the id of faces constituted by these vertices.
  If elt_full is True, only faces having all their vertices in vtx_ids
  are returned.
  Otherwise, faces having at least one vertex in vtx_ids are returned.
  """
  i_rank = comm.Get_rank()
  elt_distri = MT.getDistribution(elt_n, 'Element')[1]
  delt_vtx   = PT.get_child_from_name(elt_n, 'ElementConnectivity')[1]
  if PT.Element.CGNSName(elt_n)=='NGON_n':
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

def convert_subset_as_facelist(dist_tree, subset_path, comm):
  node = PT.get_node_from_path(dist_tree, subset_path)
  zone_path = PT.path_head(subset_path, 2)
  if PT.Subset.GridLocation(node) == 'Vertex':
    zone = PT.get_node_from_path(dist_tree, zone_path)
    pl_vtx = PT.get_child_from_name(node, 'PointList')[1][0]
    face_list = vtx_ids_to_face_ids(pl_vtx, PT.Zone.NGonNode(zone), comm, True)
    PT.update_child(node, 'GridLocation', value='FaceCenter')
    PT.update_child(node, 'PointList', value=face_list.reshape((1,-1), order='F'))
    MT.newDistribution({'Index' : par_utils.dn_to_distribution(face_list.size, comm)}, node)
  elif PT.Subset.GridLocation(node) != 'FaceCenter':
      raise ValueError(f"Unsupported location for subset {subset_path}")

