import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.transfer import protocols as EP
from maia.utils    import np_utils, par_utils, as_pdm_gnum

def vtx_ids_to_face_ids(vtx_ids, ngon, comm):
  """
  From an array of vertex ids, search in the distributed NGon node
  the id of faces constituted by these vertices.
  Only Face having all their vertices in vtx_ids are returned
  """
  i_rank = comm.Get_rank()
  dface_vtx     = PT.get_child_from_name(ngon, 'ElementConnectivity')[1]
  dface_vtx_idx = PT.get_child_from_name(ngon, 'ElementStartOffset')[1]
  ngon_distri   = MT.getDistribution(ngon, 'Element')[1]

  # Prepare next BTP : we need a block view of the vtx ids (no duplicate, increasing order)
  # on a distribution involving all the vtx, so we include dface_vtx as a partition
  # We can exchange anything, since what is important is just the *number* of recv
  # data (== number of apparitions of this vtx in vtx_ids)
  PTB = EP.PartToBlock(None, [dface_vtx, vtx_ids], comm, keep_multiple=True)
  distri = PTB.getDistributionCopy()
  count,_  = PTB.exchange_field([np.empty(0, bool), np.ones(vtx_ids.size, bool)],                      #Data
                                [np.zeros(dface_vtx.size, np.int32), np.ones(vtx_ids.size, np.int32)]) #Stride

  # Fill tag array with True if vtx appears one or more in vtx_ids
  d_vtx_tag  = np.zeros(distri[i_rank+1] - distri[i_rank], bool)
  d_vtx_tag[count > 0] = True

  # Now send this array using face_vtx as lngn (so each face will receive the flag
  # of each one of its vertices)
  dface_vtx_tag = EP.block_to_part(d_vtx_tag, distri, [dface_vtx], comm)
  # Then reduce : select face if all its vertices have flag set to 1
  dface_vtx_idx_loc = dface_vtx_idx - dface_vtx_idx[0]
  dface_vtx_tag = np.logical_and.reduceat(dface_vtx_tag[0], dface_vtx_idx_loc[:-1])
  face_ids = np.where(dface_vtx_tag)[0] + ngon_distri[0] + 1

  return np_utils.safe_int_cast(face_ids, vtx_ids.dtype)


def convert_subset_as_facelist(dist_tree, subset_path, comm):
  node = PT.get_node_from_path(dist_tree, subset_path)
  zone_path = PT.path_head(subset_path, 2)
  if PT.Subset.GridLocation(node) == 'Vertex':
    zone = PT.get_node_from_path(dist_tree, zone_path)
    pl_vtx = PT.get_child_from_name(node, 'PointList')[1][0]
    face_list = vtx_ids_to_face_ids(pl_vtx, PT.Zone.NGonNode(zone), comm)
    PT.update_child(node, 'GridLocation', value='FaceCenter')
    PT.update_child(node, 'PointList', value=face_list.reshape((1,-1), order='F'))
    MT.newDistribution({'Index' : par_utils.dn_to_distribution(face_list.size, comm)}, node)
  elif PT.Subset.GridLocation(node) != 'FaceCenter':
      raise ValueError(f"Unsupported location for subset {subset_path}")

