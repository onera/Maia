import Converter.Internal as I
import maia.sids.sids as SIDS

from .cgns_to_pdm_distributed_mesh import cgns_dist_zone_to_pdm_dmesh


def partitioning(dist_tree, dzone_to_weighted_parts, comm,
                 split_method,
                 part_weight_method,
                 reorder_methods=["NONE", "NONE"],
                 multigrid=0, multigridOption=[],
                 n_cell_per_cache=0,
                 n_face_per_pack=64):
  """
  """

  I.printTree(dist_tree)

  dmesh_list = list()
  zones = I.getZones(dist_tree)
  for zone_tree in zones:
    zone_type_n = I.getNodeFromType1(zone_tree, 'ZoneType_t')
    zone_type   = zone_type_n[1].tostring()
    if(zone_type == b'Structured'):
      raise NotImplemented
    else:
      dmesh_list.append(cgns_dist_zone_to_pdm_dmesh(zone_tree))


  part_tree = None
  return part_tree
