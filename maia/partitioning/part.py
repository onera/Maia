import Converter.Internal as I
import maia.sids.sids     as SIDS
import numpy              as NPY
import Pypdm.Pypdm        as PDM

from .cgns_to_pdm_distributed_mesh import cgns_dist_zone_to_pdm_dmesh, cgns_dist_tree_to_joinopp_array
from .pdm_mutipart_to_cgns         import pdm_mutipart_to_cgns

def partitioning(dist_tree, dzone_to_weighted_parts, comm,
                 split_method,
                 part_weight_method,
                 reorder_methods=["NONE", "NONE"],
                 multigrid=0, multigridOption=[],
                 n_cell_per_cache=0,
                 n_face_per_pack=64):
  """
  """


  dmesh_list = list()
  zones = I.getZones(dist_tree)
  for zone_tree in zones:
    zone_type_n = I.getNodeFromType1(zone_tree, 'ZoneType_t')
    zone_type   = zone_type_n[1].tostring()
    if(zone_type == b'Structured'):
      raise NotImplementedError
    else:
      dmesh_list.append(cgns_dist_zone_to_pdm_dmesh(zone_tree, comm))

  # join_to_opp_array = cgns_dist_tree_to_joinopp_array(dist_tree)

  n_zone = len(dzone_to_weighted_parts)
  n_part_per_zone = NPY.empty(n_zone, dtype='int32')
  for zone in zones:
    zone_name = zone[0]
    # > TODO : setup cgns_registery
    zoneg_id = I.getNodeFromName1(zone, ':CGNS#Registry')[1][0] - 1
    n_part_per_zone[zoneg_id] = len(dzone_to_weighted_parts[zone_name])

  if part_weight_method == 2:
    part_weight = NPY.empty(sum(n_part_per_zone), dtype='float64')
    for zone in zone:
      zone_name = zone[0]
      zoneg_id  = I.getNodeFromName1(zone, ':CGNS#Registry')[1][0] - 1
      offset    = sum(n_part_per_zone[:zoneg_id])
      part_weight[offset:offset+n_part_per_zone[zoneg_id]] = dzone_to_weighted_parts[zone_name]
  else:
    part_weight = None

  multi_part = PDM.MultiPart(n_zone, n_part_per_zone, 0, split_method, part_weight_method, part_weight, comm)
  # print("multi_part = ", multi_part)

  for i_zone, zone in enumerate(zones):
    zoneg_id = I.getNodeFromName1(zone, ':CGNS#Registry')[1][0] - 1
    dmesh    = dmesh_list[i_zone]
    # print(type(dmesh))
    # print(dir(PDM))
    # t1 = PDM.T1(10)
    # print(type(t1))
    # PDM.une_function(t1)
    # multi_part.multipart_gen(zoneg_id, t1)
    multi_part.multipart_register_block(zoneg_id, dmesh)
    # multi_part.multipart_register_block(i_zone, dmesh._id)
    # print "Set dmesh #{0} using zoneg_id {1}".format(dmesh._id, zoneg_id+1)

  # n_total_joins = join_to_opp_array.shape[0]
  # multi_part.multipart_register_joins(n_total_joins, join_to_opp_array)

  # Set reorering option -- -1 is a shortcut for all the zones
  renum_cell_method = "PDM_PART_RENUM_CELL_" + reorder_methods[0]
  renum_face_method = "PDM_PART_RENUM_FACE_" + reorder_methods[1]
  if "CACHEBLOCKING" in reorder_methods[0]:
    cacheblocking_props = NPY.array([n_cell_per_cache, 1, 1, n_face_per_pack, split_method],
                                    dtype='int32', order='c')
  else:
    cacheblocking_props = None
  multi_part.multipart_set_reordering(-1,
                                      renum_cell_method.encode('utf-8'),
                                      renum_face_method.encode('utf-8'),
                                      cacheblocking_props)

  multi_part.multipart_run_ppart()

  part_tree = pdm_mutipart_to_cgns(multi_part, dist_tree, n_part_per_zone, comm)

  del(dmesh_list) # Enforce free of PDM struct before free of numpy
  del(multi_part) # Force multi_part object to be deleted before n_part_per_zone array
  for zone in zones:
    I._rmNodesFromName1(zone, ':CGNS#MultiPart')

  return part_tree
