import Converter.Internal as I
import maia.sids.sids     as SIDS
import numpy              as np
import Pypdm.Pypdm        as PDM

from .split_U.cgns_to_pdm_dmesh import cgns_dist_zone_to_pdm_dmesh, cgns_dist_tree_to_joinopp_array
from .split_U.cgns_to_pdm_dmesh_nodal import cgns_dist_zone_to_pdm_dmesh_nodal
from .split_S import part_zone
from .pdm_mutipart_to_cgns         import pdm_mutipart_to_cgns

def partitioning(dist_tree, dzone_to_weighted_parts, comm,
                 split_method,
                 part_weight_method,
                 reorder_methods=["NONE", "NONE"],
                 n_cell_per_cache=0, n_face_per_pack=64):

  all_zones = I.getZones(dist_tree)
  u_zones   = [zone for zone in all_zones if SIDS.ZoneType(zone) == 'Unstructured']
  s_zones   = [zone for zone in all_zones if SIDS.ZoneType(zone) == 'Structured']

  if len(u_zones)*len(s_zones) != 0:
    raise RuntimeError("Hybrid meshes are not yet supported")

  part_tree = I.newCGNSTree()
  #For now only one base
  dist_base = I.getNodeFromType1(dist_tree, 'CGNSBase_t')
  part_base = I.createNode(I.getName(dist_base), 'CGNSBase_t', I.getValue(dist_base), parent=part_tree)

  #Split S zones
  for zone in s_zones:
    parts = part_zone.part_s_zone(zone, dzone_to_weighted_parts[I.getName(zone)], comm)
    for part in parts:
      I._addChild(part_base, part)

  #Split U zones
  if len(u_zones) > 0:
    n_part_per_zone_u = [len(dzone_to_weighted_parts[I.getName(zone)]) for zone in u_zones]
    n_part_per_zone_u = np.array(n_part_per_zone_u, dtype=np.int32)
    if part_weight_method == 2:
      part_weight = np.empty(sum(n_part_per_zone_u), dtype='float64')
      for izone, zone in enumerate(u_zones):
        offset    = sum(n_part_per_zone_u[:izone])
        part_weight[offset:offset+n_part_per_zone_u[izone]] = dzone_to_weighted_parts[I.getName(zone)]
    else:
      part_weight = None

    multi_part = PDM.MultiPart(len(u_zones), n_part_per_zone_u, 0, split_method,
        part_weight_method, part_weight, comm)

    dmesh_list = list()
    for izone, zone in enumerate(u_zones):
      #Determine NGON or ELMT
      elmt_types = [SIDS.ElementType(elmt) for elmt in I.getNodesFromType1(zone, 'Elements_t')]
      is_ngon = 22 in elmt_types
      if is_ngon:
        dmesh    = cgns_dist_zone_to_pdm_dmesh(zone, comm)
        dmesh_list.append(dmesh)
        multi_part.multipart_register_block(izone, dmesh)
      else:
        dmesh_nodal = cgns_dist_zone_to_pdm_dmesh_nodal(zone, comm)
        multi_part.multipart_register_dmesh_nodal(izone, dmesh_nodal)

    #Register joins
    join_to_opp_array = cgns_dist_tree_to_joinopp_array(dist_tree)
    n_total_joins = join_to_opp_array.shape[0]
    multi_part.multipart_register_joins(n_total_joins, join_to_opp_array)

    #Set reordering
    renum_cell_method = "PDM_PART_RENUM_CELL_" + reorder_methods[0]
    renum_face_method = "PDM_PART_RENUM_FACE_" + reorder_methods[1]
    if "CACHEBLOCKING" in reorder_methods[0]:
      cacheblocking_props = np.array([n_cell_per_cache, 1, 1, n_face_per_pack, split_method],
                                      dtype='int32', order='c')
    else:
      cacheblocking_props = None
    multi_part.multipart_set_reordering(-1,
                                        renum_cell_method.encode('utf-8'),
                                        renum_face_method.encode('utf-8'),
                                        cacheblocking_props)

    #Run now
    multi_part.multipart_run_ppart()

    #To rewrite to have a by zone behaviour
    pdm_mutipart_to_cgns(multi_part, dist_tree, n_part_per_zone_u, part_base, comm)

    del(dmesh_list) # Enforce free of PDM struct before free of numpy
    del(multi_part) # Force multi_part object to be deleted before n_part_per_zone array
    # for zone in zones:
      # I._rmNodesFromName1(zone, ':CGNS#MultiPart')

  #Add to level nodes
  for fam in I.getNodesFromType1(dist_base, 'Family_t'):
    I.addChild(part_base, fam)

  return part_tree
