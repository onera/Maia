import numpy as np

import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia.transfer.dist_to_part.index_exchange as IBTP
import maia.transfer.dist_to_part.recover_jn     as JBTP

def copy_additional_nodes(dist_zone, part_zone):
  """
  """
  #Zone data
  names = ['.Solver#Param']
  types = ['FamilyName_t', 'AdditionalFamilyName_t']
  for node in I.getChildren(dist_zone):
    if I.getName(node) in names or I.getType(node) in types:
      I._addChild(part_zone, node)
  #BCs
  names = ['.Solver#BC', 'BoundaryMarker']
  types = ['FamilyName_t']
  for p_zbc in I.getNodesFromType1(part_zone, 'ZoneBC_t'):
    for p_bc in I.getNodesFromType1(p_zbc, 'BC_t'):
      d_bc = I.getNodeFromPath(dist_zone, I.getName(p_zbc)+'/'+I.getName(p_bc))
      if d_bc: #Tmp, since S splitting store external JNs as bnd
        for node in I.getChildren(d_bc):
          if I.getName(node) in names or I.getType(node) in types:
            I._addChild(p_bc, node)
  #GCs
  names = ['.Solver#Property', 'GridConnectivityDonorName', 'DistInterfaceId', 'DistInterfaceOrd']
  types = ['FamilyName_t', 'GridConnectivityProperty_t', 'GridConnectivityType_t']
  for p_zgc in I.getNodesFromType1(part_zone, 'ZoneGridConnectivity_t'):
    for p_gc in I.getNodesFromType1(p_zgc, 'GridConnectivity_t'):
      d_gc = I.getNodeFromPath(dist_zone, I.getName(p_zgc)+'/'+I.getName(p_gc))
      if d_gc: #Skip created jns
        for node in I.getChildren(d_gc):
          if I.getName(node) in names or I.getType(node) in types:
            I._addChild(p_gc, node)

def split_original_joins(p_tree):
  """
  """
  for p_base, p_zone in PT.iter_children_from_predicates(p_tree, ['CGNSBase_t', 'Zone_t'], ancestors=True):
    d_zone_name = MT.conv.get_part_prefix(p_zone[0])
    for zone_gc in I.getNodesFromType1(p_zone, 'ZoneGridConnectivity_t'):
      to_remove = list()
      to_append = list()
      for gc in I.getNodesFromType1(zone_gc, 'GridConnectivity_t'):
        if not MT.conv.is_intra_gc(gc[0]): #Skip part joins
          pl       = I.getNodeFromName1(gc, 'PointList')[1]
          pl_d     = I.getNodeFromName1(gc, 'PointListDonor')[1]
          lngn     = I.getVal(MT.getGlobalNumbering(gc, 'Index'))
          donor    = I.getNodeFromName1(gc, 'Donor')[1]
          # > List of couples (procs, parts) holding the opposite join
          opposed_parts = np.unique(donor, axis=0)
          for i_sub_jn, opp_part in enumerate(opposed_parts):
            join_n = I.newGridConnectivity(name      = MT.conv.add_split_suffix(I.getName(gc), i_sub_jn),
                                           donorName = MT.conv.add_part_suffix(I.getValue(gc), *opp_part),
                                           ctype     = 'Abutting1to1')

            matching_faces_idx = np.all(donor == opp_part, axis=1)

            # Extract sub arrays. OK to modify because indexing return a copy
            sub_pl   = pl  [:,matching_faces_idx]
            sub_pl_d = pl_d[:,matching_faces_idx]
            sub_lngn = lngn[matching_faces_idx]

            # Sort both pl and pld according to min joinId to ensure that
            # order is the same
            cur_path = p_base[0] + '/' + d_zone_name + '/' + gc[0]
            opp_path = PT.getZoneDonorPath(p_base[0], gc) + '/' + I.getValue(I.getNodeFromName1(gc, 'GridConnectivityDonorName'))

            ref_pl = sub_pl if cur_path < opp_path else sub_pl_d
            sort_idx = np.argsort(ref_pl[0])
            sub_pl  [0]   = sub_pl  [0][sort_idx]
            sub_pl_d[0]   = sub_pl_d[0][sort_idx]
            sub_lngn      = sub_lngn[sort_idx]

            I.newPointList(name='PointList'     , value=sub_pl      , parent=join_n)
            I.newPointList(name='PointListDonor', value=sub_pl_d    , parent=join_n)
            MT.newGlobalNumbering({'Index' : sub_lngn}, join_n)
            #Copy decorative nodes
            skip_nodes = ['PointList', 'PointListDonor', ':CGNS#GlobalNumbering', 'Donor', 'GridConnectivityDonorName']
            for node in I.getChildren(gc):
              if I.getName(node) not in skip_nodes:
                I._addChild(join_n, node)
            to_append.append(join_n)

          to_remove.append(gc)
      for node in to_remove:
        I._rmNode(zone_gc, node)
      for node in to_append: #Append everything at the end; otherwise we may find a new jn when looking for an old one
        I._addChild(zone_gc, node)

def post_partitioning(dist_tree, part_tree, comm):
  """
  """
  dist_zones     = I.getZones(dist_tree)
  all_part_zones = I.getZones(part_tree)
  parts_prefix    = [MT.conv.get_part_prefix(I.getName(zone)) for zone in all_part_zones]
  for dist_zone in dist_zones:
    # Recover matching zones
    part_zones = [part for part,prefix in zip(all_part_zones,parts_prefix) if \
        prefix == I.getName(dist_zone)]

    # Create point list
    pl_paths = ['ZoneBC_t/BC_t', 'ZoneBC_t/BC_t/BCDataSet_t', 'ZoneSubRegion_t', 
        'FlowSolution_t', 'ZoneGridConnectivity_t/GridConnectivity_t']
    IBTP.dist_pl_to_part_pl(dist_zone, part_zones, pl_paths, 'Elements', comm)
    IBTP.dist_pl_to_part_pl(dist_zone, part_zones, pl_paths, 'Vertex'  , comm)
    for part_zone in part_zones:
      copy_additional_nodes(dist_zone, part_zone)
            
  # Match original joins
  # For now only one base is supported; this function is in fact called with CGNSBase_t
  true_dist_tree = ['CGNSTree', None, [dist_tree], 'CGNSTree_t']
  true_part_tree = ['CGNSTree', None, [part_tree], 'CGNSTree_t']
  JBTP.get_pl_donor(true_dist_tree, true_part_tree, comm)
  split_original_joins(true_part_tree)

