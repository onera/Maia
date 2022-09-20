import numpy as np

import maia
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
  for node in PT.get_children(dist_zone):
    if PT.get_name(node) in names or PT.get_label(node) in types:
      PT.add_child(part_zone, node)
  #BCs
  names = ['.Solver#BC', 'BoundaryMarker']
  types = ['FamilyName_t']
  for p_zbc, p_bc in PT.iter_nodes_from_predicates(part_zone, 'ZoneBC_t/BC_t', ancestors=True):
    d_bc = PT.get_node_from_path(dist_zone, PT.get_name(p_zbc)+'/'+PT.get_name(p_bc))
    if d_bc: #Tmp, since S splitting store external JNs as bnd
      for node in PT.get_children(d_bc):
        if PT.get_name(node) in names or PT.get_label(node) in types:
          PT.add_child(p_bc, node)
  #GCs
  names = ['.Solver#Property', 'GridConnectivityDonorName', 'DistInterfaceId', 'DistInterfaceOrd']
  types = ['FamilyName_t', 'GridConnectivityProperty_t', 'GridConnectivityType_t']
  gc_predicate = 'ZoneGridConnectivity_t/GridConnectivity_t'
  for p_zgc, p_gc in PT.iter_nodes_from_predicates(part_zone, gc_predicate, ancestors=True):
    d_gc = PT.get_node_from_path(dist_zone, PT.get_name(p_zgc)+'/'+PT.get_name(p_gc))
    if d_gc: #Skip created jns
      for node in PT.get_children(d_gc):
        if PT.get_name(node) in names or PT.get_label(node) in types:
          PT.add_child(p_gc, node)

def split_original_joins(p_tree):
  """
  """
  for p_base, p_zone in PT.iter_children_from_predicates(p_tree, ['CGNSBase_t', 'Zone_t'], ancestors=True):
    d_zone_name = MT.conv.get_part_prefix(p_zone[0])
    for zone_gc in PT.get_children_from_label(p_zone, 'ZoneGridConnectivity_t'):
      to_remove = list()
      to_append = list()
      for gc in PT.get_children_from_label(zone_gc, 'GridConnectivity_t'):
        if not MT.conv.is_intra_gc(gc[0]): #Skip part joins
          pl       = PT.get_child_from_name(gc, 'PointList')[1]
          pl_d     = PT.get_child_from_name(gc, 'PointListDonor')[1]
          lngn     = PT.get_value(MT.getGlobalNumbering(gc, 'Index'))
          donor    = PT.get_child_from_name(gc, 'Donor')[1]
          # > List of couples (procs, parts) holding the opposite join
          opposed_parts = np.unique(donor, axis=0)
          for i_sub_jn, opp_part in enumerate(opposed_parts):
            join_n = PT.new_GridConnectivity(name       = MT.conv.add_split_suffix(PT.get_name(gc), i_sub_jn),
                                             donor_name = MT.conv.add_part_suffix(PT.get_value(gc), *opp_part),
                                             type       = 'Abutting1to1')

            matching_faces_idx = np.all(donor == opp_part, axis=1)

            # Extract sub arrays. OK to modify because indexing return a copy
            sub_pl   = pl  [:,matching_faces_idx]
            sub_pl_d = pl_d[:,matching_faces_idx]
            sub_lngn = lngn[matching_faces_idx]

            # Sort both pl and pld according to min joinId to ensure that
            # order is the same
            cur_path = p_base[0] + '/' + d_zone_name + '/' + gc[0]
            opp_path = PT.getZoneDonorPath(p_base[0], gc) + '/' + PT.get_value(PT.get_child_from_name(gc, 'GridConnectivityDonorName'))

            ref_pl = sub_pl if cur_path < opp_path else sub_pl_d
            sort_idx = np.argsort(ref_pl[0])
            sub_pl  [0]   = sub_pl  [0][sort_idx]
            sub_pl_d[0]   = sub_pl_d[0][sort_idx]
            sub_lngn      = sub_lngn[sort_idx]

            PT.new_PointList(name='PointList'     , value=sub_pl      , parent=join_n)
            PT.new_PointList(name='PointListDonor', value=sub_pl_d    , parent=join_n)
            MT.newGlobalNumbering({'Index' : sub_lngn}, join_n)
            #Copy decorative nodes
            skip_nodes = ['PointList', 'PointListDonor', ':CGNS#GlobalNumbering', 'Donor', \
                'GridConnectivityType', 'GridConnectivityDonorName']
            for node in PT.get_children(gc):
              if PT.get_name(node) not in skip_nodes:
                print(node[0])
                PT.add_child(join_n, node)
            to_append.append(join_n)

          to_remove.append(PT.get_name(gc))
      for node in to_remove:
        PT.rm_children_from_name(zone_gc, node)
      for node in to_append: #Append everything at the end; otherwise we may find a new jn when looking for an old one
        PT.add_child(zone_gc, node)

def post_partitioning(dist_tree, part_tree, comm):
  """
  """
  dist_zones     = PT.get_all_Zone_t(dist_tree)
  all_part_zones = PT.get_all_Zone_t(part_tree)
  parts_prefix    = [MT.conv.get_part_prefix(PT.get_name(zone)) for zone in all_part_zones]
  for dist_zone_path in PT.predicates_to_paths(dist_tree, 'CGNSBase_t/Zone_t'):
    # Recover matching zones
    dist_zone  = PT.get_node_from_path(dist_tree, dist_zone_path)
    part_zones = maia.transfer.utils.get_partitioned_zones(part_tree, dist_zone_path)

    # Create point list
    pl_paths = ['ZoneBC_t/BC_t', 'ZoneBC_t/BC_t/BCDataSet_t', 'ZoneSubRegion_t', 
        'FlowSolution_t', 'ZoneGridConnectivity_t/GridConnectivity_t']
    IBTP.dist_pl_to_part_pl(dist_zone, part_zones, pl_paths, 'Elements', comm)
    IBTP.dist_pl_to_part_pl(dist_zone, part_zones, pl_paths, 'Vertex'  , comm)
    for part_zone in part_zones:
      copy_additional_nodes(dist_zone, part_zone)
            
  # Match original joins
  JBTP.get_pl_donor(dist_tree, part_tree, comm)
  split_original_joins(part_tree)

