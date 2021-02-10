import Converter.Internal as I
import numpy as np

import maia.tree_exchange.dist_to_part.index_exchange as IBTP
import maia.tree_exchange.dist_to_part.recover_jn     as JBTP

def copy_additional_nodes(dist_zone, part_zone):
  """
  """
  #BCs
  names = ['.Solver#BC', 'BoundaryMarker']
  types = ['FamilyName_t']
  for p_zbc in I.getNodesFromType1(part_zone, 'ZoneBC_t'):
    for p_bc in I.getNodesFromType1(p_zbc, 'BC_t'):
      d_bc = I.getNodeFromPath(dist_zone, I.getName(p_zbc)+'/'+I.getName(p_bc))
      for node in I.getChildren(d_bc):
        if I.getName(node) in names or I.getType(node) in types:
          I._addChild(p_bc, node)
  #GCs
  names = ['.Solver#Property', 'Ordinal', 'OrdinalOpp']
  types = ['FamilyName_t', 'GridConnectivityProperty_t']
  for p_zgc in I.getNodesFromType1(part_zone, 'ZoneGridConnectivity_t'):
    for p_gc in I.getNodesFromType1(p_zgc, 'GridConnectivity_t'):
      d_gc = I.getNodeFromPath(dist_zone, I.getName(p_zgc)+'/'+I.getName(p_gc))
      if d_gc: #Skip created jns
        for node in I.getChildren(d_gc):
          if I.getName(node) in names or I.getType(node) in types:
            I._addChild(p_gc, node)

def split_original_joins(p_zone):
  """
  """
  to_remove = list()
  for zone_gc in I.getNodesFromType1(p_zone, 'ZoneGridConnectivity_t'):
    for gc in I.getNodesFromType1(zone_gc, 'GridConnectivity_t'):
      if I.getNodeFromName1(gc, 'Ordinal') is not None: #Skip part joins
        pl       = I.getNodeFromName1(gc, 'PointList')[1]
        pl_d     = I.getNodeFromName1(gc, 'PointListDonor')[1]
        lngn     = I.getNodeFromPath (gc, ':CGNS#GlobalNumbering/Index')[1]
        donor    = I.getNodeFromName1(gc, 'Donor')[1]
        # > List of couples (procs, parts) holding the opposite join
        opposed_parts = np.unique(donor, axis=0)
        for i_sub_jn, opp_part in enumerate(opposed_parts):
          join_n = I.newGridConnectivity(name      = I.getName(gc)+'.{0}'.format(i_sub_jn),
                                         donorName = I.getValue(gc)+'.P{0}.N{1}'.format(*opp_part),
                                         ctype     = 'Abutting1to1',
                                         parent    = zone_gc)

          matching_faces_idx = np.all(donor == opp_part, axis=1)

          # Extract sub arrays. OK to modify because indexing return a copy
          sub_pl   = pl  [:,matching_faces_idx]
          sub_pl_d = pl_d[:,matching_faces_idx]
          sub_lngn = lngn[matching_faces_idx]

          # Sort both pl and pld according to min joinId to ensure that
          # order is the same
          ordinal_cur = I.getNodeFromName1(gc, 'Ordinal')[1][0]
          ordinal_opp = I.getNodeFromName1(gc, 'OrdinalOpp')[1][0]
          ref_pl = sub_pl if ordinal_cur < ordinal_opp else sub_pl_d
          sort_idx = np.argsort(ref_pl[0])
          sub_pl  [0]   = sub_pl  [0][sort_idx]
          sub_pl_d[0]   = sub_pl_d[0][sort_idx]
          sub_lngn      = sub_lngn[sort_idx]

          I.newPointList(name='PointList'     , value=sub_pl      , parent=join_n)
          I.newPointList(name='PointListDonor', value=sub_pl_d    , parent=join_n)
          lntogn_ud = I.createUniqueChild(join_n, ':CGNS#GlobalNumbering', 'UserDefinedData_t')
          I.newDataArray('Index', value=sub_lngn, parent=lntogn_ud)
          #Copy decorative nodes
          skip_nodes = ['PointList', 'PointListDonor', ':CGNS#GlobalNumbering', 'Donor', 'Ordinal', 'OrdinalOpp']
          for node in I.getChildren(gc):
            if I.getName(node) not in skip_nodes:
              I._addChild(join_n, node)

        to_remove.append(gc)
  for node in to_remove:
    I._rmNode(p_zone, node)

def post_partitioning(dist_tree, part_tree, comm):
  """
  """
  dist_zones     = I.getZones(dist_tree)
  all_part_zones = I.getZones(part_tree)
  parts_prefix    = ['.'.join(I.getName(zone).split('.')[:-2]) for zone in all_part_zones]
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
  JBTP.get_pl_donor(dist_zones, all_part_zones, comm)
  for part_zone in all_part_zones:
    split_original_joins(part_zone)

