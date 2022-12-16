import numpy as np

import maia
import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia.transfer.dist_to_part.index_exchange as IBTP
import maia.transfer.dist_to_part.recover_jn     as JBTP

from maia.utils     import s_numbering

ijk_to_idx_from_loc = {'IFaceCenter' : s_numbering.ijk_to_faceiIndex,
                       'JFaceCenter' : s_numbering.ijk_to_facejIndex,
                       'KFaceCenter' : s_numbering.ijk_to_facekIndex}
idx_to_ijk_from_loc = {'IFaceCenter' : s_numbering.faceiIndex_to_ijk,
                       'JFaceCenter' : s_numbering.facejIndex_to_ijk,
                       'KFaceCenter' : s_numbering.facekIndex_to_ijk}

is_zone_s = lambda n: PT.get_label(n) == 'Zone_t' and PT.Zone.Type(n)=='Structured'
is_zone_u = lambda n: PT.get_label(n) == 'Zone_t' and PT.Zone.Type(n)=='Unstructured'
is_initial_match = lambda n : PT.get_label(n) == 'GridConnectivity_t' and PT.GridConnectivity.is1to1(n) \
    and not MT.conv.is_intra_gc(n[0])

def pl_as_idx(zone, subset_predicate):
  """
  Assume that the PLs found following subset_predicates are (i,j,k) triplets
  and convert it to global faces indexes
  """
  assert PT.Zone.Type(zone) == 'Structured'
  for subset in PT.get_children_from_predicates(zone, subset_predicate):
    pl_node = PT.get_node_from_name(subset, 'PointList')
    if pl_node is not None:
      loc = PT.Subset.GridLocation(subset)
      pl = ijk_to_idx_from_loc[loc](*pl_node[1], PT.Zone.CellSize(zone), PT.Zone.VertexSize(zone))
      pl_node[1] = pl.reshape((1,-1), order='F')

def pl_as_ijk(zone, subset_predicate):
  """
  Assume that the PLs found following subset_predicates are global faces indexes
  and convert it to (i,j,k) triplets
  """
  assert PT.Zone.Type(zone) == 'Structured'
  for subset in PT.get_children_from_predicates(zone, subset_predicate):
    pl_node = PT.get_node_from_name(subset, 'PointList')
    if pl_node is not None:
      loc = PT.Subset.GridLocation(subset)
      pl_ijk = idx_to_ijk_from_loc[loc](pl_node[1][0], PT.Zone.CellSize(zone), PT.Zone.VertexSize(zone))
      PT.set_value(pl_node, pl_ijk)

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
            skip_nodes = ['PointList', 'PointListDonor', ':CGNS#GlobalNumbering', 'Donor', 'GridConnectivityType']
            for node in PT.get_children(gc):
              if PT.get_name(node) not in skip_nodes:
                PT.add_child(join_n, node)
            to_append.append(join_n)

          to_remove.append(PT.get_name(gc))
      for node in to_remove:
        PT.rm_children_from_name(zone_gc, node)
      for node in to_append: #Append everything at the end; otherwise we may find a new jn when looking for an old one
        PT.add_child(zone_gc, node)

def update_gc_donor_name(part_tree, comm):
  """
  Update or add the GridConnectivityDonorName name afted join splitting
  """
  is_1to1_gc    = lambda n: PT.get_label(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t'] \
                            and PT.GridConnectivity.is1to1(n)
  is_initial_gc = lambda n: is_1to1_gc(n) and not MT.conv.is_intra_gc(PT.get_name(n))
  send_l = [list() for n in range(comm.Get_size())]
  for p_base, p_zone in PT.iter_children_from_predicates(part_tree, 'CGNSBase_t/Zone_t', ancestors=True):
    for gc in PT.iter_children_from_predicates(p_zone, ['ZoneGridConnectivity_t', is_initial_gc]):
      cur_zone_path = PT.get_name(p_base) + '/' + PT.get_name(p_zone)
      opp_zone_path = PT.getZoneDonorPath(PT.get_name(p_base), gc)
      opp_rank = MT.conv.get_part_suffix(opp_zone_path)[0]
      send_l[opp_rank].append((PT.get_name(gc), cur_zone_path, opp_zone_path))

  recv_l = comm.alltoall(send_l)

  for p_base, p_zone in PT.iter_children_from_predicates(part_tree, 'CGNSBase_t/Zone_t', ancestors=True):
    for gc in PT.iter_children_from_predicates(p_zone, ['ZoneGridConnectivity_t', is_1to1_gc]):
      cur_zone_path = PT.get_name(p_base) + '/' + PT.get_name(p_zone)
      opp_zone_path = PT.getZoneDonorPath(PT.get_name(p_base), gc)
      if MT.conv.is_intra_gc(PT.get_name(gc)):
        opp_name = MT.conv.name_intra_gc(*MT.conv.get_part_suffix(opp_zone_path),
                                         *MT.conv.get_part_suffix(cur_zone_path))
        PT.new_child(gc, 'GridConnectivityDonorName', 'Descriptor_t', opp_name)
      else:
        opp_rank = MT.conv.get_part_suffix(opp_zone_path)[0]
        candidate_jns = [c[0] for c in recv_l[opp_rank] if c[1:] == (opp_zone_path, cur_zone_path)]
        dist_donor_name = PT.get_value(PT.get_child_from_name(gc, 'GridConnectivityDonorName'))
        candidate_jns = [jn for jn in candidate_jns if MT.conv.get_split_prefix(jn) == dist_donor_name]
        assert len(candidate_jns) == 1
        PT.rm_children_from_name(gc, 'GridConnectivityDonorName')
        PT.new_child(gc, 'GridConnectivityDonorName', 'Descriptor_t', candidate_jns[0])

def hybrid_jns_as_idx(part_tree):
  for s_zone in PT.get_nodes_from_predicate(part_tree, is_zone_s, depth=2):
    pl_as_idx(s_zone, ['ZoneGridConnectivity_t', is_initial_match])

def hybrid_jns_as_ijk(part_tree, comm):
  gc_predicate = ['ZoneGridConnectivity_t', is_initial_match]
  zone_s_data = {}
  for zone_s_path in PT.predicates_to_paths(part_tree, ['CGNSBase_t', is_zone_s]):
    zone_s = PT.get_node_from_path(part_tree, zone_s_path)
    pl_as_ijk(zone_s, gc_predicate)
    jn_dict = dict()
    for gc in PT.get_children_from_predicates(zone_s, gc_predicate):
      jn_dict[PT.get_name(gc)] = PT.Subset.GridLocation(gc)
    zone_s_data[zone_s_path] = (PT.Zone.CellSize(zone_s), jn_dict)
  zone_s_data_all = comm.allgather(zone_s_data)

  for zone_u_path in PT.predicates_to_paths(part_tree, ['CGNSBase_t', is_zone_u]):
    basename = PT.path_head(zone_u_path, 1)
    zone_u = PT.get_node_from_path(part_tree, zone_u_path)
    for gc in PT.get_children_from_predicates(zone_u, gc_predicate):
      opp_zone_path = PT.getZoneDonorPath(basename, gc)
      opp_rank = MT.conv.get_part_suffix(opp_zone_path)[0]
      opp_jn_name = PT.get_value(PT.get_child_from_name(gc, 'GridConnectivityDonorName'))
      try:
        opp_zone_size, opp_zone_jns = zone_s_data_all[opp_rank][opp_zone_path]
        opp_loc = opp_zone_jns[opp_jn_name]
        pl_donor = PT.get_child_from_name(gc, 'PointListDonor')
        pld_ijk = idx_to_ijk_from_loc[opp_loc](pl_donor[1][0], opp_zone_size, opp_zone_size+1)
        PT.set_value(pl_donor, pld_ijk)
      except KeyError:
        pass # Opp zone is unstructured

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
    if PT.Zone.Type(dist_zone) == 'Structured':
      IBTP.dist_pl_to_part_pl(dist_zone, part_zones, pl_paths, 'SFace', comm)
    for part_zone in part_zones:
      copy_additional_nodes(dist_zone, part_zone)

  # Next functions works on S meshes if PointList refers to global faces indices
  hybrid_jns_as_idx(part_tree)
            
  # Match original joins
  JBTP.get_pl_donor(dist_tree, part_tree, comm)
  split_original_joins(part_tree)
  update_gc_donor_name(part_tree, comm)

  # Go back to ijk for PointList
  hybrid_jns_as_ijk(part_tree, comm)
