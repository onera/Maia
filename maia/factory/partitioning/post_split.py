import numpy as np
import os

import maia
import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia.transfer.dist_to_part.index_exchange as IBTP
import maia.transfer.dist_to_part.recover_jn     as JBTP

from maia.utils     import s_numbering
from maia.utils     import logging as mlog

is_initial_match = PT.pred.label_is('GridConnectivity_t') & MT.pred.is_gc_of_kind(is_intra=False, is_1to1=True)

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
      pl = s_numbering.ijk_to_index_from_loc(*pl_node[1], loc, PT.Zone.VertexSize(zone))
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
      pl_ijk = s_numbering.index_to_ijk_from_loc(pl_node[1][0], loc, PT.Zone.VertexSize(zone))
      PT.set_value(pl_node, pl_ijk)

def update_zone_pointers(part_tree):
  """ Update ZonePointers values in BaseIterativeData, if any """
  for base in PT.iter_all_CGNSBase_t(part_tree):
    for base_it_data in PT.get_children_from_label(base, 'BaseIterativeData_t'):
      zone_pointers_n = PT.get_child_from_name(base_it_data, 'ZonePointers')
      if zone_pointers_n is not None:
        zone_pointers_all = PT.get_value(zone_pointers_n)
        for i, zone_points_inst in enumerate(zone_pointers_all): # Given instant
          pznames = []
          for zname in zone_points_inst: # Given zone, for this instant
            pzones = PT.get_children_from_predicate(base, lambda n : PT.get_label(n) == 'Zone_t' and \
                                                                     MT.conv.get_part_prefix(n[0]) == zname)
            pznames.extend([PT.get_name(z) for z in pzones])
          zone_pointers_all[i] = pznames
        number_of_zones = [len(k) for k in zone_pointers_all]
        if max(number_of_zones) == 0:
          # If we have only empty lists, set_value does not understand that it is strings -> enforce it
          zone_pointers_all = np.empty( (32,0,len(number_of_zones)), dtype='c', order='F')
          zone_pointers_all[:,:,:] = ' '
        PT.set_value(zone_pointers_n, zone_pointers_all)
        PT.update_child(base_it_data, 'NumberOfZones', 'DataArray_t', number_of_zones)

def _copy_additional_nodes_zsr(d_zsr, p_zsr):
  labels   = ['FamilyName_t', 'AdditionalFamilyName_t']
  reg_name = ['BCRegionName','GridConnectivityRegionName'] 

  for node in PT.get_children_from_predicate(d_zsr, PT.pred.label_in(labels)):
    PT.add_child(p_zsr, PT.deep_copy(node))
  for node in PT.get_children_from_predicate(d_zsr, PT.pred.label_is('Descriptor_t') \
                                                 & ~PT.pred.name_in(reg_name)):
    PT.add_child(p_zsr, PT.deep_copy(node))

def copy_additional_nodes(dist_zone, part_zone):
  """
  """
  is_container = PT.pred.label_in(['FlowSolution_t', 'DiscreteData_t'])
  IS_FULL = ~PT.pred.IS_SUBSET

  # Add Full containers (FS, DD & BCDataSet w/PL) -- partial containers are created before
  for d_fs in PT.iter_children_from_predicate(dist_zone, is_container & IS_FULL):
    p_fs = PT.new_FlowSolution(PT.get_name(d_fs), loc=PT.Container.GridLocation(d_fs), parent=part_zone)
    PT.set_label(p_fs, PT.get_label(d_fs))
  for nodes in PT.iter_children_from_predicates(dist_zone, ['ZoneBC_t', 'BC_t', PT.pred.label_is('BCDataSet_t') & IS_FULL], ancestors=True):
    bc_path = '/'.join(PT.get_name(n) for n in nodes[:-1])
    d_dset = nodes[-1]
    if (p_bc := PT.get_node_from_path(part_zone, bc_path)) is not None:
      PT.new_child(p_bc, PT.get_name(d_dset), PT.get_label(d_dset), PT.get_value(d_dset))


  #Zone data
  types = ['FamilyName_t', 'AdditionalFamilyName_t', 'ZoneIterativeData_t', 'ReferenceState_t',
           'FlowEquationSet_t', 'Descriptor_t', 'ConvergenceHistory_t', 'IntegralData_t']
  for node in PT.get_children_from_predicate(dist_zone, PT.pred.label_in(types)):
    PT.add_child(part_zone, PT.deep_copy(node))

  # Containers (FS & DD)
  types = ['Descriptor_t']
  for p_fs in PT.iter_children_from_predicate(part_zone, is_container):
    d_fs = PT.find_node_from_name(dist_zone, PT.get_name(p_fs))
    for node in PT.get_children_from_predicate(d_fs, PT.pred.label_in(types)):
      PT.add_child(p_fs, PT.deep_copy(node))
    
  #BCs
  bc_types = ['FamilyName_t', 'AdditionalFamilyName_t', 'ReferenceState_t', 'Ordinal_t', 'Descriptor_t']
  bcds_types = ['ReferenceState_t', 'Descriptor_t']
  for p_zbc, p_bc in PT.iter_nodes_from_predicates(part_zone, 'ZoneBC_t/BC_t', ancestors=True):
    d_bc = PT.get_node_from_path(dist_zone, PT.get_name(p_zbc)+'/'+PT.get_name(p_bc))
    if d_bc: #Tmp, since S splitting store external JNs as bnd
      for node in PT.get_children_from_predicate(d_bc, PT.pred.label_in(bc_types)):
        PT.add_child(p_bc, PT.deep_copy(node))
      # BCDS
      for p_dset in PT.iter_children_from_label(p_bc, 'BCDataSet_t'):
        d_dset = PT.find_child_from_name(d_bc, PT.get_name(p_dset))
        for node in PT.iter_children_from_predicate(d_dset, PT.pred.label_in(bcds_types)):
          PT.add_child(p_dset, PT.deep_copy(node))
    
  #GCs
  names = ['GridConnectivityDonorName']
  types = ['FamilyName_t', 'GridConnectivityProperty_t', 'GridConnectivityType_t', 'Descriptor_t']
  gc_predicate = 'ZoneGridConnectivity_t/GridConnectivity_t'
  for p_zgc, p_gc in PT.iter_nodes_from_predicates(part_zone, gc_predicate, ancestors=True):
    d_gc = PT.get_node_from_path(dist_zone, PT.get_name(p_zgc)+'/'+PT.get_name(p_gc))
    if d_gc: #Skip created jns
      for node in PT.get_children(d_gc):
        if PT.get_name(node) in names or PT.get_label(node) in types:
          PT.add_child(p_gc, PT.deep_copy(node))

  #ZSRs
  for p_zsr in PT.iter_children_from_label(part_zone, 'ZoneSubRegion_t'):
    d_zsr = PT.get_child_from_name(dist_zone, PT.get_name(p_zsr))
    _copy_additional_nodes_zsr(d_zsr, p_zsr)


def generate_related_zsr(dist_zone, part_zone):
  """
  """
  is_inter_gc = MT.pred.is_gc_of_kind(is_intra=False)
  for d_zsr in PT.iter_nodes_from_predicates(dist_zone, 'ZoneSubRegion_t'):
    bc_descriptor = PT.get_child_from_name(d_zsr, 'BCRegionName')
    gc_descriptor = PT.get_child_from_name(d_zsr, 'GridConnectivityRegionName')
    assert not (bc_descriptor and gc_descriptor)
    if bc_descriptor is not None:
      bc_name = PT.get_value(bc_descriptor)
      bc_n = PT.get_child_from_predicates(part_zone, f'ZoneBC_t/{bc_name}')
      if bc_n is not None and PT.Subset.n_elem(bc_n) > 0: # BC can exists, but be empty (if it holds BCDS of different location)
        p_zsr = PT.new_ZoneSubRegion(PT.get_name(d_zsr), bc_name=bc_name, parent=part_zone)
        _copy_additional_nodes_zsr(d_zsr, p_zsr)
    elif gc_descriptor is not None:
      gc_name = PT.get_value(gc_descriptor)
      is_related_gc = lambda n: is_inter_gc(n) and MT.conv.get_split_prefix(PT.get_name(n)) == gc_name and PT.Subset.n_elem(n) > 0
      gcs_n = PT.get_children_from_predicates(part_zone, ['ZoneGridConnectivity_t', is_related_gc])
      for gc_n in gcs_n:
        pgc_name = PT.get_name(gc_n)
        pzsr_name = MT.conv.add_split_suffix(PT.get_name(d_zsr), MT.conv.get_split_suffix(pgc_name))
        p_zsr = PT.new_ZoneSubRegion(pzsr_name, gc_name=pgc_name, parent=part_zone)
        _copy_additional_nodes_zsr(d_zsr, p_zsr)

def split_original_joins(p_tree):
  """
  """
  is_initial_gc = PT.pred.label_is('GridConnectivity_t') & MT.pred.is_gc_of_kind(is_intra=False, is_1to1=True)
  is_nomatch_gc = PT.pred.label_is('GridConnectivity_t') & PT.pred.is_gc_of_kind(is_1to1=False)
  for p_base, p_zone in PT.iter_children_from_predicates(p_tree, ['CGNSBase_t', 'Zone_t'], ancestors=True):
    d_zone_name = MT.conv.get_part_prefix(p_zone[0])
    for zone_gc in PT.get_children_from_label(p_zone, 'ZoneGridConnectivity_t'):
      to_remove = list()
      to_append = list()
      for gc in PT.get_children_from_predicate(zone_gc, is_initial_gc):
        pl       = PT.get_child_from_name(gc, 'PointList')[1]
        pl_d     = PT.get_child_from_name(gc, 'PointListDonor')[1]
        lngn     = MT.Subset.globalnumbering(gc)
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
          opp_path = PT.GridConnectivity.ZoneDonorPath(gc, p_base[0]) + '/' + PT.get_value(PT.get_child_from_name(gc, 'GridConnectivityDonorName'))

          ref_pl = sub_pl if cur_path < opp_path else sub_pl_d
          sort_idx = np.argsort(ref_pl[0])
          sub_pl  [0]   = sub_pl  [0][sort_idx]
          sub_pl_d[0]   = sub_pl_d[0][sort_idx]
          sub_lngn      = sub_lngn[sort_idx]

          PT.new_IndexArray(name='PointList'     , value=sub_pl      , parent=join_n)
          PT.new_IndexArray(name='PointListDonor', value=sub_pl_d    , parent=join_n)
          MT.new_GlobalNumbering({'Index' : sub_lngn}, join_n)
          #Copy decorative nodes
          skip_nodes = ['PointList', 'PointListDonor', ':CGNS#GlobalNumbering', 'Donor', 'GridConnectivityType']
          for node in PT.get_children(gc):
            if PT.get_name(node) not in skip_nodes:
              PT.add_child(join_n, PT.deep_copy(node))
          to_append.append(join_n)

        to_remove.append(PT.get_name(gc))
      for node in to_remove:
        PT.rm_children_from_name(zone_gc, node)
      for node in to_append: #Append everything at the end; otherwise we may find a new jn when looking for an old one
        PT.add_child(zone_gc, node)

      # Now deal non 1to1 JNs : we are unable to cut it properly, but we still
      # need to rename it to have correct naming conventions for other functions (see #165)
      # We use .P?.N? as donor name suffix to emphasize the fact that the join is not really splitted
      for gc in PT.get_children_from_predicate(zone_gc, is_nomatch_gc):
        PT.update_node(gc, name=PT.get_name(gc) + '.0', value=PT.get_value(gc) + '.P?.N?')

def update_gc_donor_name(part_tree, comm):
  """
  Update or add the GridConnectivityDonorName name afted join splitting
  """
  is_1to1_gc    = PT.pred.is_gc_of_kind(is_1to1=True)
  is_initial_gc = MT.pred.is_gc_of_kind(is_intra=False, is_1to1=True)
  send_l = [list() for n in range(comm.Get_size())]
  for p_base, p_zone in PT.iter_children_from_predicates(part_tree, 'CGNSBase_t/Zone_t', ancestors=True):
    for gc in PT.iter_children_from_predicates(p_zone, ['ZoneGridConnectivity_t', is_initial_gc]):
      cur_zone_path = PT.get_name(p_base) + '/' + PT.get_name(p_zone)
      opp_zone_path = PT.GridConnectivity.ZoneDonorPath(gc, PT.get_name(p_base))
      opp_rank = MT.conv.get_part_suffix(opp_zone_path)[0]
      send_l[opp_rank].append((PT.get_name(gc), cur_zone_path, opp_zone_path))

  recv_l = comm.alltoall(send_l)

  for p_base, p_zone in PT.iter_children_from_predicates(part_tree, 'CGNSBase_t/Zone_t', ancestors=True):
    for gc in PT.iter_children_from_predicates(p_zone, ['ZoneGridConnectivity_t', is_1to1_gc]):
      cur_zone_path = PT.get_name(p_base) + '/' + PT.get_name(p_zone)
      opp_zone_path = PT.GridConnectivity.ZoneDonorPath(gc, PT.get_name(p_base))
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
  for s_zone in PT.get_nodes_from_predicate(part_tree, PT.pred.is_zone_of_kind('S'), depth=2):
    pl_as_idx(s_zone, ['ZoneGridConnectivity_t', is_initial_match])

def hybrid_jns_as_ijk(part_tree, comm):
  gc_predicate = ['ZoneGridConnectivity_t', is_initial_match]
  zone_s_data = {}
  for zone_s_path in PT.predicates_to_paths(part_tree, ['CGNSBase_t', PT.pred.is_zone_of_kind('S')]):
    zone_s = PT.get_node_from_path(part_tree, zone_s_path)
    pl_as_ijk(zone_s, gc_predicate)
    jn_dict = dict()
    for gc in PT.get_children_from_predicates(zone_s, gc_predicate):
      jn_dict[PT.get_name(gc)] = PT.Subset.GridLocation(gc)
    zone_s_data[zone_s_path] = (PT.Zone.CellSize(zone_s), jn_dict)
  zone_s_data_all = comm.allgather(zone_s_data)

  for zone_u_path in PT.predicates_to_paths(part_tree, ['CGNSBase_t', PT.pred.is_zone_of_kind('U')]):
    basename = PT.utils.path_head(zone_u_path, 1)
    zone_u = PT.get_node_from_path(part_tree, zone_u_path)
    for gc in PT.get_children_from_predicates(zone_u, gc_predicate):
      opp_zone_path = PT.GridConnectivity.ZoneDonorPath(gc, basename)
      opp_rank = MT.conv.get_part_suffix(opp_zone_path)[0]
      opp_jn_name = PT.get_value(PT.get_child_from_name(gc, 'GridConnectivityDonorName'))
      try:
        opp_zone_size, opp_zone_jns = zone_s_data_all[opp_rank][opp_zone_path]
        opp_zone_size_vtx = tuple(k+1 for k in opp_zone_size)
        opp_loc = opp_zone_jns[opp_jn_name]
        pl_donor = PT.get_child_from_name(gc, 'PointListDonor')
        pld_ijk = s_numbering.index_to_ijk_from_loc(pl_donor[1][0], opp_loc, opp_zone_size_vtx)

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
    part_zones = MT.get_partitioned_zones(part_tree, dist_zone_path)

    # Create point list
    pl_paths = ['ZoneBC_t/BC_t', 'ZoneBC_t/BC_t/BCDataSet_t', 'ZoneSubRegion_t', 
        'FlowSolution_t', 'DiscreteData_t', 'ZoneGridConnectivity_t/GridConnectivity_t']
    IBTP.dist_pl_to_part_pl(dist_zone, part_zones, pl_paths, 'Elements', comm)
    for p_zone in part_zones:
      PT.rm_children_from_label(p_zone, 'FakeElements_t')
    IBTP.dist_pl_to_part_pl(dist_zone, part_zones, pl_paths, 'Vertex'  , comm)
    if PT.Zone.Type(dist_zone) == 'Structured' and PT.Zone.IndexDimension(dist_zone) == 3:
      IBTP.dist_pl_to_part_pl(dist_zone, part_zones, pl_paths, 'SFace', comm)
    for part_zone in part_zones:
      copy_additional_nodes(dist_zone, part_zone)

  # Next functions works on S meshes if PointList refers to global faces indices
  hybrid_jns_as_idx(part_tree)
            
  # Match original joins
  JBTP.get_pl_donor(dist_tree, part_tree, comm)
  split_original_joins(part_tree)
  for dist_zone_path in PT.predicates_to_paths(dist_tree, 'CGNSBase_t/Zone_t'):
    dist_zone  = PT.get_node_from_path(dist_tree, dist_zone_path)
    part_zones = MT.get_partitioned_zones(part_tree, dist_zone_path)
    for part_zone in part_zones:
      generate_related_zsr(dist_zone, part_zone) # Make BC_ZSR and GC_ZSR
  update_gc_donor_name(part_tree, comm)

  # Go back to ijk for PointList
  hybrid_jns_as_ijk(part_tree, comm)

  update_zone_pointers(part_tree)
