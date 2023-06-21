import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.utils    import par_utils
from maia.transfer import utils     as te_utils,\
                          protocols as EP

from maia.factory.dist_from_part import discover_nodes_from_matching
from .                           import index_exchange as IPTB

def _lngn_to_distri(lngn_list, comm):
  ptb = EP.PartToBlock(None, lngn_list, comm)
  pdm_distri = ptb.getDistributionCopy()
  return par_utils.full_to_partial_distribution(pdm_distri, comm)


def _discover_wrapper(dist_zone, part_zones, pl_path, data_path, comm):
  """
  Wrapper for discover_nodes_from_matching which add the node path in distree,
  but also recreate the distributed pointlist if needed
  """
  if pl_path.split('/')[0] == 'ZoneSubRegion_t':
    is_gc_zsr = lambda n: PT.get_label(n) == 'ZoneSubRegion_t' and \
                          PT.get_child_from_name(n, 'GridConnectivityRegionName') is not None
    ini_zsr_nodes_names = [PT.get_name(n) for n in PT.get_nodes_from_predicate(dist_zone, is_gc_zsr)]

  discover_nodes_from_matching(dist_zone, part_zones, pl_path,   comm, child_list=['GridLocation_t', 'Descriptor_t'])
  discover_nodes_from_matching(dist_zone, part_zones, data_path, comm)
  for nodes in PT.iter_children_from_predicates(dist_zone, pl_path, ancestors=True):
    node_path   = '/'.join([PT.get_name(node) for node in nodes])
    if PT.get_node_from_path(nodes[-1], 'PointRange') is None and \
       par_utils.exists_anywhere(part_zones, node_path+'/PointRange', comm):
       # PointRange must be computed on dist node
       if not par_utils.exists_anywhere(part_zones, node_path+'/:CGNS#GlobalNumbering/Index', comm):
         # > GlobalNumbering is required to do that
         IPTB.create_part_pr_gnum(dist_zone, part_zones, node_path, comm)
       IPTB.part_pr_to_dist_pr(dist_zone, part_zones, node_path, comm)
    if PT.get_node_from_path(nodes[-1], 'PointList') is None and \
       par_utils.exists_anywhere(part_zones, node_path+'/PointList', comm):
      # > Pointlist must be computed on dist node
      if not par_utils.exists_anywhere(part_zones, node_path+'/:CGNS#GlobalNumbering/Index', comm):
        # > GlobalNumbering is required to do that
        IPTB.create_part_pl_gnum(dist_zone, part_zones, node_path, comm)
      IPTB.part_pl_to_dist_pl(dist_zone, part_zones, node_path, comm)
  
  if pl_path.split('/')[0] == 'ZoneSubRegion_t':
    # If we have a splitted ZSR (because of GridConnectivityRegionName), we need to merge it
    # Here we assume that GC related ZSR can only refer to original jns, named using maia conventions
    # First remove split node already present in dist_zone
    is_gc_zsr_split = lambda n: is_gc_zsr(n) and PT.get_name(n) != name and PT.get_name(n).startswith(name)
    for name in ini_zsr_nodes_names:
      PT.rm_children_from_predicate(dist_zone, is_gc_zsr_split)
    # Now manage split node not present in dist zone (remove all but .0 and rename .0)
    is_gc_zsr_split = lambda n: is_gc_zsr(n) and PT.get_name(n) not in ini_zsr_nodes_names
    PT.rm_children_from_predicate(dist_zone, lambda n: is_gc_zsr_split(n) \
            and int(MT.conv.get_split_suffix(PT.get_name(n))) > 0)
    for zsr in PT.get_children_from_predicate(dist_zone, is_gc_zsr_split):
      PT.update_node(zsr, name=MT.conv.get_split_prefix(PT.get_name(zsr)))
      descri = PT.get_child_from_name(zsr, 'GridConnectivityRegionName')
      PT.update_node(descri, value=MT.conv.get_split_prefix(PT.get_value(descri)))
          
def part_coords_to_dist_coords(dist_zone, part_zones, comm, reduce_func=None):

  distribution = te_utils.get_cgns_distribution(dist_zone, 'Vertex')
  lntogn_list  = te_utils.collect_cgns_g_numbering(part_zones, 'Vertex')

  d_grid_co = PT.get_child_from_label(dist_zone, "GridCoordinates_t")
  part_data = dict()
  for coord in PT.iter_children_from_label(d_grid_co, 'DataArray_t'):
    part_data[PT.get_name(coord)] = list()

  for part_zone in part_zones:
    p_grid_co = PT.get_child_from_name(part_zone, PT.get_name(d_grid_co))
    for coord in PT.iter_children_from_label(p_grid_co, 'DataArray_t'):
      flat_data = coord[1].ravel(order='A') #Reshape structured arrays for PDM exchange
      part_data[PT.get_name(coord)].append(flat_data)

  # Exchange
  dist_data = EP.part_to_block(part_data, distribution, lntogn_list, comm, reduce_func)
  for coord, array in dist_data.items():
    dist_coord = PT.get_child_from_name(d_grid_co, coord)
    PT.set_value(dist_coord, array)

def _part_to_dist_sollike(dist_zone, part_zones, mask_tree, comm, reduce_func=None):
  """
  Shared code for FlowSolution_t and DiscreteData_t
  """
  for mask_sol in PT.get_children(mask_tree):
    d_sol = PT.get_child_from_name(dist_zone, PT.get_name(mask_sol)) #True container

    if not par_utils.exists_anywhere(part_zones, PT.get_name(d_sol), comm):
      continue #Skip FS that remains on dist_tree but are not present on part tree

    location = PT.Subset.GridLocation(d_sol)
    has_pl   = PT.get_child_from_name(d_sol, 'PointList') is not None

    if has_pl:
      distribution = te_utils.get_cgns_distribution(d_sol, 'Index')
      lntogn_list  = te_utils.collect_cgns_g_numbering(part_zones, 'Index', PT.get_name(d_sol))
    else:
      assert location in ['Vertex', 'CellCenter']
      if location == 'Vertex':
        distribution = te_utils.get_cgns_distribution(dist_zone, 'Vertex')
        lntogn_list  = te_utils.collect_cgns_g_numbering(part_zones, 'Vertex')
      elif location == 'CellCenter':
        distribution = te_utils.get_cgns_distribution(dist_zone, 'Cell')
        lntogn_list  = te_utils.collect_cgns_g_numbering(part_zones, 'Cell')

    #Discover data
    def can_be_transfered(n):
      exists = lambda zone : PT.get_node_from_path(zone, f'{PT.get_name(d_sol)}/{PT.get_name(n)}') is not None or \
                             PT.get_node_from_path(zone, f'{PT.get_name(d_sol)}'                 ) is     None
      return par_utils.all_true(part_zones, exists, comm)
    fields = [PT.get_name(n) for n in PT.get_children(mask_sol) if can_be_transfered(n)]

    part_data = {field : [] for field in fields}

    for part_zone in part_zones:
      p_sol = PT.get_child_from_name(part_zone, PT.get_name(d_sol))
      for field in fields:
        if p_sol is not None:
          flat_data = PT.get_child_from_name(p_sol, field)[1].ravel(order='A') #Reshape structured arrays for PDM exchange
          part_data[field].append(flat_data)

    # Exchange
    _lntogn_list = [lntogn for lntogn in lntogn_list if lntogn.size > 0]
    dist_data = EP.part_to_block(part_data, distribution, _lntogn_list, comm, reduce_func)
    for field, array in dist_data.items():
      dist_field = PT.get_child_from_name(d_sol, field)
      PT.set_value(dist_field, array)

def part_sol_to_dist_sol(dist_zone, part_zones, comm, include=[], exclude=[], reduce_func=None):
  """
  Transfert all the data included in FlowSolution_t nodes from partitioned
  zones to the distributed zone. Data created on (one or more) partitions and not present in dist_tree
  is also reported to the distributed zone.
  """
  # Complete distree with partitioned fields and exchange PL if needed
  _discover_wrapper(dist_zone, part_zones, 'FlowSolution_t', 'FlowSolution_t/DataArray_t', comm)
  mask_tree = te_utils.create_mask_tree(dist_zone, ['FlowSolution_t', 'DataArray_t'], include, exclude)
  _part_to_dist_sollike(dist_zone, part_zones, mask_tree, comm, reduce_func)
  #Cleanup : if field is None, data has been added by wrapper and must be removed
  for dist_sol in PT.iter_children_from_label(dist_zone, 'FlowSolution_t'):
    PT.rm_children_from_predicate(dist_sol, lambda n : PT.get_label(n) == 'DataArray_t' and n[1] is None)

def part_discdata_to_dist_discdata(dist_zone, part_zones, comm, include=[], exclude=[], reduce_func=None):
  """
  Transfert all the data included in DiscreteData_t from partitioned
  zones to the distributed zone. Data created on (one or more) partitions and not present in dist_tree
  is also reported to the distributed zone.
  """
  # Complete distree with partitioned fields and exchange PL if needed
  _discover_wrapper(dist_zone, part_zones, 'DiscreteData_t', 'DiscreteData_t/DataArray_t', comm)
  mask_tree = te_utils.create_mask_tree(dist_zone, ['DiscreteData_t', 'DataArray_t'], include, exclude)
  _part_to_dist_sollike(dist_zone, part_zones, mask_tree, comm, reduce_func)
  #Cleanup : if field is None, data has been added by wrapper and must be removed
  for dist_sol in PT.iter_children_from_label(dist_zone, 'DiscreteData_t'):
    PT.rm_children_from_predicate(dist_sol, lambda n : PT.get_label(n) == 'DataArray_t' and n[1] is None)

def part_subregion_to_dist_subregion(dist_zone, part_zones, comm, include=[], exclude=[], reduce_func=None):
  """
  Transfert all the data included in ZoneSubRegion_t nodes from the partitioned
  zones to the distributed zone.
  """
  _discover_wrapper(dist_zone, part_zones, 'ZoneSubRegion_t', 'ZoneSubRegion_t/DataArray_t', comm)
  mask_tree = te_utils.create_mask_tree(dist_zone, ['ZoneSubRegion_t', 'DataArray_t'], include, exclude)
  for mask_zsr in PT.get_children(mask_tree):
    d_zsr = PT.get_child_from_name(dist_zone, PT.get_name(mask_zsr)) #True ZSR
    # Search matching region
    matching_region_path = PT.getSubregionExtent(d_zsr, dist_zone)
    matching_region = PT.get_node_from_path(dist_zone, matching_region_path)
    assert matching_region is not None

    #Get distribution
    distribution = te_utils.get_cgns_distribution(matching_region, 'Index')

    #Get lngn and data
    fields = [PT.get_name(n) for n in PT.get_children(mask_zsr)]
    part_data = {field : [] for field in fields}
    if PT.get_label(matching_region) in ['GridConnectivity_t', 'GridConnectivity1to1_t']:
      # ZSR have been split
      ancestor, leaf = PT.path_head(matching_region_path), PT.path_tail(matching_region_path)
      lngn_list = []
      for part_zone in part_zones:
        for node in PT.iter_children_from_predicates(part_zone, [ancestor, leaf+'*']):
          # Get corresponding part ZSR
          lngn_list.append(PT.get_value(MT.getGlobalNumbering(node, 'Index')))
          good_zsr = lambda n: PT.get_label(n) == 'ZoneSubRegion_t' \
                               and PT.get_child_from_name(n, 'GridConnectivityRegionName') is not None \
                               and PT.get_value(PT.get_child_from_name(n, 'GridConnectivityRegionName')) == PT.get_name(node)
          p_zsr = PT.get_node_from_predicate(part_zone, good_zsr)
          for field in fields:
            part_data[field].append(PT.get_child_from_name(p_zsr, field)[1])
    else:
      lngn_list    = te_utils.collect_cgns_g_numbering(part_zones, 'Index', matching_region_path)
      #Discover data
      for part_zone in part_zones:
        p_zsr = PT.get_node_from_path(part_zone, PT.get_name(d_zsr))
        if p_zsr is not None:
          for field in fields:
            part_data[field].append(PT.get_child_from_name(p_zsr, field)[1])

      #Partitions having no data must be removed from lngn list since they have no contribution
      empty_parts_ids = [ipart for ipart, part_zone in enumerate(part_zones)\
          if PT.get_node_from_path(part_zone, PT.get_name(d_zsr)) is None]
      for ipart in empty_parts_ids[::-1]:
        lngn_list.pop(ipart)

    # Exchange
    dist_data = EP.part_to_block(part_data, distribution, lngn_list, comm, reduce_func)
    for field, array in dist_data.items():
      dist_field = PT.get_child_from_name(d_zsr, field)
      PT.set_value(dist_field, array)

  #Cleanup : if field is None, data has been added by wrapper and must be removed
  for dist_zsr in PT.iter_children_from_label(dist_zone, 'ZoneSubRegion_t'):
    PT.rm_children_from_predicate(dist_zsr, lambda n : PT.get_label(n) == 'DataArray_t' and n[1] is None)

def part_dataset_to_dist_dataset(dist_zone, part_zones, comm, include=[], exclude=[], reduce_func=None):
  """
  Transfert all the data included in BCDataSet_t/BCData_t nodes from partitioned
  zones to the distributed zone.
  """

  # Complete distree with partitioned fields and exchange PL if needed
  bc_ds_path = 'ZoneBC_t/BC_t/BCDataSet_t'
  _discover_wrapper(dist_zone, part_zones, bc_ds_path, bc_ds_path+'/BCData_t/DataArray_t', comm)

  for d_zbc in PT.iter_children_from_label(dist_zone, "ZoneBC_t"):
    labels = ['BC_t', 'BCDataSet_t', 'BCData_t', 'DataArray_t']
    mask_tree = te_utils.create_mask_tree(d_zbc, labels, include, exclude)
    for mask_bc in PT.get_children(mask_tree):
      bc_path   = PT.get_name(d_zbc) + '/' + PT.get_name(mask_bc)
      d_bc = PT.get_node_from_path(dist_zone, bc_path) #True BC
      for mask_dataset in PT.get_children(mask_bc):
        ds_path = bc_path + '/' + PT.get_name(mask_dataset)
        d_dataset = PT.get_node_from_path(dist_zone, ds_path) #True DataSet
        #If dataset has its own PointList, we must override bc distribution and lngn
        if MT.getDistribution(d_dataset) is not None:
          distribution = te_utils.get_cgns_distribution(d_dataset, 'Index')
          lngn_list    = te_utils.collect_cgns_g_numbering(part_zones, 'Index', ds_path)
        else: #Fallback to bc distribution
          distribution = te_utils.get_cgns_distribution(d_bc, 'Index')
          lngn_list    = te_utils.collect_cgns_g_numbering(part_zones, 'Index', bc_path)

        #Discover data
        data_paths = PT.predicates_to_paths(mask_dataset, ['*', '*'])
        part_data = {path : [] for path in data_paths}

        for part_zone in part_zones:
          p_dataset = PT.get_node_from_path(part_zone, ds_path)
          if p_dataset is not None:
            for path in data_paths:
              part_data[path].append(PT.get_node_from_path(p_dataset, path)[1])

        #Partitions having no data must be removed from lngn list since they have no contribution
        empty_parts_ids = [ipart for ipart, part_zone in enumerate(part_zones)\
            if PT.get_node_from_path(part_zone, ds_path) is None]
        for ipart in empty_parts_ids[::-1]:
          lngn_list.pop(ipart)

        #Exchange
        dist_data = EP.part_to_block(part_data, distribution, lngn_list, comm, reduce_func)
        for field, array in dist_data.items():
          dist_field = PT.get_node_from_path(d_dataset, field)
          PT.set_value(dist_field, array)
  #Cleanup : if field is None, data has been added by wrapper and must be removed
  for dist_ddata in PT.iter_nodes_from_predicates(dist_zone, bc_ds_path+'/BCData_t'):
    PT.rm_children_from_predicate(dist_ddata, lambda n : PT.get_label(n) == 'DataArray_t' and n[1] is None)

