import maia.pytree      as PT
import maia.pytree.maia as MT
from maia.utils    import par_utils
from maia.transfer import utils     as te_utils,\
                          protocols as EP

def dist_coords_to_part_coords(dist_zone, part_zones, comm):
  """
  Transfert all the data included in GridCoordinates_t nodes from a distributed
  zone to the partitioned zones
  """
  #Get distribution
  distribution_vtx = te_utils.get_cgns_distribution(dist_zone, 'Vertex')

  #Get data
  dist_data = dict()
  dist_gc = PT.get_child_from_label(dist_zone, "GridCoordinates_t")
  for grid_co in PT.iter_children_from_label(dist_gc, 'DataArray_t'):
    dist_data[PT.get_name(grid_co)] = grid_co[1] #Prevent np->scalar conversion


  vtx_lntogn_list = te_utils.collect_cgns_g_numbering(part_zones, 'Vertex')
  part_data = EP.block_to_part(dist_data, distribution_vtx, vtx_lntogn_list, comm)

  for ipart, part_zone in enumerate(part_zones):
    part_gc = PT.new_node('GridCoordinates', 'GridCoordinates_t', parent=part_zone)
    for data_name, data in part_data.items():
      #F is mandatory to keep shared reference. Normally no copy is done
      shaped_data = data[ipart].reshape(PT.Zone.VertexSize(part_zone), order='F')
      PT.new_DataArray(data_name, shaped_data, parent=part_gc)



def _dist_to_part_sollike(dist_zone, part_zones, mask_tree, comm):
  """
  Shared code for FlowSolution_t and DiscreteData_t
  """
  #Get distribution
  for mask_sol in PT.get_children(mask_tree):
    d_sol = PT.get_child_from_name(dist_zone, PT.get_name(mask_sol)) #True container
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

    #Get data
    fields = [PT.get_name(n) for n in PT.get_children(mask_sol)]
    dist_data = {field : PT.get_child_from_name(d_sol, field)[1] for field in fields}

    #Exchange
    part_data = EP.block_to_part(dist_data, distribution, lntogn_list, comm)

    for ipart, part_zone in enumerate(part_zones):
      #Skip void flow solution (can occur with point lists)
      if lntogn_list[ipart].size > 0:
        if has_pl:
          p_sol = PT.get_child_from_name(part_zone, PT.get_name(d_sol))
          shape = PT.get_child_from_name(p_sol, 'PointList')[1].shape[1]
        else:
          p_sol = PT.new_child(part_zone, PT.get_name(d_sol), PT.get_label(d_sol))
          PT.new_GridLocation(location, parent=p_sol)
          shape = PT.Zone.VertexSize(part_zone) if location == 'Vertex' else PT.Zone.CellSize(part_zone)
        for data_name, data in part_data.items():
          #F is mandatory to keep shared reference. Normally no copy is done
          shaped_data = data[ipart].reshape(shape, order='F')
          PT.new_DataArray(data_name, shaped_data, parent=p_sol)

def dist_sol_to_part_sol(dist_zone, part_zones, comm, include=[], exclude=[]):
  """
  Transfert all the data included in FlowSolution_t nodes from a distributed
  zone to the partitioned zones
  """
  mask_tree = te_utils.create_mask_tree(dist_zone, ['FlowSolution_t', 'DataArray_t'], include, exclude)
  _dist_to_part_sollike(dist_zone, part_zones, mask_tree, comm)

def dist_discdata_to_part_discdata(dist_zone, part_zones, comm, include=[], exclude=[]):
  """
  Transfert all the data included in DiscreteData_t nodes from a distributed
  zone to the partitioned zones
  """
  mask_tree = te_utils.create_mask_tree(dist_zone, ['DiscreteData_t', 'DataArray_t'], include, exclude)
  _dist_to_part_sollike(dist_zone, part_zones, mask_tree, comm)

def dist_dataset_to_part_dataset(dist_zone, part_zones, comm, include=[], exclude=[]):
  """
  Transfert all the data included in BCDataSet_t/BCData_t nodes from a distributed
  zone to the partitioned zones
  """
  for d_zbc in PT.iter_children_from_label(dist_zone, "ZoneBC_t"):
    labels = ['BC_t', 'BCDataSet_t', 'BCData_t', 'DataArray_t']
    mask_tree = te_utils.create_mask_tree(d_zbc, labels, include, exclude)
    for mask_bc in PT.get_children(mask_tree):
      bc_path = PT.get_name(d_zbc) + '/' + PT.get_name(mask_bc)
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
        #Get data
        data_paths = PT.predicates_to_paths(mask_dataset, ['*', '*'])
        dist_data = {data_path : PT.get_node_from_path(d_dataset, data_path)[1] for data_path in data_paths}

        #Exchange
        part_data = EP.block_to_part(dist_data, distribution, lngn_list, comm)

        #Put part data in tree
        for ipart, part_zone in enumerate(part_zones):
          part_bc = PT.get_node_from_path(part_zone, bc_path)
          # Skip void bcs
          if lngn_list[ipart].size > 0:
            # Create dataset if no existing
            part_ds = PT.update_child(part_bc, PT.get_name(d_dataset), PT.get_label(d_dataset), PT.get_value(d_dataset))
            # Add data
            for data_name, data in part_data.items():
              container_name, field_name = data_name.split('/')
              p_container = PT.update_child(part_ds, container_name, 'BCData_t')
              PT.new_DataArray(field_name, data[ipart], parent=p_container)

def dist_subregion_to_part_subregion(dist_zone, part_zones, comm, include=[], exclude=[]):
  """
  Transfert all the data included in ZoneSubRegion_t nodes from a distributed
  zone to the partitioned zones
  """
  mask_tree = te_utils.create_mask_tree(dist_zone, ['ZoneSubRegion_t', 'DataArray_t'], include, exclude)
  for mask_zsr in PT.get_children(mask_tree):
    d_zsr = PT.get_child_from_name(dist_zone, PT.get_name(mask_zsr)) #True ZSR
    # Search matching region
    matching_region_path = PT.getSubregionExtent(d_zsr, dist_zone)
    matching_region = PT.get_node_from_path(dist_zone, matching_region_path)
    assert matching_region is not None

    if PT.get_label(matching_region)!='GridConnectivity_t':
      #Get distribution and lngn
      distribution = te_utils.get_cgns_distribution(matching_region, 'Index')
      lngn_list    = te_utils.collect_cgns_g_numbering(part_zones, 'Index', matching_region_path)

      #Get Data
      fields = [PT.get_name(n) for n in PT.get_children(mask_zsr)]
      dist_data = {field : PT.get_child_from_name(d_zsr, field)[1] for field in fields}
      #Exchange
      part_data = EP.block_to_part(dist_data, distribution, lngn_list, comm)

      #Put part data in tree
      for ipart, part_zone in enumerate(part_zones):
        # Skip void zsr
        if lngn_list[ipart].size > 0:
          # Create ZSR if not existing (eg was defined by bc/gc)
          p_zsr = PT.update_child(part_zone, PT.get_name(d_zsr), PT.get_label(d_zsr), PT.get_value(d_zsr))
          for field_name, data in part_data.items():
            PT.new_DataArray(field_name, data[ipart], parent=p_zsr)

    else:
      distribution = te_utils.get_cgns_distribution(matching_region, 'Index')

      path_m1 = '/'.join(matching_region_path.split('/')[:-1])
      dgc_name = matching_region_path.split('/')[-1]
      all_paths = list()
      for part_zone in part_zones:
        ppaths = PT.predicates_to_paths(part_zone, [path_m1, lambda n: PT.get_name(n).split('.')[0]==dgc_name])
        for path in ppaths:
          if path not in all_paths:
            all_paths.append(path)

      for path in all_paths:
        #Get distribution and lngn
        distribution = te_utils.get_cgns_distribution(matching_region, 'Index')
        lngn_list    = te_utils.collect_cgns_g_numbering(part_zones, 'Index', path)

        #Get Data
        fields = [PT.get_name(n) for n in PT.get_children(mask_zsr)]
        dist_data = {field : PT.get_child_from_name(d_zsr, field)[1] for field in fields}

        #Exchange
        part_data = EP.block_to_part(dist_data, distribution, lngn_list, comm)

        #Put part data in tree
        for ipart, part_zone in enumerate(part_zones):
          # Skip void zsr
          if lngn_list[ipart].size > 0:
            # Create ZSR if not existing (eg was defined by bc/gc)
            # p_zsr = PT.update_child(part_zone, PT.get_name(d_zsr), PT.get_label(d_zsr), PT.get_value(d_zsr))
            matching_region_name = path.split('/')[-1]
            is_matching_pzsr = lambda n:  PT.get_label(n)=='ZoneSubRegion_t' and \
                                          PT.get_value(PT.get_child_from_label(n, 'Descriptor_t'))==matching_region_name
            p_zsr = PT.get_node_from_predicate(part_zone, is_matching_pzsr)
            for field_name, data in part_data.items():
              PT.new_DataArray(field_name, data[ipart], parent=p_zsr)
