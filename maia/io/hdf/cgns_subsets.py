import maia.pytree        as PT
import maia.pytree.maia   as MT

from .              import utils
from .hdf_dataspace import create_pointlist_dataspace, create_data_array_filter

def _create_pl_filter(node, node_path, pl_name, distri_index, hdf_filter):
  pl_node = PT.get_child_from_name(node, pl_name)
  if pl_node is not None:
    # Retrieve index to create filter
    if pl_node[1] is not None:
      idx_dim = pl_node[1].shape[0]
    else: #When reading, PL are None -> catch from #Size node
      idx_dim = PT.get_child_from_name(node, f'{pl_name}#Size')[1][0]
    hdf_filter[f"{node_path}/{pl_name}"] = create_pointlist_dataspace(distri_index, idx_dim)

def create_zone_bc_filter(zone, zone_path, hdf_filter):
  """
  Fill up the hdf filter for the BC_t nodes present in
  the zone.
  Filter is created for the following nodes :
   - PointList (if present = unstruct. only)
   - All arrays founds in BCDataSets. Those arrays are supposed
     to be shaped as the PointList array. If a BCDataSet contains
     no PointList/PointRange node, the data is assumed to be consistent
     with the PointList/PointRange of the BC. Otherwise, the PointList/
     PointRange node of the BCDataSet is used to set the size of the BCData
     arrays. In this case, the PointList (if any) of the BCDataSet is
     written in the filter as well.
  """
  for zone_bc in PT.iter_children_from_label(zone, 'ZoneBC_t'):
    zone_bc_path = zone_path+"/"+zone_bc[0]
    for bc in PT.iter_children_from_label(zone_bc, 'BC_t'):
      bc_path = zone_bc_path+"/"+bc[0]

      distrib_bc   = PT.get_value(MT.getDistribution(bc, 'Index'))

      _create_pl_filter(bc, bc_path, 'PointList', distrib_bc, hdf_filter)

      for bcds in PT.iter_children_from_label(bc, "BCDataSet_t"):
        bcds_path = bc_path + "/" + bcds[0]
        distrib_bcds_n = MT.getDistribution(bcds)

        if distrib_bcds_n is None: #BCDS uses BC distribution
          distrib_data = distrib_bc
        else: #BCDS has its own distribution
          distrib_data = PT.get_child_from_name(distrib_bcds_n, 'Index')[1]
          _create_pl_filter(bcds, bcds_path, 'PointList', distrib_data, hdf_filter)

        # At read time, BCDataSet can be badly shaped (1,N) or (N1,N2) instead of (M,)
        # We use the #Size node to reshape it
        size_node = PT.get_child_from_name(bcds,'*#Size',depth=2)
        data_shape = PT.get_value(size_node) if size_node else None
        data_space_array = create_data_array_filter(distrib_data, data_shape)

        for bcdata in PT.iter_children_from_label(bcds, 'BCData_t'):
          bcdata_path = bcds_path + "/" + bcdata[0]
          utils.apply_dataspace_to_arrays(bcdata, bcdata_path, data_space_array, hdf_filter)


def create_zone_grid_connectivity_filter(zone, zone_path, hdf_filter):
  """
  Fill up the hdf filter for the GC_t nodes present in the zone.
  For unstructured GC (GridConnectivity_t), the filter is set up for
  the PointList and PointListDonor arrays.
  Structured GC (GridConnectivity1to1_t) are skipped since there is
  no data to load for these nodes.
  """
  for zone_gc in PT.iter_children_from_label(zone, 'ZoneGridConnectivity_t'):
    zone_gc_path = zone_path+"/"+zone_gc[0]
    for gc in PT.iter_children_from_label(zone_gc, 'GridConnectivity_t'):
      gc_path = zone_gc_path+"/"+gc[0]
      distrib_ia = PT.get_value(MT.getDistribution(gc, 'Index'))
      _create_pl_filter(gc, gc_path, 'PointList', distrib_ia, hdf_filter)
      _create_pl_filter(gc, gc_path, 'PointListDonor', distrib_ia, hdf_filter)

def create_flow_solution_filter(zone, zone_path, hdf_filter):
  """
  Fill up the hdf filter for the FlowSolution_t nodes present in the
  zone. The size of the dataspace are computed from the pointList node
  if present, or using allCells / allVertex if no pointList is present.
  Filter is created for the arrays and for the PointList if present
  """
  distrib_vtx  = PT.get_value(MT.getDistribution(zone, 'Vertex'))
  distrib_cell = PT.get_value(MT.getDistribution(zone, 'Cell'))
  for flow_solution in PT.iter_children_from_label(zone, 'FlowSolution_t'):
    flow_solution_path = zone_path + "/" + PT.get_name(flow_solution)
    grid_location = PT.Subset.GridLocation(flow_solution)
    distrib_ud_n = MT.getDistribution(flow_solution)
    if distrib_ud_n:
      distrib_data = PT.get_child_from_name(distrib_ud_n, 'Index')[1]
      _create_pl_filter(flow_solution, flow_solution_path, 'PointList', distrib_data, hdf_filter)
      data_space = create_data_array_filter(distrib_data)
    elif(grid_location == 'CellCenter'):
      data_space = create_data_array_filter(distrib_cell, zone[1][:,1])
    elif(grid_location == 'Vertex'):
      data_space = create_data_array_filter(distrib_vtx, zone[1][:,0])
    else:
      raise RuntimeError(f"GridLocation {grid_location} is not allowed without PL")
    utils.apply_dataspace_to_arrays(flow_solution, flow_solution_path, data_space, hdf_filter)

def create_zone_subregion_filter(zone, zone_path, hdf_filter):
  """
  Fill up the hdf filter for the ZoneSubRegion_t nodes present in
  the zone.
  The size of the dataspace are computed from
  - the corresponding BC or GC if the subregion is related to one of them
    (this information is given by a BCRegionName or GridConnectivityRegionName
    node)
  - the PointList / PointSize node of the subregion otherwise.
  Filter is created for the following nodes:
   - All arrays present in ZoneSubRegion;
   - PointList array if the zone is unstructured and if the subregion
     is not related to a BC/GC.
  """
  for zone_subregion in PT.iter_children_from_label(zone, 'ZoneSubRegion_t'):
    zone_subregion_path = zone_path+"/"+zone_subregion[0]

    # Search matching region
    matching_region_path = PT.Subset.ZSRExtent(zone_subregion, zone)
    matching_region = PT.get_node_from_path(zone, matching_region_path)
    assert(matching_region is not None)

    distrib_ud_n = MT.getDistribution(matching_region)
    if not distrib_ud_n:
      raise RuntimeError("ZoneSubRegion {0} is not well defined".format(zone_subregion[0]))
    distrib_data = PT.get_child_from_name(distrib_ud_n, 'Index')[1]

    _create_pl_filter(zone_subregion, zone_subregion_path, 'PointList', distrib_data, hdf_filter)

    data_space_ar = create_data_array_filter(distrib_data)
    utils.apply_dataspace_to_arrays(zone_subregion, zone_subregion_path, data_space_ar, hdf_filter)
