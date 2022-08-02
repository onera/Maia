import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.maia   as MT

from .              import utils
from .hdf_dataspace import create_data_array_filter

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

      distrib_bc   = I.getVal(MT.getDistribution(bc, 'Index'))

      bc_shape = utils.pl_or_pr_size(bc)
      data_space = create_data_array_filter(distrib_bc, bc_shape)
      utils.apply_dataspace_to_pointlist(bc, bc_path, data_space, hdf_filter)

      for bcds in PT.iter_children_from_label(bc, "BCDataSet_t"):
        bcds_path = bc_path + "/" + bcds[0]
        distrib_bcds_n = MT.getDistribution(bcds)

        if distrib_bcds_n is None: #BCDS uses BC distribution
          distrib_data = distrib_bc
          data_shape   = bc_shape
        else: #BCDS has its own distribution
          distrib_data = I.getNodeFromName1(distrib_bcds_n, 'Index')[1]
          data_shape = utils.pl_or_pr_size(bcds)

        data_space_pl    = create_data_array_filter(distrib_data, data_shape)
        #BCDataSet always use flat data array
        data_space_array = create_data_array_filter(distrib_data, [data_shape.prod()])
        utils.apply_dataspace_to_pointlist(bcds, bcds_path, data_space_pl, hdf_filter)
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
      distrib_ia = I.getVal(MT.getDistribution(gc, 'Index'))

      gc_shape   = utils.pl_or_pr_size(gc)
      data_space = create_data_array_filter(distrib_ia, gc_shape)
      utils.apply_dataspace_to_pointlist(gc, gc_path, data_space, hdf_filter)

def create_flow_solution_filter(zone, zone_path, hdf_filter):
  """
  Fill up the hdf filter for the FlowSolution_t nodes present in the
  zone. The size of the dataspace are computed from the pointList node
  if present, or using allCells / allVertex if no pointList is present.
  Filter is created for the arrays and for the PointList if present
  """
  distrib_vtx  = I.getVal(MT.getDistribution(zone, 'Vertex'))
  distrib_cell = I.getVal(MT.getDistribution(zone, 'Cell'))
  for flow_solution in PT.iter_children_from_label(zone, 'FlowSolution_t'):
    flow_solution_path = zone_path + "/" + I.getName(flow_solution)
    grid_location = PT.Subset.GridLocation(flow_solution)
    distrib_ud_n = MT.getDistribution(flow_solution)
    if distrib_ud_n:
      distrib_data = I.getNodeFromName1(distrib_ud_n, 'Index')[1]
      data_shape = utils.pl_or_pr_size(flow_solution)
      data_space_pl = create_data_array_filter(distrib_data, data_shape)
      data_space = create_data_array_filter(distrib_data, [data_shape.prod()])
      utils.apply_dataspace_to_pointlist(flow_solution, flow_solution_path, data_space_pl, hdf_filter)
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
    matching_region_path = PT.getSubregionExtent(zone_subregion, zone)
    matching_region = I.getNodeFromPath(zone, matching_region_path)
    assert(matching_region is not None)

    distrib_ud_n = MT.getDistribution(matching_region)
    if not distrib_ud_n:
      raise RuntimeError("ZoneSubRegion {0} is not well defined".format(zone_subregion[0]))
    distrib_data = I.getNodeFromName1(distrib_ud_n, 'Index')[1]

    data_shape = utils.pl_or_pr_size(matching_region)
    data_space_pl = create_data_array_filter(distrib_data, data_shape)
    data_space_ar = create_data_array_filter(distrib_data, [data_shape.prod()])

    utils.apply_dataspace_to_pointlist(zone_subregion, zone_subregion_path, data_space_pl, hdf_filter)
    utils.apply_dataspace_to_arrays(zone_subregion, zone_subregion_path, data_space_ar, hdf_filter)
