import Converter.Internal as I
from .              import utils
from .hdf_dataspace import create_data_array_filter

import maia.sids.sids     as SIDS

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
  for zone_bc in I.getNodesFromType1(zone, 'ZoneBC_t'):
    zone_bc_path = zone_path+"/"+zone_bc[0]
    for bc in I.getNodesFromType1(zone_bc, 'BC_t'):
      bc_path = zone_bc_path+"/"+bc[0]

      distrib_bc_n = I.getNodeFromName1(bc          , ':CGNS#Distribution')
      distrib_bc   = I.getNodeFromName1(distrib_bc_n, 'Index')[1]

      bc_shape = utils.pl_or_pr_size(bc)
      data_space = create_data_array_filter(distrib_bc, bc_shape)
      utils.apply_dataspace_to_pointlist(bc, bc_path, data_space, hdf_filter)

      for bcds in I.getNodesFromType1(bc, "BCDataSet_t"):
        bcds_path = bc_path + "/" + bcds[0]
        distrib_bcds_n = I.getNodeFromName1(bcds, ':CGNS#Distribution')

        if distrib_bcds_n is None: #BCDS uses BC distribution
          distrib_data = distrib_bc
          data_shape   = bc_shape
        else: #BCDS has its own distribution
          distrib_data = I.getNodeFromName1(distrib_bcds_n, 'Index')[1]
          data_shape = utils.pl_or_pr_size(bcds)

        data_space = create_data_array_filter(distrib_data, data_shape)
        utils.apply_dataspace_to_pointlist(bcds, bcds_path, data_space, hdf_filter)
        for bcdata in I.getNodesFromType1(bcds, 'BCData_t'):
          bcdata_path = bcds_path + "/" + bcdata[0]
          utils.apply_dataspace_to_arrays(bcdata, bcdata_path, data_space, hdf_filter)


def create_zone_grid_connectivity_filter(zone, zone_path, hdf_filter):
  """
  Fill up the hdf filter for the GC_t nodes present in the zone.
  For unstructured GC (GridConnectivity_t), the filter is set up for
  the PointList and PointListDonor arrays.
  Structured GC (GridConnectivity1to1_t) are skipped since there is
  no data to load for these nodes.
  """
  for zone_gc in I.getNodesFromType1(zone, 'ZoneGridConnectivity_t'):
    zone_gc_path = zone_path+"/"+zone_gc[0]
    for gc in I.getNodesFromType1(zone_gc, 'GridConnectivity_t'):
      gc_path = zone_gc_path+"/"+gc[0]
      distrib_ud = I.getNodeFromName1(gc        , ':CGNS#Distribution')
      distrib_ia = I.getNodeFromName1(distrib_ud, 'Index')[1]

      gc_shape   = utils.pl_or_pr_size(gc)
      data_space = create_data_array_filter(distrib_ia, gc_shape)
      utils.apply_dataspace_to_pointlist(gc, gc_path, data_space, hdf_filter)


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
  for zone_subregion in I.getNodesFromType1(zone, 'ZoneSubRegion_t'):
    zone_subregion_path = zone_path+"/"+zone_subregion[0]

    # Search matching region
    matching_region_path = SIDS.get_subregion_extent(zone_subregion, zone)
    matching_region = I.getNodeFromPath(zone, matching_region_path)
    assert(matching_region is not None)

    distrib_ud_n = I.getNodeFromName1(matching_region , ':CGNS#Distribution')
    if not distrib_ud_n:
      raise RuntimeError("ZoneSubRegion {0} is not well defined".format(zone_subregion[0]))
    distrib_data = I.getNodeFromName1(distrib_ud_n, 'Index')[1]

    data_shape = utils.pl_or_pr_size(matching_region)
    data_space = create_data_array_filter(distrib_data, data_shape)

    utils.apply_dataspace_to_pointlist(zone_subregion, zone_subregion_path, data_space, hdf_filter)
    utils.apply_dataspace_to_arrays(zone_subregion, zone_subregion_path, data_space, hdf_filter)
