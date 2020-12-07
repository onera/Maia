import Converter.Internal as I
from .              import utils
from .hdf_dataspace import create_data_array_filter

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
    zone_sub_region_path = zone_path+"/"+zone_subregion[0]

    distrib_ud = I.getNodeFromName1(zone_subregion, ':CGNS#Distribution')
    if distrib_ud is not None: #Subregion has its own pointlist / range
      matching_region = zone_subregion

    else: #Subregion must be related to an other bc/gc, just copy the hdf_filter
      bc_name_n = I.getNodeFromName1(zone_subregion, 'BCRegionName')
      gc_name_n = I.getNodeFromName1(zone_subregion, 'GridConnectivityRegionName')

      if bc_name_n is not None:
        for zbc in I.getNodesFromType1(zone, 'ZoneBC_t'):
          matching_region = I.getNodeFromName1(zbc, I.getValue(bc_name_n))
          if matching_region:
            break
      elif gc_name_n is not None:
        for zone_gc in I.getNodesFromType1(zone, 'ZoneGridConnectivity_t'):
          matching_region = I.getNodeFromName1(zone_gc, I.getValue(gc_name_n))
          if matching_region:
            break
      else:
        raise RuntimeError("ZoneSubRegion {0} is not well defined".format(zone_subregion[0]))

    assert(matching_region is not None)
    distrib_ud_n = I.getNodeFromName1(matching_region , ':CGNS#Distribution')
    distrib_data = I.getNodeFromName1(distrib_ud_n, 'Distribution')[1]

    data_shape = utils.pl_or_pr_size(matching_region)
    data_space = create_data_array_filter(distrib_data, data_shape)

    utils.apply_dataspace_to_pointlist(zone_subregion, zone_sub_region_path, data_space, hdf_filter)
    utils.apply_dataspace_to_arrays(zone_subregion, zone_sub_region_path, data_space, hdf_filter)

