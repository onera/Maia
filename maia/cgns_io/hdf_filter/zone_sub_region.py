import Converter.Internal as I

import maia.sids.sids as SIDS
from .data_array import create_data_array_filter

def create_zone_subregion_filter(zone, zone_subregion, zone_sub_region_path, hdf_filter):
  """
  """
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

  pr_n = I.getNodeFromName1(matching_region, 'PointRange')
  pl_n = I.getNodeFromName1(matching_region, 'PointList#Size')
  if pr_n is not None:
    data_shape = SIDS.point_range_size(pr_n)
  if pl_n is not None:
    data_shape = pl_n[1]

  data_space = create_data_array_filter(distrib_data, data_shape)
  if matching_region == zone_subregion and pl_n is not None:
    hdf_filter[zone_sub_region_path + "/PointList"] = data_space
  for data_array in I.getNodesFromType1(zone_subregion, 'DataArray_t'):
    path = zone_sub_region_path+"/"+data_array[0]
    hdf_filter[path] = data_space

