import Converter.Internal as I
from .              import utils
from .hdf_dataspace import create_data_array_filter

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
      distrib_ia = I.getNodeFromName1(distrib_ud, 'Distribution')[1]

      gc_shape   = utils.pl_or_pr_size(gc)
      data_space = create_data_array_filter(distrib_ia, gc_shape)
      utils.apply_dataspace_to_pointlist(gc, gc_path, data_space, hdf_filter)
