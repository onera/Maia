import Converter.PyTree   as C
import Converter.Internal as I
import Converter.Filter   as CFilter


# --------------------------------------------------------------------------
def create_boundary_filter_unstructured(zone_tree, zone_path, cgns_filter):
  """
  """
  raise NotImplementedError

# --------------------------------------------------------------------------
def create_boundary_filter_structured(zone_tree, zone_path, cgns_filter):
  """
  """
  raise NotImplementedError

# --------------------------------------------------------------------------
def create_boundary_filter(zone_tree, zone_path, cgns_filter):
  """
  Create filter for the boundary condition for structured and unstructured meshes
  Args:
    zone_tree (pyTree) : The zone tree (pyTree)
    zone_path (string) : the zone path in order to setup the correct path for filter
    cgns_filter (dict) : for each path store the infomation for load/unload
  """

  zone_type_n = I.getNodeFromType1(zone_tree, 'ZoneType_t')
  zone_type   = zone_type_n[1].tostring()
  if(zone_type == b'Unstructured'):
    create_boundary_filter_unstructured(zone_tree, zone_path, cgns_filter)
  else:
    create_boundary_filter_structured(zone_tree, zone_path, cgns_filter)
    raise NotImplementedError

