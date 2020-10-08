import Converter.PyTree   as C
import Converter.Internal as I
import maia.utils         as UTL

from   .distribution_function import create_distribution_node

def compute_elements_distribution(zone, comm):
  """
  """
  zone_type_n = I.getNodeFromType1(zone, 'ZoneType_t')
  zone_type   = zone_type_n[1].to_string()
  if(zone_type == b'Structured'):
    return

  elmts = I.getNodesFromType1(zone, 'Elements_t')

  for elmt in elmts:
    print(elmt)
