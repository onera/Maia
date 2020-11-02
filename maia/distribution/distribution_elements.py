import Converter.PyTree   as C
import Converter.Internal as I

from   .distribution_function import create_distribution_node

def compute_elements_distribution(zone, comm):
  """
  """
  zone_type_n = I.getNodeFromType1(zone, 'ZoneType_t')
  zone_type   = zone_type_n[1].tostring()
  if(zone_type == b'Structured'):
    pass
  else:
    elts = I.getNodesFromType1(zone, 'Elements_t')

    for elt in elts:
      er = I.getNodeFromName(elt, 'ElementRange')
      n_tot_elmt = er[1][1] - er[1][0] + 1
      create_distribution_node(n_tot_elmt, comm, 'Distribution', elt)
