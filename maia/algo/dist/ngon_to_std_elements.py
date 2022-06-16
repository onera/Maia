from cmaia import dist_algo as cdist_algo
from maia.algo.apply_function_to_nodes import apply_to_zones


def convert_ngon_to_std_elements(t):
  """
  Transform a polyedric (NGon) based connectivity into a standard nodal
  connectivity.
  
  Tree is modified in place : Polyedric element are removed from the zones
  and Pointlist (under the BC_t nodes) are updated.

  Requirement : polygonal elements are supposed to describe only standard
  elements (ie tris, quads, tets, pyras, prisms and hexas)

  Args:
    disttree   (CGNSTree): Tree with connectivity described by NGons
    comm       (`MPIComm`) : MPI communicator
  """
  apply_to_zones(t,cdist_algo.convert_zone_to_std_elements)

