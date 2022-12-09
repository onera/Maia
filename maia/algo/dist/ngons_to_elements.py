from cmaia import dist_algo as cdist_algo
from maia.algo.apply_function_to_nodes import apply_to_zones


def ngons_to_elements(t,comm):
  """
  Transform a polyedric (NGon) based connectivity into a standard nodal
  connectivity.
  
  Tree is modified in place : Polyedric element are removed from the zones
  and Pointlist (under the BC_t nodes) are updated.

  Requirement : polygonal elements are supposed to describe only standard
  elements (ie tris, quads, tets, pyras, prisms and hexas)

  WARNING: this function has not been parallelized yet

  Args:
    disttree   (CGNSTree): Tree with connectivity described by NGons
    comm       (`MPIComm`) : MPI communicator
  """
  if (comm.Get_size() > 1):
    raise RuntimeError("WARNING: this function has not been parallelized yet. Run it on only one process")
  apply_to_zones(cdist_algo.convert_zone_to_std_elements, t)

