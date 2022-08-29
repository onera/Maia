from cmaia import dist_algo as cdist_algo
from maia.algo.apply_function_to_nodes import apply_to_bases,apply_to_zones

def put_boundary_first(t, comm):
  apply_to_bases(cdist_algo.put_boundary_first, t, comm)

