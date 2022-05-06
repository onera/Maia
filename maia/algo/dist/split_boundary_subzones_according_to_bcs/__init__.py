from cmaia import dist_algo as cdist_algo
from maia.algo.apply_function_to_nodes import apply_to_bases,apply_to_zones

def split_boundary_subzones_according_to_bcs(t):
  apply_to_bases(t, cdist_algo.split_boundary_subzones_according_to_bcs)

