from cmaia import dist_algo as cdist_algo
from maia.algo.apply_function_to_nodes import apply_to_bases,apply_to_zones

def merge_by_elt_type(dist_tree, comm):
  apply_to_bases(dist_tree, cdist_algo.merge_by_elt_type, comm)

