from maia.algo.apply_function_to_nodes import apply_to_bases,apply_to_zones
from cmaia import tree_algo as ctree_algo

from .import dist, part

from .indexing import pe_to_nface, nface_to_pe

def ngon_new_to_old(t):
  apply_to_zones(t, ctree_algo.ngon_new_to_old)

