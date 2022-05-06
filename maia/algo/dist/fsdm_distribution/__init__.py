from cmaia import dist_algo as cdist_algo
from maia.algo.apply_function_to_nodes import apply_to_bases,apply_to_zones

def add_fsdm_distribution(t, comm):
  apply_to_bases(t, cdist_algo.add_fsdm_distribution, comm)

