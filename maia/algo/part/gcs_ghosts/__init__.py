from cmaia import part_algo as cpart_algo
from maia.algo.apply_function_to_nodes import apply_to_bases,apply_to_zones

def gcs_only_for_ghosts(t):
  apply_to_bases(cpart_algo.gcs_only_for_ghosts, t)

def remove_ghost_info(t):
  apply_to_bases(cpart_algo.remove_ghost_info, t)

