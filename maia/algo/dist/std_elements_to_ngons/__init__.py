import cmaia.dist_algo as cdist_algo
from maia.algo.apply_function_to_nodes import apply_to_bases,apply_to_zones

def generate_interior_faces_and_parents(t,comm):
  apply_to_zones(t, cdist_algo.generate_interior_faces_and_parents, comm)

def convert_std_elements_to_ngons(t,comm):
  apply_to_zones(t, cdist_algo.std_elements_to_ngons, comm)

