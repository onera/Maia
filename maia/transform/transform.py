from cmaia.transform import transform as ctransform
from maia.transform.apply_function_to_nodes import apply_to_bases,apply_to_zones
import Converter.Internal as I
from mpi4py import MPI


def merge_by_elt_type(dist_tree,comm):
  apply_to_bases(dist_tree,ctransform.merge_by_elt_type,comm)


def remove_ghost_info(t):
  apply_to_bases(t,ctransform.remove_ghost_info)

def add_fsdm_distribution(t,comm):
  apply_to_bases(t,ctransform.add_fsdm_distribution,comm)


def gcs_only_for_ghosts(t):
  apply_to_bases(t,ctransform.gcs_only_for_ghosts)



def put_boundary_first(t,comm):
  apply_to_bases(t,ctransform.put_boundary_first,comm)

def ngon_new_to_old(t):
  apply_to_zones(t,ctransform.ngon_new_to_old)

def split_boundary_subzones_according_to_bcs(t):
  apply_to_bases(t,ctransform.split_boundary_subzones_according_to_bcs)

def generate_interior_faces_and_parents(t,comm):
  apply_to_zones(t,ctransform.generate_interior_faces_and_parents,comm)

def std_elements_to_ngons(t,comm):
  apply_to_zones(t,ctransform.std_elements_to_ngons,comm)
