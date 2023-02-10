import cmaia.dist_algo as cdist_algo
from maia.algo.apply_function_to_nodes import apply_to_bases,apply_to_zones

from maia.utils import require_cpp20

@require_cpp20
def generate_interior_faces_and_parents(dist_tree,comm):
  apply_to_zones(cdist_algo.generate_interior_faces_and_parents, dist_tree, comm)

@require_cpp20
def elements_to_ngons(dist_tree,comm):
  """
  Transform an element based connectivity into a polyedric (NGon based)
  connectivity.

  The tree is modified in place: standard elements are removed from the zones
  and PointLists are updated.

  Requirement: the ``Element_t`` nodes must be divided into:
  first 2D element sections, then 3D element sections

  See details :ref:`here <elements_to_ngons_impl>`

  Args:
    dist_tree  (CGNSTree): Tree with an element-based connectivity
    comm       (`MPIComm`): MPI communicator
  """
  apply_to_zones(cdist_algo.elements_to_ngons, dist_tree, comm)
