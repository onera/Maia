import cmaia.dist_algo as cdist_algo
from maia.algo.apply_function_to_nodes import apply_to_bases,apply_to_zones

def generate_interior_faces_and_parents(dist_tree,comm):
  apply_to_zones(dist_tree, cdist_algo.generate_interior_faces_and_parents, comm)

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

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #elements_to_ngons@start
        :end-before: #elements_to_ngons@end
        :dedent: 2
  """
  apply_to_zones(dist_tree, cdist_algo.elements_to_ngons, comm)
