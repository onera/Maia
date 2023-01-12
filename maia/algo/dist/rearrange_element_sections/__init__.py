from cmaia import dist_algo as cdist_algo
from maia.algo.apply_function_to_nodes import apply_to_bases,apply_to_zones

from maia.utils import require_cpp20

@require_cpp20
def rearrange_element_sections(dist_tree, comm):
  """
  Rearanges Elements_t sections such that for each zone,
  sections are ordered in ascending dimensions order
  and there is only one section by ElementType.
  Sections are renamed based on their ElementType.

  The tree is modified in place.
  The Elements_t nodes are guaranteed to be ordered by ascending ElementRange.

  Args:
    dist_tree  (CGNSTree): Tree with an element-based connectivity
    comm       (`MPIComm`): MPI communicator

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #rearrange_element_sections@start
        :end-before: #rearrange_element_sections@end
        :dedent: 2
  """
  apply_to_bases(cdist_algo.rearrange_element_sections, dist_tree, comm)

