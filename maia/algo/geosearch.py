from   maia.typing import *
import maia.pytree        as PT
import maia.pytree.maia   as MT

from .dist import closest_points as dist_closest
from .part import closest_points as part_closest
from .dist import localize as dist_localize
from .part import localize as part_localize
from maia.pytree.maia.check_tree import check_cgns_dist_tree, check_cgns_part_tree
from typing import overload

def is_distributed(tree):
  for zone in PT.get_all_Zone_t(tree):
    if MT.getDistribution(zone) is not None:
      return True
  return False

@overload
def localize_points(src_tree: CGNSDistTree,
                    tgt_tree: CGNSDistTree,
                    location: Literal['CellCenter', 'Vertex'],
                    comm: MPIComm,
                    **options) -> None: ...
@overload
def localize_points(src_tree: CGNSPartTree,
                    tgt_tree: CGNSPartTree,
                    location: Literal['CellCenter', 'Vertex'],
                    comm: MPIComm, 
                    **options) -> None: ...

def localize_points(src_tree: Union[CGNSDistTree, CGNSPartTree],
                    tgt_tree: Union[CGNSDistTree, CGNSPartTree],
                    location: Literal['CellCenter', 'Vertex'],
                    comm: MPIComm, 
                    **options) -> None:
  """Localize points between two trees.

  For all the points of the target tree matching the given location,
  search the cell of the source tree in which it is enclosed.
  The result, i.e. the gnum & domain number of the source cell (or -1 if the point is not localized),
  are stored in a ``DiscreteData_t`` container called "Localization" on the target zones.
  Note that if the source tree is structured, the output gnum is still a scalar index
  and not a (i,j,k) triplet.

  Localization can be parametred thought the options kwargs:

  - ``loc_tolerance`` (default = 1E-6) -- Geometric tolerance for the method.

  Inputs trees can be either distributed or partitionned, but both must be of same kind.

  Args:
    src_tree (CGNSDistTree|CGNSPartTree): Source tree
    tgt_tree (CGNSDistTree|CGNSPartTree): Target tree
    location ({'CellCenter', 'Vertex'}) : Target points to localize
    comm       (MPIComm): MPI communicator
    **options: Additional options related to location strategy

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #localize_points@start
        :end-before: #localize_points@end
        :dedent: 2
  """
  src_dist = is_distributed(src_tree)
  tgt_dist = is_distributed(tgt_tree)

  if src_dist ^ tgt_dist:
    raise ValueError("Source and target tree must be both distributed or partitionned")
  
  if src_dist:
    check_cgns_dist_tree(src_tree)
    check_cgns_dist_tree(tgt_tree)
    dist_localize.localize_points(src_tree, tgt_tree, location, comm, **options)
  else:
    check_cgns_part_tree(src_tree)
    check_cgns_part_tree(tgt_tree)
    part_localize.localize_points(CGNSPartTree(src_tree), CGNSPartTree(tgt_tree), location, comm, **options)

@overload
def find_closest_points(src_tree: CGNSDistTree,
                        tgt_tree: CGNSDistTree,
                        location: Literal['CellCenter', 'Vertex'],
                        comm: MPIComm) -> None: ...
@overload 
def find_closest_points(src_tree: CGNSPartTree,
                        tgt_tree: CGNSPartTree,
                        location: Literal['CellCenter', 'Vertex'],
                        comm: MPIComm) -> None: ...

def find_closest_points(src_tree: Union[CGNSDistTree, CGNSPartTree],
                        tgt_tree: Union[CGNSDistTree, CGNSPartTree],
                        location: Literal['CellCenter', 'Vertex'],
                        comm: MPIComm) -> None:
  """Find the closest points between two trees.

  For all points of the target tree matching the given location,
  search the closest point of same location in the source tree.
  The result, i.e. the gnum & domain number of the source point, are stored in a ``DiscreteData_t``
  container called "ClosestPoint" on the target zones.
  The ids of source points refers to cells or vertices depending on the chosen location.

  Inputs trees can be either distributed or partitionned, but both must be of same kind.

  Args:
    src_tree (CGNSDistTree|CGNSPartTree): Source tree
    tgt_tree (CGNSDistTree|CGNSPartTree): Target tree
    location  ({'CellCenter', 'Vertex'}): Entity to use to compute closest points
    comm      (MPIComm): MPI communicator

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #find_closest_points@start
        :end-before: #find_closest_points@end
        :dedent: 2
  """
  src_dist = is_distributed(src_tree)
  tgt_dist = is_distributed(tgt_tree)

  if src_dist ^ tgt_dist:
    raise ValueError("Source and target tree must be both distributed or partitionned")
  
  if src_dist:
    check_cgns_dist_tree(src_tree)
    check_cgns_dist_tree(tgt_tree)
    dist_closest.find_closest_points(src_tree, tgt_tree, location, comm)
  else:
    check_cgns_part_tree(src_tree)
    check_cgns_part_tree(tgt_tree)
    part_closest.find_closest_points(CGNSPartTree(src_tree), CGNSPartTree(tgt_tree), location, comm)