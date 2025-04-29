from maia.typing import *

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.io          import distribution_tree
from maia.algo.dist   import redistribute

def _reshape_S_arrays(tree: CGNSTree) -> None:
  """ Some structured arrays (under FlowSolution_t, GridCoordinates_t) have been
  flattened in distributed tree. This function regive them a 2D/3D shape
  """
  for zone in PT.get_all_Zone_t(tree):
    if PT.Zone.Type(zone) == "Structured":
      loc_to_shape = {'Vertex' : PT.Zone.VertexSize(zone), 'CellCenter' : PT.Zone.CellSize(zone)}
      for array in PT.get_nodes_from_predicates(zone, 'GridCoordinates_t/DataArray_t'):
        assert (array_val:=array[1]) is not None
        PT.set_value(array, array_val.reshape(loc_to_shape['Vertex'], order='F'))
      for container in PT.get_nodes_from_label(zone, 'FlowSolution_t'):
        wanted_shape = loc_to_shape[PT.Subset.GridLocation(container)]
        for array in PT.get_nodes_from_label(container, 'DataArray_t'):
          assert (array_val:=array[1]) is not None
          PT.set_value(array, array_val.reshape(wanted_shape, order='F'))

def dist_to_full_tree(dist_tree: CGNSDistTree, 
                      comm: MPIComm, 
                      target: int = 0) -> Optional[CGNSTree]:
  """ Generate a standard (full) CGNS Tree from a distributed tree.

  The output tree can be used with sequential tools, but is no more compatible with
  maia parallel algorithms.

  Args:
    dist_tree   (CGNSDistTree) : Distributed CGNS tree
    comm         (MPIComm)     : MPI communicator
    target (int, optional)     : MPI rank holding the output tree. Defaults to 0.
  Returns:
    CGNSTree: Full (not distributed) tree or None

  Example:
      .. literalinclude:: snippets/test_factory.py
        :start-after: #dist_to_full_tree@start
        :end-before: #dist_to_full_tree@end
        :dedent: 2
  """
  MT.check_cgns_dist_tree(dist_tree)
  _dist_tree = PT.deep_copy(dist_tree)

  redistribute.redistribute_tree(_dist_tree, f'gather.{target}', comm)
  if comm.Get_rank() == target:
    _reshape_S_arrays(_dist_tree)
    distribution_tree.clean_distribution_info(_dist_tree)
    full_tree = _dist_tree
  else:
    full_tree = None

  return full_tree

