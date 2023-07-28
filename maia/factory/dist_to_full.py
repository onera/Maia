import maia.pytree as PT
from maia.io          import distribution_tree
from maia.algo.dist   import redistribute

def reshape_S_arrays(tree):
  """ Some structured arrays (under FlowSolution_t, GridCoordinates_t) have been
  flattened in distributed tree. This function regive them a 2D/3D shape
  """
  for zone in PT.get_all_Zone_t(tree):
    if PT.Zone.Type(zone) == "Structured":
      loc_to_shape = {'Vertex' : PT.Zone.VertexSize(zone), 'CellCenter' : PT.Zone.CellSize(zone)}
      for array in PT.get_nodes_from_predicates(zone, 'GridCoordinates_t/DataArray_t'):
        array[1] = array[1].reshape(loc_to_shape['Vertex'], order='F')
      for container in PT.get_nodes_from_label(zone, 'FlowSolution_t'):
        wanted_shape = loc_to_shape[PT.Subset.GridLocation(container)]
        for array in PT.get_nodes_from_label(container, 'DataArray_t'):
          array[1] = array[1].reshape(wanted_shape, order='F')

def undistribute_tree(dist_tree, comm, target=0):
  """ Generate a standard (full) CGNS Tree from a distributed tree.

  The output tree can be used with sequential tools, but is no more compatible with
  maia parallel algorithms.

  Args:
    dist_tree   (CGNSTree) : Distributed CGNS tree
    comm         (MPIComm) : MPI communicator
    target (int, optional) : MPI rank holding the output tree. Defaults to 0.
  Returns:
    CGNSTree: Full (not distributed) tree or None

  Example:
      .. literalinclude:: snippets/test_factory.py
        :start-after: #undistribute_tree@start
        :end-before: #undistribute_tree@end
        :dedent: 2
  """

  full_tree = PT.deep_copy(dist_tree)

  redistribute.redistribute_tree(full_tree, f'gather.{target}', comm)
  if comm.Get_rank() == target:
    reshape_S_arrays(full_tree)
    distribution_tree.clean_distribution_info(full_tree)
  else:
    full_tree = None

  return full_tree

