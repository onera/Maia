import maia.pytree        as PT
import maia.pytree.maia   as MT

from .dist import interpolation as dist_interpolation
from .part import interpolation as part_interpolation

def is_distributed(tree):
  for zone in PT.get_all_Zone_t(tree):
    if MT.getDistribution(zone) is not None:
      return True
  return False

def interpolate(src_tree, tgt_tree, comm, containers_name, location, **options):
  """Interpolate fields between two trees.

  This function can transfer CellCenter or Vertex located fields, but not both
  at the same time.
  Target tree is modified inplace: the requested FlowSolution_t containers are transfered
  from the source tree.

  Interpolation strategy can be controled thought the options kwargs:

  - ``strategy`` (default = 'Closest') -- control interpolation method

    - 'Closest' : Target points use the inverse distance weighting on the ``n_closest_pt`` source point values.
    - 'Location' : For ``CellCenter`` fields, target points take the value of the cell in which they are located.
      For ``Vertex`` fields, target points use finite element weights of source cell vertices to compute interpolation.
      In both cases, unlocated points take the value ``NaN``.
    - 'LocationAndClosest' : Use 'Location' method and then 'ClosestPoint' method
      for the unlocated points.

  - ``n_closest_pt`` (default = 1) -- If strategy is 'Closest' or 'LocationAndClosest', 
    specify the number of closest points used for interpolation.

  - ``loc_tolerance`` (default = 1E-6) -- Geometric tolerance for Location method.

  Inputs trees can be either distributed or partitioned, but both must be of same kind.

  See also:
    :func:`create_interpolator` takes the same parameters (excepted ``containers_name``,
    which must be replaced by ``src_location``), and returns an Interpolator object which can be used
    to exchange containers more than once through its ``Interpolator.exchange_fields(container_name)`` method.

  Args:
    src_tree (CGNSTree): Source tree
    tgt_tree (CGNSTree): Target tree
    comm       (MPIComm): MPI communicator
    containers_name (list of str) : List of the names of the source FlowSolution_t nodes to transfer.
    location ({'CellCenter', 'Vertex'}) : Expected target location of the fields.
    **options: Options related to interpolation strategy

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #interpolate@start
        :end-before: #interpolate@end
        :dedent: 2
  """
  src_dist = is_distributed(src_tree)
  tgt_dist = is_distributed(tgt_tree)

  if src_dist ^ tgt_dist:
    raise ValueError("Source and target tree must be both distributed or partitioned")

  if src_dist:
    dist_interpolation.interpolate(src_tree, tgt_tree, comm, containers_name, location, **options)
  else:
    part_interpolation.interpolate(src_tree, tgt_tree, comm, containers_name, location, **options)



def create_interpolator(src_tree, tgt_tree, comm, src_location, tgt_location, **options):
  """
  Same as interpolate, but return the interpolator object instead
  of doing interpolations. Interpolator can be called multiple time to exchange
  fields without recomputing the src_to_tgt indirection (geometry must remain the same).
  """
  src_dist = is_distributed(src_tree)
  tgt_dist = is_distributed(tgt_tree)

  if src_dist ^ tgt_dist:
    raise ValueError("Source and target tree must be both distributed or partitioned")

  if src_dist:
    return dist_interpolation.create_interpolator(src_tree, tgt_tree, comm, src_location, tgt_location, **options)
  else:
    return part_interpolation.create_interpolator(src_tree, tgt_tree, comm, src_location, tgt_location, **options)