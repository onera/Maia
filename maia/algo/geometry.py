import maia.pytree        as PT
import maia.pytree.maia   as MT
from maia.algo.apply_function_to_nodes import zones_iterator

from .dist import geometry as dist_geometry
from .part import geometry as part_geometry


def _compute_elements_center(zone, dim, comm=None):
  """Dispatch centers computing according to zone dimension and 
  requested dimension """
  if MT.getDistribution(zone) is not None:
    assert comm is not None
    return dist_geometry._compute_elements_center(zone, dim, comm)
  else:
    return part_geometry._compute_elements_center(zone, dim)

def _compute_elements_measure(zone, dim, comm=None):
  """Dispatch measure computing according to zone dimension and 
  requested dimension """
  if MT.getDistribution(zone) is not None:
    assert comm is not None
    return dist_geometry._compute_elements_measure(zone, dim, comm)
  else:
    return part_geometry._compute_elements_measure(zone, dim)
  

def compute_elements_center(t, dim, comm=None):
  """Compute the centers of the specified mesh entity.

  The mesh entity on which centers are computed must be specified using
  ``dim`` parameter: values of 1, 2, and 3 correspond respectively to edges,
  faces and cells. 
  For convenience, the keyword ``CellCenter`` can be used to indicate, on
  each zone, the higher available dimension. The following table summarizes
  the possibilities. Note that some combinations do not make sense (zones in this
  situation are skipped).

  +---------+-------+-------+-------+----------------+
  |         | dim=1 | dim=2 | dim=3 | dim=CellCenter |
  +=========+=======+=======+=======+================+
  | 3D mesh | Edges | Faces | Cells | Cells          |
  +---------+-------+-------+-------+----------------+
  | 2D mesh | Edges | Faces |       | Faces          |
  +---------+-------+-------+-------+----------------+
  | 1D mesh | Edges |       |       | Edges          |
  +---------+-------+-------+-------+----------------+

  Warning:
    For structured meshes, ``dim = 1`` is not yet implemented.

  Centers are computed using a basic average over the vertices of the entity.
  Cartesian and cylindrical coordinates are supported.

  Input tree is modified inplace : results are stored in a
  ``DiscreteData_t`` container named ``Geometry_{1|2|3}d``. Note that for
  unstructured zones described by standard elements, centers are computed
  only for elements explicitly defined in sections.

  Args:
    t    (CGNSTree)            : Tree starting at Zone_t level or higher
    dim  (int or 'CellCenter') : Entity on which centers are computed (see above)
    comm       (MPIComm)       : MPI communicator, mandatory only for distributed trees

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #compute_elements_center@start
        :end-before: #compute_elements_center@end
        :dedent: 2
  """

  for zone in zones_iterator(t):
    
    if MT.getDistribution(zone) is not None:
      assert comm is not None
      dist_geometry.compute_elements_center(zone, dim, comm)
    else:
      part_geometry.compute_elements_center(zone, dim)

def compute_elements_measure(t, dim, comm=None):
  """Compute the length, area or volume of the specified mesh entity.

  As for :func:`compute_elements_center`, the mesh entity on which measures
  are computed must be specified using ``dim`` parameter: values of 1, 2, and 3
  correspond respectively to edges, faces and cells. 
  For convenience, the keyword ``CellCenter`` can be used to indicate, on
  each zone, the higher available dimension. See :func:`compute_elements_center` for the
  summarizing table.
  Note that some combinations do not make sense (zones in this
  situation are skipped).

  Warning:
    - For structured meshes, ``dim = 1`` is not yet implemented.
    - Only cartesian coordinates are supported.

  Input tree is modified inplace : results are stored in a
  ``DiscreteData_t`` container named ``Geometry_{1|2|3}d``. Note that for
  unstructured zones described by standard elements, measures are computed
  only for elements explicitly defined in sections.

  Args:
    t    (CGNSTree)            : Tree starting at Zone_t level or higher
    dim  (int or 'CellCenter') : Entity on which measures are computed (see above)
    comm       (MPIComm)       : MPI communicator, mandatory only for distributed trees

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #compute_elements_measure@start
        :end-before: #compute_elements_measure@end
        :dedent: 2
  """

  for zone in zones_iterator(t):
    
    if MT.getDistribution(zone) is not None:
      assert comm is not None
      dist_geometry.compute_elements_measure(zone, dim, comm)
    else:
      part_geometry.compute_elements_measure(zone, dim)

