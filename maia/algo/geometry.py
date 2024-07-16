import maia.pytree        as PT
import maia.pytree.maia   as MT
from maia.algo.apply_function_to_nodes import zones_iterator

from .dist import geometry as dist_geometry
from .part import geometry as part_geometry

def _compute_vol_center(zone, comm=None):
  if MT.getDistribution(zone) is not None:
    assert comm is not None
    return dist_geometry.compute_cell_center(zone, comm)
  else:
    return part_geometry.compute_cell_center(zone)

def _compute_face_center(zone, comm=None):
  if MT.getDistribution(zone) is not None:
    assert comm is not None
    return dist_geometry.compute_face_center(zone, comm)
  else:
    return part_geometry.compute_face_center(zone)

def _compute_edge_center(zone, comm=None):
  if MT.getDistribution(zone) is not None:
    assert comm is not None
    return dist_geometry.compute_edge_center(zone, comm)
  else:
    return part_geometry.compute_edge_center(zone)

def _compute_centers(zone, dim, comm=None):
  """Dispatch centers computing according to zone dimension and 
  requested dimension """
  zone_dim = PT.Zone.CellDimension(zone)
  if dim == 'Cell':
    dim = zone_dim
  if dim == 3 and zone_dim >= 3:
    return _compute_vol_center(zone, comm)
  elif dim == 2 and zone_dim >= 2:
    return _compute_face_center(zone, comm)
  elif dim == 1 and zone_dim >= 1:
    return _compute_edge_center(zone, comm)
  



def compute_centers(t, dim, comm=None, out_fs_name='', method='mean'):
  """Compute the cell centers of a partitioned zone.

  Input zone must have cartesian or cylindrical coordinates recorded under a unique
  GridCoordinates node.
  Centers are computed using a basic average over the vertices of the cells.

  Args:
    t    (CGNSTree(s)): Tree (or sequences of) starting at Zone_t level or higher.
    dim  (int): XXXXX
    comm       (MPIComm) : MPI communicator, mandatory only for distributed trees

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #compute_cell_center@start
        :end-before: #compute_cell_center@end
        :dedent: 2
  """

  """       VolCenter     FaceCenter    EdgeCenter   | CellCenter
  dim 3        X              X             X        |     Volu
  dim 2                       X             X        |     Face
  dim 1                                     X        |     Edge
  
  Maillages (celldim / phydim): 3D(3), 2D(3), 2D(2), 1D(3), 1D(2), 1D(1) --> 6 choix
  Connectivity : Ungon / Uelt / S   ---> 3 choix
  Parallel : Dist / part --> 2 choix 
  Cartésien / Cylindrique --> 2 choix
  Total : 6*3*2*2 = 72 possibilité


  """

  for zone in zones_iterator(t):
    
    if MT.getDistribution(zone) is not None:
      # call distributed
      dist_geometry.compute_zone_centers(zone, dim, comm, out_fs_name, method)
    else:
      part_geometry.compute_zone_centers(zone, dim, out_fs_name, method)
