import numpy as np
import Pypdm.Pypdm as PDM

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils import par_utils, layouts

# --------------------------------------------------------------------------
def _dcloud_to_cgns(dpoint_cloud, comm):
  """
  """
  # > Generate dist_tree
  n_g_vtx = dpoint_cloud['np_distrib_pts'][comm.size]
  dist_zone = PT.new_Zone('zone', size=[[n_g_vtx, 0, 0]], type='Unstructured')

  # > Grid coordinates
  cx, cy, cz = layouts.interlaced_to_tuple_coords(dpoint_cloud['np_dpts_coord'])
  coords = {'CoordinateX' : cx, 'CoordinateY' : cy, 'CoordinateZ' : cz}
  grid_coord = PT.new_GridCoordinates(fields=coords, parent=dist_zone)

  np_distrib_pts  = par_utils.full_to_partial_distribution(dpoint_cloud['np_distrib_pts'], comm)
  MT.newDistribution({'Vertex' : np_distrib_pts, 'Cell' : np.zeros(3, np_distrib_pts.dtype)}, parent=dist_zone)

  return dist_zone

# --------------------------------------------------------------------------
def dpoint_cloud_cartesian_generate(n_vtx, coord_min, coord_max, comm):
  """
  This function calls paradigm to generate a distributed set of points a cloud of points, in a cartesian grid, and
  return a CGNS PyTree
  """

  assert len(coord_min) == len(coord_max)
  cloud_dim = len(coord_min)
  assert cloud_dim >= 1

  if isinstance(n_vtx, int): # Expand scalar to list
    n_vtx = cloud_dim * [n_vtx]
  assert isinstance(n_vtx, list)

  # Complete to fake 3D
  _coord_min = np.empty(3)
  _coord_max = np.empty(3)
  for i in range(cloud_dim):
    _coord_min[i] = coord_min[i]
    _coord_max[i] = coord_max[i]
  for i in range(cloud_dim, 3):
    _coord_min[i] = 0.
    _coord_max[i] = 0.
  _n_vtx = n_vtx + (3-cloud_dim) * [1]

  # Adjust coords if only one vtx is requested
  for i in range(cloud_dim):
    if n_vtx[i] == 1:
      _coord_min[i] = 0.5*(coord_min[i] + coord_max[i])
      _coord_max[i] = _coord_min[i]

  dpoint_cloud = PDM.dpoint_cloud_gen_cartesian(comm, *_n_vtx, *_coord_min, *_coord_max)
  dist_zone = _dcloud_to_cgns(dpoint_cloud, comm)

  # Remove useless coords if fake 3D was used
  for dir in ['Z', 'Y', 'X'][:3-cloud_dim]:
    PT.rm_nodes_from_name(dist_zone, f'Coordinate{dir}')

  # Complete tree and return
  dist_tree = PT.new_CGNSTree()
  dist_base = PT.new_CGNSBase('Base', cell_dim=cloud_dim, phy_dim=cloud_dim, parent=dist_tree)
  PT.add_child(dist_base, dist_zone)

  return dist_tree


# --------------------------------------------------------------------------
def dpoint_cloud_random_generate(n_g_pts, coord_min, coord_max, comm, seed=None):
  """
  This function calls paradigm to generate a distributed set of points a cloud of points, in a random way and
  return a CGNS PyTree
  """
  assert len(coord_min) == len(coord_max)
  cloud_dim = len(coord_min)

  # Complete to fake 3D
  _coord_min = np.empty(3)
  _coord_max = np.empty(3)
  for i in range(cloud_dim):
    _coord_min[i] = coord_min[i]
    _coord_max[i] = coord_max[i]
  for i in range(cloud_dim, 3):
    _coord_min[i] = 0.
    _coord_max[i] = 0.

  if seed is None:
    seed = np.random.randint(np.iinfo(np.int32).max, dtype=np.int32)

  dpoint_cloud = PDM.dpoint_cloud_gen_random(comm, seed, n_g_pts, *_coord_min, *_coord_max)
  dist_zone = _dcloud_to_cgns(dpoint_cloud, comm)

  # Remove useless coords if fake 3D was used
  for dir in ['Z', 'Y', 'X'][:3-cloud_dim]:
    PT.rm_nodes_from_name(dist_zone, f'Coordinate{dir}')

  dist_tree = PT.new_CGNSTree()
  dist_base = PT.new_CGNSBase('Base', cell_dim=cloud_dim, phy_dim=cloud_dim, parent=dist_tree)
  PT.add_child(dist_base, dist_zone)

  return dist_tree

def generate_dist_points(n_vtx, zone_type, comm, origin=np.zeros(3), max_coords=np.ones(3)):
  """Generate a distributed mesh including only cartesian points.
  
  Returns a distributed CGNSTree containing a single :cgns:`CGNSBase_t` and
  :cgns:`Zone_t`. The kind 
  of the zone is controled by the ``zone_type`` parameter: 

  - ``"Structured"`` (or ``"S"``) produces a structured zone
  - ``"Unstructured"`` (or ``"U"``) produces an unstructured zone

  In all cases, the created zone contains only the cartesian grid coordinates; no connectivities are created.
  The `physical dimension <https://cgns.github.io/CGNS_docs_current/sids/cgnsbase.html#CGNSBase>`_ of the output
  is set equal to the length of the origin parameter.

  Args:
    n_vtx (int or array of int) : Number of vertices in each direction. Scalars
      automatically extend to uniform array.
    zone_type (str) : requested kind of points cloud
    comm       (MPIComm) : MPI communicator
    origin (array, optional) : Coordinates of the origin of the generated mesh. Defaults
        to zero vector.
    max_coords (array, optional) : Coordinates of the higher point of the generated mesh. Defaults to ones vector.
  Returns:
    CGNSTree: distributed cgns tree

  Example:
      .. literalinclude:: snippets/test_factory.py
        :start-after: #generate_dist_points@start
        :end-before: #generate_dist_points@end
        :dedent: 2
  """

  dist_tree = dpoint_cloud_cartesian_generate(n_vtx, origin, max_coords, comm)

  if zone_type in ["Unstructured", "U"]:
    return dist_tree
  elif zone_type in ["Structured", "S"]:
    for zone in PT.iter_all_Zone_t(dist_tree):
      if isinstance(n_vtx, int):
        n_vtx = len(origin) * [n_vtx]
      zsize = [[_n_vtx, 0, 0] for _n_vtx in n_vtx]
      PT.set_value(zone, zsize)
      PT.update_child(zone, 'ZoneType', value='Structured')
    return dist_tree
  else:
    raise ValueError(f"Unexpected value for zone_type parameter : {zone_type}")

