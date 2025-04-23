import numpy as np

from maia.typing import *
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia       import npy_pdm_gnum_dtype           as pdm_gnum_dtype
from maia.utils import par_utils, layouts
import Pypdm.Pypdm as PDM 

# --------------------------------------------------------------------------
def _dcloud_to_cgns(dpoint_cloud: Dict[str, Any], comm: MPIComm):
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
def dpoint_cloud_cartesian_generate(n_vtx: Union[int, Sequence[int]],
                                    coord_min: Sequence[float],
                                    coord_max: Sequence[float],
                                    comm: MPIComm) -> CGNSDistTree:
  """
  This function calls paradigm to generate a distributed set of points a cloud of points, in a cartesian grid, and
  return a CGNS PyTree
  """

  assert len(coord_min) == len(coord_max), f"Dimension of coord_min ({len(coord_min)}) and coord_max ({len(coord_max)}) must be equal"
  phy_dim = len(coord_min)
  assert phy_dim >= 1

  if isinstance(n_vtx, int): # Expand scalar to list
    n_vtx = phy_dim * [n_vtx]
  elif isinstance(n_vtx, tuple):
    n_vtx = list(n_vtx)
  assert isinstance(n_vtx, list)
  cell_dim = len(n_vtx)
  assert cell_dim <= phy_dim, f"CellDimension ({cell_dim}) can not exceed PhysicalDimension ({phy_dim})"
  
  while (len(n_vtx) > 1 and n_vtx[-1] == 1): # Remove trailing 1 to compute cell_dim
    n_vtx = n_vtx[:-1]
    cell_dim -= 1

  # Complete to fake 3D
  _coord_min = np.empty(3)
  _coord_max = np.empty(3)
  for i in range(phy_dim):
    _coord_min[i] = coord_min[i]
    _coord_max[i] = coord_max[i]
  for i in range(phy_dim, 3):
    _coord_min[i] = 0.
    _coord_max[i] = 0.
  _n_vtx = n_vtx + (3-cell_dim) * [1]


  dpoint_cloud = PDM.dpoint_cloud_gen_cartesian(comm, *_n_vtx, *_coord_min, *_coord_max)
  dist_zone = _dcloud_to_cgns(dpoint_cloud, comm)
  # Easier to create zone as structured
  zsize = np.array([[_n_vtx, 0, 0] for _n_vtx in n_vtx[:cell_dim]], dtype=pdm_gnum_dtype)
  PT.set_value(dist_zone, zsize)
  PT.update_child(dist_zone, 'ZoneType', value='Structured')

  # Remove useless coords if fake 3D was used
  for dir in ['Z', 'Y', 'X'][:3-phy_dim]:
    PT.rm_nodes_from_name(dist_zone, f'Coordinate{dir}')

  # Complete tree and return
  dist_tree = PT.new_CGNSTree()
  # Cell dimension is not specified for a mesh without Elements_t, so we put same value as phy_dim...
  dist_base = PT.new_CGNSBase('Base', cell_dim=cell_dim, phy_dim=phy_dim, parent=dist_tree)
  PT.add_child(dist_base, dist_zone)

  return dist_tree


# --------------------------------------------------------------------------
def dpoint_cloud_random_generate(n_g_pts: int, 
                                 coord_min: List[float], 
                                 coord_max: List[float],
                                 comm: MPIComm, 
                                 seed: Optional[int]=None) -> CGNSTree:
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

def generate_dist_points(n_vtx: Union[int, Sequence[int]], 
                         zone_type: str, 
                         comm: MPIComm, 
                         origin: Sequence[float] = (0,0,0),
                         max_coords: Sequence[float] = (1,1,1)) -> CGNSDistTree:
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
    for base in PT.iter_all_CGNSBase_t(dist_tree):
      assert base[1] is not None
      base[1].fill(base[1][1]) # Update cell_dim to be == to phydim (no proper def in S)
      for zone in PT.iter_all_Zone_t(base):
        zsize = np.array([[PT.Zone.n_vtx(zone), 0, 0]], order='F', dtype=pdm_gnum_dtype)
        PT.set_value(zone, zsize)
        PT.update_child(zone, 'ZoneType', value='Unstructured')
    return dist_tree
  elif zone_type in ["Structured", "S"]:
    return dist_tree
  else:
    raise ValueError(f"Unexpected value for zone_type parameter : {zone_type}")

