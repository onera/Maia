import numpy as np

from maia.typing import *
import maia.pytree        as PT
import maia.pytree.maia   as MT
from maia.utils      import np_utils, as_pdm_gnum, layouts

from .geometry       import _compute_elements_center
from .multidom_gnum  import _get_shifted_arrays
import Pypdm.Pypdm as PDM

def _get_zone_ln_to_gn_from_loc(zone: CGNSTree, location: str) -> NDArray:
  """ Wrapper to get the expected lngn value 
  
  Args:
    zone: CGNS zone node
    location: Grid location ('Vertex', 'CellCenter', etc.)
  
  Returns:
    Global numbering array for the specified location
  """
  _loc = location.replace('Center', '')
  ln_to_gn = as_pdm_gnum(PT.get_np_value(MT.requestGlobalNumbering(zone, _loc)))
  return ln_to_gn

def get_point_cloud(zone: CGNSTree, location: str = 'CellCenter') -> Tuple[NDArray, NDArray]:
  """
  If location == Vertex, return the (interlaced) coordinates of vertices 
  and vertex global numbering of a partitioned zone
  If location == Center, compute and return the (interlaced) coordinates of
  cell centers and cell global numbering of a partitioned zone
  
  Args:
    zone: CGNS zone node
    location: Grid location ('Vertex', 'CellCenter', etc.)
  
  Returns:
    Tuple containing:
      - Interlaced coordinates array
      - Global numbering array
  
  Raises:
    RuntimeError: If location is unknown or node not found
  """
  if location == 'Vertex':
    cx,cy,cz = PT.Zone.coordinates(zone)
    assert (cx is not None) and (cy is not None) and (cz is not None)
    coords = [c.reshape(-1, order='F') for c in [cx,cy,cz]]
    vtx_coords   = np_utils.interweave_arrays(coords)
    vtx_ln_to_gn = _get_zone_ln_to_gn_from_loc(zone, location)
    return vtx_coords, vtx_ln_to_gn

  elif location == 'CellCenter':
    cell_ln_to_gn = _get_zone_ln_to_gn_from_loc(zone, location)
    center_cell = _compute_elements_center(zone, 'CellCenter')
    return center_cell, cell_ln_to_gn
  
  else: #Try to catch a container with the given name
    container = PT.get_child_from_name(zone, location)
    if container:
      coords = [PT.get_np_value(c).reshape(-1, order='F') for c in PT.get_children_from_name(container, 'Coordinate*')]
      int_coords = np_utils.interweave_arrays(coords)
      ln_to_gn = _get_zone_ln_to_gn_from_loc(zone, PT.Subset.GridLocation(container))
      return int_coords, ln_to_gn

  raise RuntimeError("Unknow location or node")

def get_shifted_point_clouds(parts_per_dom: List[List[CGNSPartTree]], 
                             location: str,
                             comm: MPIComm) -> Tuple[NDArray, List[List[Tuple[NDArray, NDArray]]]]:
  """ Wraps get_point_cloud around multiple domains,
  shifting lngn with previous values
  
  Args:
    parts_per_dom: List of lists of partitioned zones per domain
    location: Grid location ('Vertex', 'CellCenter', etc.)
    comm: MPI communicator
  
  Returns:
    Tuple containing:
      - Offset value
      - List of lists of (coords, lngn) tuples per domain
  """
  coords_per_dom = []
  lngn_per_dom = []
  for part_zones in parts_per_dom:
    point_clouds_dom = [get_point_cloud(part, location) for part in part_zones]
    coords_per_dom.append([pc[0] for pc in point_clouds_dom])
    lngn_per_dom.append([pc[1] for pc in point_clouds_dom])

  offset, shifted_lgns = _get_shifted_arrays(lngn_per_dom, comm)

  clouds_per_dom = []
  for dom_coords, dom_lngns in zip(coords_per_dom, shifted_lgns):
    clouds_per_dom.append(list(zip(dom_coords, dom_lngns)))
  return offset, clouds_per_dom

def extract_sub_cloud(coords: NDArray, 
                      lngn: NDArray,
                      indices: NDArray) -> Tuple[NDArray, NDArray]:
  """
  Extract coordinates and lngn from a list of indices, starting at 0.
  
  Args:
    coords: Coordinates array
    lngn: Global numbering array
    indices: Indices to extract
  
  Returns:
    Tuple containing:
      - Extracted coordinates array
      - Extracted global numbering array
  """
  sub_lngn   = layouts.extract_from_indices(lngn  , indices, 1, 0)
  sub_coords = layouts.extract_from_indices(coords, indices, 3, 0)
  return sub_coords, sub_lngn

def create_sub_numbering(lngn_l: List[NDArray], comm: MPIComm) -> List[NDArray]:
  """
  Create a new compact, starting at 1 numbering from a list of
  gnums.
  
  Args:
    lngn_l: List of global numbering arrays
    comm: MPI communicator
  
  Returns:
    List of new global numbering arrays
  """
  n_part = len(lngn_l)
  if comm.allreduce(n_part) == 0:
    return []

  gen_gnum = PDM.GlobalNumbering(3, n_part, 0, 0., comm)

  for i_part, lngn in enumerate(lngn_l):
    gen_gnum.set_from_parent(i_part, lngn)

  gen_gnum.compute()

  return [gen_gnum.get(i_part) for i_part in range(n_part)]
