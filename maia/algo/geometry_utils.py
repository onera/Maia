import numpy as np

from maia.typing import *
import maia.pytree as PT
from typing import Dict, List, Optional, Tuple, Any
from maia.utils import np_utils
from maia.utils import vstride as vs

# For each cell_dimension, list of output GridLocation depending of requested dim argument
DIM_TO_LOC = {3: ['Vertex', 'EdgeCenter', 'FaceCenter', 'CellCenter'],
              2: ['Vertex', 'EdgeCenter', 'CellCenter',  None],
              1: ['Vertex', 'CellCenter',  None,         None],}

# Definition of face_vtx of each basic element
ELT_FACE_VTX = {'TETRA_4' : (np.array([3,3,3,3], np.int32),
                             np.array([1,3,2, 1,2,4, 2,3,4, 3,1,4]) - 1),
                'PYRA_5'  : (np.array([4,3,3,3,3], np.int32),
                             np.array([1,4,3,2, 1,2,5, 2,3,5, 3,4,5, 4,1,5]) - 1),
                'PENTA_6' : (np.array([4,4,4,3,3], np.int32),
                             np.array([1,2,5,4, 2,3,6,5 ,3,1,4,6, 1,3,2, 4,5,6]) - 1),
                'HEXA_8'  : (np.array([4,4,4,4,4,4], np.int32),
                             np.array([1,4,3,2, 1,2,6,5 ,2,3,7,6, 3,4,8,7, 1,5,8,4, 5,6,7,8]) - 1)
                }

def compute_center_and_flux(local_coords: List[Optional [ArrayLike]],
                            face_vtx_idx: np.ndarray,
                            face_vtx_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """
  Compute, for each face, the term nF|F| where nF is the unit outward normal
  and |F| the area of the face.
  Also return face meancenter to save computations
  Coordinates are supposed to be already expended, following face_vtx array.
  """

  # Filter void coords if phy_dim == 2
  local_coords = [c for c in local_coords if c is not None]

  local_coords_next = [vs.roll(vs.from_displs(face_vtx_idx, coords), -1, vs.INNER_AXIS).values for coords in local_coords]

  if len(local_coords) == 2 : # Complete with 0 if phy_dim == 2
    local_coords.append(np.zeros_like(local_coords[0]))
    local_coords_next.append(np.zeros_like(local_coords[0]))

  _local_coords      = np.stack(local_coords, axis=1)
  _local_coords_next = np.stack(local_coords_next, axis=1)
  center = np.add.reduceat(_local_coords, face_vtx_idx[:-1]) / face_vtx_n.reshape((-1,1))

  # Compute mean normal flux on each face : ½ || sum_i CV_i ⨯ CV_{i+1}|| (C := face center)
  reps = np_utils.repeated_arange(face_vtx_n) # To access face center
  face_center_reps = center[reps]
  crossprod = np.cross(_local_coords - face_center_reps, _local_coords_next - face_center_reps)
  normalflux = 0.5*np.add.reduceat(crossprod, face_vtx_idx[:-1])

  return center, normalflux

def update_container(zone: CGNSTree, 
                     container_name: str,
                     loc: str,
                     fields: Dict[str, ArrayLike]={}) -> Optional[CGNSTree]:
  """ Utility to retrieve a container from its name, or create it """
  container = PT.get_child_from_name(zone, container_name)
  if container is not None: # Container exists
    cnt_loc = PT.Subset.GridLocation(container) 
    if cnt_loc != loc:
      raise RuntimeError(f"Container {container_name} already exists in zone "
                         f"{PT.get_name(zone)} but has incompatible GridLocation "
                         f"(expected {loc}, found {cnt_loc})")
  else:  # Create container
    container = PT.new_child(zone, container_name, 'DiscreteData_t')
    PT.new_GridLocation(loc, container)

  for name, array in fields.items():
    PT.update_child(container, name, 'DataArray_t', array)

  return container
