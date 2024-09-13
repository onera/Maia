import numpy as np

import maia.pytree as PT

from maia.utils import np_utils

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

def compute_face_circulation(local_coords, face_vtx_idx, face_vtx_n):
  """
  Compute, for each face, the term xF.nF|F| where xF is the face mean center, nF the unit outward normal
  and |F| the area of the face.
  Coordinates are supposed to be already expended, following face_vtx array.
  """
  local_coords_next = [np_utils.roll_once_by_stride(face_vtx_idx, coords) for coords in local_coords]

  _local_coords      = np.stack(local_coords, axis=1)
  _local_coords_next = np.stack(local_coords_next, axis=1)
  center = np.add.reduceat(_local_coords, face_vtx_idx[:-1]) / face_vtx_n.reshape((-1,1))

  # Compute mean normal flux on each face : ½ || sum_i CV_i ⨯ CV_{i+1}|| (C := face center)
  reps = np_utils.repeated_arange(face_vtx_n) # To access face center
  face_center_reps = center[reps]
  crossprod = np.cross(_local_coords - face_center_reps, _local_coords_next - face_center_reps)
  normalflux = 0.5*np.add.reduceat(crossprod, face_vtx_idx[:-1])

  face_contrib = np.sum(center*normalflux, axis=1) # Scalar product face_center * normal_flux
  return face_contrib

def update_container(zone, container_name, loc, fields={}):
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
