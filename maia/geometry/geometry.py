from cmaia.geometry.geometry import *

import numpy as np
from math import cos,sin

import Converter.Internal       as I
import maia.sids.Internal_ext   as IE
import maia.sids.sids           as sids
import maia.utils.py_utils      as py_utils

@IE.check_is_label("Zone_t")
def compute_cell_center(zone):
  """
  Compute the cell centers of a NGon unstructured zone or a structured zone
  and return it as a flat (interlaced) np array. Centers are computed using
  a basic average over the vertices of the cells
  """
  cx, cy, cz = sids.coordinates(zone)

  if sids.Zone.Type(zone) == "Unstructured":
    n_cell     = sids.Zone.n_cell(zone)
    ngons  = [e for e in I.getNodesFromType1(zone, 'Elements_t') if sids.ElementCGNSName(e) == 'NGON_n']
    if len(ngons) != 1:
      raise NotImplementedError(f"Cell center computation is only available for NGON connectivity")
    face_vtx, face_vtx_idx, ngon_pe = sids.ngon_connectivity(zone)
    center_cell = compute_center_cell_u(n_cell, cx, cy, cz, face_vtx, face_vtx_idx, ngon_pe)
  else:
    center_cell = compute_center_cell_s(*sids.Zone.CellSize(zone), cx, cy, cz)

  return center_cell

def transform_cart_matrix(vectors, translation=np.zeros(3), rotation_center=np.zeros(3), rotation_angle=np.zeros(3)):
  """
  Apply the defined cartesian transformation on concatenated components of vectors described by :
  [vx1 vx2 ... vxN]
  [vy1 vy2 ... vyN]
  [vz1 vz2 ... vzN]
  and return the modified components of the vectors in the same format
  """
  rotation_center = np.array(rotation_center).reshape((3,1))
  alpha, beta, gamma  = rotation_angle
  rotation_matx = np.array([[1, 0, 0], [0, cos(alpha), -sin(alpha)], [0, sin(alpha), cos(alpha)]])
  rotation_maty = np.array([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]])
  rotation_matz = np.array([[cos(gamma), -sin(gamma), 0], [sin(gamma), cos(gamma), 0], [0, 0, 1]])
  rotation_mat  = np.dot(rotation_matx, np.dot(rotation_maty, rotation_matz))
  rotated_vectors = ((np.dot(rotation_mat, vectors-rotation_center)+rotation_center).T + translation).T
  return (rotated_vectors.astype(vectors.dtype,copy=False))

def transform_cart_vectors(vx, vy, vz, translation=np.zeros(3), rotation_center=np.zeros(3), rotation_angle=np.zeros(3)):
  """
  Apply the defined cartesian transformation on separated components of vectors and return a tuple with each of the modified components of the vectors
  """
  vectors = np.array([vx,vy,vz], order='F')
  modified_components = transform_cart_matrix(vectors, translation, rotation_center, rotation_angle)
  return (modified_components[0], modified_components[1], modified_components[2])

def transform_zone(zone,
                   rotation_center = np.zeros(3),
                   rotation_angle  = np.zeros(3),
                   translation     = np.zeros(3),
                   apply_to_fields = False):
  """
  Apply the affine transformation to the coordinates of the given zone.
  If apply_to_fields is True, also rotate all the vector fields in CGNS nodes of type
  "FlowSolution_t", "DiscreteData_t", "ZoneSubRegion_t", "BCDataset_t"
  """
  # Transform coords
  for grid_co in I.getNodesFromType1(zone, "GridCoordinates_t"):
    coords_n = [I.getNodeFromName1(grid_co, f"Coordinate{c}")  for c in ['X', 'Y', 'Z']]
    coords = [I.getVal(n) for n in coords_n]
  
    tr_coords = transform_cart_vectors(*coords, translation, rotation_center, rotation_angle)
    for coord_n, tr_coord in zip(coords_n, tr_coords):
      I.setValue(coord_n, tr_coord)

  # Transform fields
  if apply_to_fields:
    fields_nodes  = I.getNodesFromType1(zone, "FlowSolution_t")
    fields_nodes += I.getNodesFromType1(zone, "DiscreteData_t")
    fields_nodes += I.getNodesFromType1(zone, "ZoneSubRegion_t")
    for zoneBC in I.getNodesFromType1(zone, "ZoneBC_t"):
      for bc in I.getNodesFromType1(zoneBC, "BC_t"):
        fields_nodes += I.getNodesFromType1(bc, "BCDataSet_t")
    for fields_node in fields_nodes:
      data_names = [I.getName(data) for data in I.getNodesFromType(fields_node, "DataArray_t")]
      cartesian_vectors_basenames = py_utils.find_cartesian_vector_names(data_names)
      for basename in cartesian_vectors_basenames:
        vectors_n = [I.getNodeFromNameAndType(fields_node, f"{basename}{c}", 'DataArray_t')  for c in ['X', 'Y', 'Z']]
        vectors = [I.getVal(n) for n in vectors_n]
        # Assume that vectors are position independant
        # Be careful, if coordinates vector needs to be transform, the translation is not applied !
        tr_vectors = transform_cart_vectors(*vectors, rotation_center=rotation_center, rotation_angle=rotation_angle)
        for vector_n, tr_vector in zip(vectors_n, tr_vectors):
          I.setValue(vector_n, tr_vector)
