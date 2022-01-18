from cmaia.geometry.geometry import *

import numpy as np
from math import cos,sin

import Converter.Internal       as I
import maia.sids.Internal_ext   as IE
import maia.sids.sids           as sids

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

# def apply_transformation_on_concatenated_coords(rotation_center, rotation_angle, translation, coords):
#   """
#   Apply the defined transformation on concatenated coords described by :
#   [cx1 cx2 ... cxN]
#   [cy1 cy2 ... cyN]
#   [cz1 cz2 ... czN]
#   and return the modified coordinates in the same format
#   """
#   rotation_center = np.array(rotation_center).reshape((3,1))
#   alpha, beta, gamma  = rotation_angle
#   rotation_matx = np.array([[1, 0, 0], [0, cos(alpha), -sin(alpha)], [0, sin(alpha), cos(alpha)]])
#   rotation_maty = np.array([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]])
#   rotation_matz = np.array([[cos(gamma), -sin(gamma), 0], [sin(gamma), cos(gamma), 0], [0, 0, 1]])
#   rotation_mat  = np.dot(rotation_matx, np.dot(rotation_maty, rotation_matz))
#   return (((np.dot(rotation_mat, coords-rotation_center)+rotation_center).T + translation).T)

# def apply_transformation_on_separated_coords(rotation_center, rotation_angle, translation, cx, cy, cz):
#   """
#   Apply the defined transformation on coordinates and return a tuple with each of the modified coordinates
#   """
#   coords = np.array([cx,cy,cz], order='F')
#   modified_coords = apply_transformation_on_concatenated_coords(rotation_center, rotation_angle, translation, coords)
#   return (modified_coords[0],modified_coords[1],modified_coords[2])

def apply_transformation_on_concatenated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, vectors):
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
  return (((np.dot(rotation_mat, vectors-rotation_center)+rotation_center).T + translation).T)

def apply_transformation_on_separated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, vx, vy, vz):
  """
  Apply the defined cartesian transformation on separated components of vectors and return a tuple with each of the modified components of the vectors
  """
  vectors = np.array([vx,vy,vz], order='F')
  modified_components = apply_transformation_on_concatenated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, vectors)
  return (modified_components[0],modified_components[1],modified_components[2])
