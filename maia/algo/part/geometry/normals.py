from mpi4py import MPI
import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

import maia

from maia.utils     import logging as mlog
from maia.utils     import np_utils

from maia.algo.part import connectivity_utils as CU
from maia.algo.dist import s_to_u             as S2U

from .utils import place_in_container

import cmaia.part_algo as cpart_algo


def compute_face_normal(zone, unitary=False):
  """
  Compute the face normal of a partitioned zone, for phydim = 3
  """
  zone_dim = PT.Zone.CellDimension(zone)
  phy_dim  = PT.Zone.PhysicalDimension(zone)
  assert zone_dim >= 2, "CellDimension of zone must be >= 2 to compute face normals"
  assert  phy_dim == 3, "PhysicalDimension of zone must be 3 to compute face normals"


  # Get face_vtx
  if PT.Zone.Type(zone) == "Unstructured":
    # Careful : if zone is poly2d, the ngon element may be absent
    if PT.pred.IS_POLY2D_ZONE(zone) and not PT.Zone.has_ngon_elements(zone):
      maia.algo.edge_pe_to_ngon(zone, None)
    if PT.Zone.has_ngon_elements(zone):
      ngon_node = PT.Zone.NGonNode(zone)
      face_vtx = MT.Element.connectivity(ngon_node)
    else: # Zone has std elements
      face_vtx = CU.cell_vtx_connectivity(zone, 2)
  elif PT.Zone.Type(zone) == 'Structured':
    if zone_dim == 3:
      ngon_node = S2U.zonedims_to_ngon(PT.Zone.VertexSize(zone), MPI.COMM_SELF)
      face_vtx = MT.Element.connectivity(ngon_node)
    elif zone_dim == 2:
      face_vtx = CU.cell_vtx_connectivity_S(zone, zone_dim)

  coords = PT.Zone.coordinates(zone)
  assert isinstance(coords, PT.CartesianCoordinates)
  if PT.Zone.Type(zone) == 'Structured':
    coords = [c.flatten(order='F') for c in coords]
  _face_vtx_values = face_vtx.values-1
  extended_coords = [coord[_face_vtx_values] for coord in coords]
  
  face_normal = cpart_algo.compute_face_normal_u(face_vtx.displs.astype(np.int32, copy=False), *extended_coords)

  if unitary:
    np_utils.normalize_interweaved_inplace(face_normal, 3)

  return face_normal

def compute_edge_normal(zone, unitary=False):
  """
  Compute the face normal of a partitioned zone, for phydim = 2
  """
  zone_dim = PT.Zone.CellDimension(zone)
  phy_dim  = PT.Zone.PhysicalDimension(zone)
  assert zone_dim in [1,2], "CellDimension of zone must be >= 2 to compute face normals"
  assert phy_dim == 2, "PhysicalDimension of zone must be 3 to compute face normals"


  # Get face_vtx
  if PT.Zone.Type(zone) == "Unstructured":
    if PT.pred.IS_POLY2D_ZONE(zone):
      edge_node = MT.Zone.EdgeNode(zone)
      edge_vtx = MT.Element.connectivity(edge_node)
    else: # Zone has std elements
      edge_vtx = CU.cell_vtx_connectivity(zone, 1)
  elif PT.Zone.Type(zone) == 'Structured':
    if zone_dim == 2:
      edge_node = S2U.zonedims_to_ngon(PT.Zone.VertexSize(zone), MPI.COMM_SELF)
      edge_vtx = MT.Element.connectivity(edge_node)
    if zone_dim == 1:
      edge_vtx = CU.cell_vtx_connectivity_S(zone, zone_dim)

  coords = PT.Zone.coordinates(zone)
  assert isinstance(coords, PT.CartesianCoordinates)
  coords = [coords[0], coords[1]] # Remove Z since PhyDim is 2
  if PT.Zone.Type(zone) == 'Structured':
    coords = [c.flatten(order='F') for c in coords]

  _edge_vtx_values = edge_vtx.values-1
  extended_coords = [coord[_edge_vtx_values] for coord in coords]

  edge_normal = np.empty(2*len(edge_vtx))
  edge_normal[0::2] = extended_coords[1][1::2] - extended_coords[1][0::2] # nx =   yb - ya
  edge_normal[1::2] = extended_coords[0][0::2] - extended_coords[0][1::2] # ny = -(xb - xa)

  if unitary:
    np_utils.normalize_interweaved_inplace(edge_normal, 2)

  return edge_normal

def _compute_elements_normal(zone, unitary=False):
  """
  Partitioned implementation of _compute_elements_normal, which compute normal vectors
  and return a raw vector (phydim component per entity)
  """
  cell_dim = PT.Zone.CellDimension(zone)
  phy_dim = PT.Zone.PhysicalDimension(zone)
  if phy_dim == 3 and cell_dim >= 2:
    return compute_face_normal(zone, unitary)
  elif phy_dim == 2 and cell_dim <= 2:
    return compute_edge_normal(zone, unitary)


def compute_elements_normal(zone, unitary=False):
  """
  Partitioned implementation of compute_elements_normal, which compute normal vectors
  and add the result in tree
  """
  phy_dim  = PT.Zone.PhysicalDimension(zone)
  interlaced_normal = _compute_elements_normal(zone, unitary)
  basename = 'UnitNormal' if unitary else 'Normal'
  if interlaced_normal is None:
    msg = f"Zone '{PT.get_name(zone)}' skipped during normal computing because "\
          f"its physical dimension is too low (phy_dim={phy_dim})"
    mlog.warning(msg)
  elif interlaced_normal.size > 0:
    vectors = {f'{basename}{d}' : interlaced_normal[i::phy_dim] \
               for i,d in enumerate('XYZ'[:phy_dim])}
    place_in_container(zone, phy_dim-1, vectors)

