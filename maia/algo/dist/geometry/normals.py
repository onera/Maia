import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

import maia

from maia.utils     import logging as mlog
from maia.utils     import np_utils

from maia.algo.dist import connectivity_utils as CU
from maia.algo.dist import s_to_u             as S2U

from .utils import get_local_coordinates, place_in_container

import cmaia.part_algo as cpart_algo


def compute_face_normal(zone, comm, unitary=False):
  """
  Compute the face normal of a distributed zone, for phydim = 3
  """
  zone_dim = PT.Zone.CellDimension(zone)
  phy_dim  = PT.Zone.PhysicalDimension(zone)
  assert zone_dim >= 2, "CellDimension of zone must be >= 2 to compute face normals"
  assert phy_dim  == 3, "PhysicalDimension of zone must be 3 to compute face normals"


  # Get face_vtx
  if PT.Zone.Type(zone) == "Unstructured":
    # Careful : if zone is poly2d, the ngon element may be absent
    if PT.pred.IS_POLY2D_ZONE(zone) and not PT.Zone.has_ngon_elements(zone):
      maia.algo.edge_pe_to_ngon(zone, comm)
    if PT.Zone.has_ngon_elements(zone):
      ngon_node = PT.Zone.NGonNode(zone)
      face_vtx = MT.Element.connectivity(ngon_node)
    else: # Zone has std elements
      global_distri = (zone_dim == 2)
      face_vtx = CU.entity_vtx_connectivity_elt(zone, comm, 2, global_distri)
  elif PT.Zone.Type(zone) == 'Structured':
    if zone_dim == 3:
      ngon_node = S2U.zonedims_to_ngon(PT.Zone.VertexSize(zone), comm)
      face_vtx = MT.Element.connectivity(ngon_node)
    elif zone_dim == 2:
      face_vtx = CU.cell_vtx_connectivity_S(zone, zone_dim)

  local_coords = get_local_coordinates(zone, face_vtx.values, comm)
  
  face_normal = cpart_algo.compute_face_normal_u(face_vtx.displs.astype(np.int32, copy=False), *local_coords)

  if unitary:
    np_utils.normalize_interweaved_inplace(face_normal, 3)

  return face_normal

def compute_edge_normal(zone, comm, unitary=False):
  """
  Compute the face normal of a distributed zone, for phydim = 2
  """
  zone_dim = PT.Zone.CellDimension(zone)
  phy_dim  = PT.Zone.PhysicalDimension(zone)
  assert zone_dim in [1,2], "CellDimension of zone must be >= 2 to compute face normals"
  assert phy_dim  == 2, "PhysicalDimension of zone must be 3 to compute face normals"


  # Get face_vtx
  if PT.Zone.Type(zone) == "Unstructured":
    if PT.pred.IS_POLY2D_ZONE(zone):
      edge_node = MT.Zone.EdgeNode(zone)
      edge_vtx = MT.Element.connectivity(edge_node)
    else: # Zone has std elements
      global_distri = (zone_dim == 1)
      edge_vtx = CU.entity_vtx_connectivity_elt(zone, comm, 1, global_distri)
  elif PT.Zone.Type(zone) == 'Structured':
    if zone_dim == 2:
      edge_node = S2U.zonedims_to_ngon(PT.Zone.VertexSize(zone), comm)
      edge_vtx = MT.Element.connectivity(edge_node)
    if zone_dim == 1:
      edge_vtx = CU.cell_vtx_connectivity_S(zone, zone_dim)

  local_coords = get_local_coordinates(zone, edge_vtx.values, comm)

  edge_normal = np.empty(2*len(edge_vtx))
  edge_normal[0::2] = local_coords[1][1::2] - local_coords[1][0::2] # nx =   yb - ya
  edge_normal[1::2] = local_coords[0][0::2] - local_coords[0][1::2] # ny = -(xb - xa)

  if unitary:
    np_utils.normalize_interweaved_inplace(edge_normal, 2)

  return edge_normal

def _compute_elements_normal(zone, comm, unitary=False):
  """
  Distributed implementation of _compute_elements_normal, which compute normal vectors
  and return a raw vector (phydim component per entity)
  """
  cell_dim = PT.Zone.CellDimension(zone)
  phy_dim = PT.Zone.PhysicalDimension(zone)
  if phy_dim == 3 and cell_dim >= 2:
    return compute_face_normal(zone, comm, unitary)
  elif phy_dim == 2 and cell_dim <= 2:
    return compute_edge_normal(zone, comm, unitary)


def compute_elements_normal(zone, comm, unitary=False):
  """
  Distributed implementation of compute_elements_normal, which compute normal vectors
  and add the result in tree
  """
  phy_dim  = PT.Zone.PhysicalDimension(zone)
  interlaced_normal = _compute_elements_normal(zone, comm, unitary)
  basename = 'UnitNormal' if unitary else 'Normal'
  if interlaced_normal is None:
    msg = f"Zone '{PT.get_name(zone)}' skipped during normal computing because "\
          f"its physical dimension is too low (phy_dim={phy_dim})"
    mlog.warning(msg)
  else:
    vectors = {f'{basename}{d}' : interlaced_normal[i::phy_dim] \
               for i,d in enumerate('XYZ'[:phy_dim])}
    place_in_container(zone, phy_dim-1, vectors, comm)

