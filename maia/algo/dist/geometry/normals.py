import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

import maia

from maia.utils     import logging as mlog
from maia.utils     import np_utils, s_numbering

from maia.algo.dist import connectivity_utils as CU
from maia.algo.dist import s_to_u             as S2U

from maia.transfer  import protocols as EP

from .utils import get_local_coordinates, place_in_container

import cmaia.part_algo as cpart_algo


def compute_face_normal(zone, comm, unitary=False, face_indices=None, face_indices_loc=None):
  """
  Compute the face normal of a distributed zone, for phydim = 3
  """
  zone_dim = PT.Zone.CellDimension(zone)
  phy_dim  = PT.Zone.PhysicalDimension(zone)
  assert zone_dim >= 2, "CellDimension of zone must be >= 2 to compute face normals"
  assert phy_dim  == 3, "PhysicalDimension of zone must be 3 to compute face normals"

  if face_indices is not None:
    assert isinstance(face_indices, np.ndarray) and face_indices.ndim == 2

  # Get face_vtx
  if PT.Zone.Type(zone) == "Unstructured":
    # Careful : if zone is poly2d, the ngon element may be absent
    if PT.pred.IS_POLY2D_ZONE(zone) and not PT.Zone.has_ngon_elements(zone):
      maia.algo.edge_pe_to_ngon(zone, comm)
    if PT.Zone.has_ngon_elements(zone):
      ngon_node = PT.Zone.NGonNode(zone)
      face_vtx = MT.Element.connectivity(ngon_node)
      if face_indices is not None:
        face_distri = MT.distribution_value(ngon_node, 'Element')
        face_vtx = EP.block_to_part(face_vtx, face_distri, face_indices-PT.Element.Range(ngon_node)[0], comm)
    else: # Zone has std elements
      global_distri = (zone_dim == 2)
      _face_indices = face_indices[0] if face_indices is not None else None
      face_vtx = CU.entity_vtx_connectivity_elt(zone, comm, 2, global_distri, _face_indices)
  elif PT.Zone.Type(zone) == 'Structured':
    if zone_dim == 3:
      ngon_node = S2U.zonedims_to_ngon(PT.Zone.VertexSize(zone), comm)
      face_vtx = MT.Element.connectivity(ngon_node)
      if face_indices is not None:
        assert face_indices_loc in ['IFaceCenter', 'JFaceCenter', 'KFaceCenter'], \
          "Indices location must be specified when filtering faces on 3D structured meshes"
        _face_indices = s_numbering.ijk_to_index_from_loc(*face_indices, face_indices_loc, PT.Zone.VertexSize(zone))
        face_distri = MT.distribution_value(ngon_node, 'Element')
        face_vtx = EP.block_to_part(face_vtx, face_distri, _face_indices-1, comm)
    elif zone_dim == 2:
      face_vtx = CU.cell_vtx_connectivity_S(zone, zone_dim, face_indices)

  local_coords = get_local_coordinates(zone, face_vtx.values, comm)
  
  face_normal = cpart_algo.compute_face_normal_u(face_vtx.displs.astype(np.int32, copy=False), *local_coords)

  if unitary:
    np_utils.normalize_interweaved_inplace(face_normal, 3)

  return face_normal

def compute_edge_normal(zone, comm, unitary=False, edge_indices=None, edge_indices_loc=None):
  """
  Compute the face normal of a distributed zone, for phydim = 2
  """
  zone_dim = PT.Zone.CellDimension(zone)
  phy_dim  = PT.Zone.PhysicalDimension(zone)
  assert zone_dim in [1,2], "CellDimension of zone must be <= 2 to compute edge normals"
  assert phy_dim  == 2, "PhysicalDimension of zone must be 2 to compute edge normals"

  if edge_indices is not None:
    assert isinstance(edge_indices, np.ndarray) and edge_indices.ndim == 2


  # Get face_vtx
  if PT.Zone.Type(zone) == "Unstructured":
    if PT.pred.IS_POLY2D_ZONE(zone):
      edge_node = MT.Zone.EdgeNode(zone)
      edge_vtx = MT.Element.connectivity(edge_node)
      if edge_indices is not None:
        edge_distri = MT.distribution_value(edge_node, 'Element')
        edge_vtx = EP.block_to_part(edge_vtx, edge_distri, edge_indices-PT.Element.Range(edge_node)[0], comm)
    else: # Zone has std elements
      global_distri = (zone_dim == 1)
      _edge_indices = edge_indices[0] if edge_indices is not None else None
      edge_vtx = CU.entity_vtx_connectivity_elt(zone, comm, 1, global_distri, _edge_indices)
  elif PT.Zone.Type(zone) == 'Structured':
    if zone_dim == 2:
      edge_node = S2U.zonedims_to_ngon(PT.Zone.VertexSize(zone), comm)
      edge_vtx = MT.Element.connectivity(edge_node)
      if edge_indices is not None:
        assert edge_indices_loc in ['IEdgeCenter', 'JEdgeCenter'], \
          "Indices location must be specified when filtering edges on 2D structured meshes"
        _edge_indices = s_numbering.ij_to_index_from_loc(*edge_indices, edge_indices_loc, PT.Zone.VertexSize(zone))
        edge_distri = MT.distribution_value(edge_node, 'Element')
        edge_vtx = EP.block_to_part(edge_vtx, edge_distri, _edge_indices-1, comm)
    if zone_dim == 1:
      edge_vtx = CU.cell_vtx_connectivity_S(zone, zone_dim, edge_indices)

  local_coords = get_local_coordinates(zone, edge_vtx.values, comm)

  edge_normal = np.empty(2*len(edge_vtx))
  edge_normal[0::2] = local_coords[1][1::2] - local_coords[1][0::2] # nx =   yb - ya
  edge_normal[1::2] = local_coords[0][0::2] - local_coords[0][1::2] # ny = -(xb - xa)

  if unitary:
    np_utils.normalize_interweaved_inplace(edge_normal, 2)

  return edge_normal

def _compute_elements_normal(zone, comm, unitary=False, element_indices=None, element_loc=None):
  """
  Distributed implementation of _compute_elements_normal, which compute normal vectors
  and return a raw vector (phydim component per entity)

  Output vector are normalized if unitary is True, otherwise their norm is
  equal to the face area or edge lenght.

  If element_indices is None, normal is computed for all elements
  of relevant dimension of the grid (distributed).
  Otherwise, a PointList-like array is expected: normal will be computed
  only for the specified indices. Indices must be provided in absolute 'cgns numbering',
  (ie. refering to ElementRange_t ids, independantly of element dimension).
  In addition, element_loc is mandatory when filtering faces (resp edges) on 
  3D/S (resp. 2D/S) meshes, to specify if faces (resp. edges) are in I,J, or K
  direction (using IFaceCenter, JFaceCenter, ... JEdgeCenter value).
  """
  cell_dim = PT.Zone.CellDimension(zone)
  phy_dim = PT.Zone.PhysicalDimension(zone)
  if phy_dim == 3 and cell_dim >= 2:
    return compute_face_normal(zone, comm, unitary, element_indices, element_loc)
  elif phy_dim == 2 and cell_dim <= 2:
    return compute_edge_normal(zone, comm, unitary, element_indices, element_loc)


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

