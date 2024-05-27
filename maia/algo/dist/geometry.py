import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.algo     import indexing
from maia.utils    import np_utils, par_utils, s_numbering, as_pdm_gnum
from maia.transfer import protocols as EP
from .ngon_tools   import PDM_dfacecell_to_dcellface

import cmaia.part_algo as cpart_algo

import Pypdm.Pypdm as PDM

def _to_xyz(r, theta, z):
  return r*np.cos(theta), r*np.sin(theta), z
def _to_rthetaz(x, y, z):
  return np.sqrt(x**2+y**2), np.arctan2(y, x), z

def _cell_vtx_connectivity_S(zone_S, dim):
  # NB this is not factorised with part.connectivity_utils because arrays layout seems different
  # Maybe we could merge it 
  vertex_size = PT.Zone.VertexSize(zone_S)
  cell_distri = MT.getDistribution(zone_S, 'Cell')[1]

  cell_idx = np.arange(cell_distri[0]+1, cell_distri[1]+1, dtype=zone_S[1].dtype) # Distributed view of cells, as idx  
  dn_cell  = cell_idx.size

  cell_i, cell_j, cell_k = s_numbering.index_to_ijk(cell_idx, PT.Zone.CellSize(zone_S))

  if dim == 2:
    cell_vtx = np.zeros(4*dn_cell, zone_S[1].dtype)
    cell_vtx_idx = 4*np.arange(0, dn_cell+1, dtype=np.int32)
    cell_vtx[0::4] = s_numbering.ijk_to_index(cell_i,   cell_j,   1, vertex_size).flatten()
    cell_vtx[1::4] = s_numbering.ijk_to_index(cell_i+1, cell_j,   1, vertex_size).flatten()
    cell_vtx[2::4] = s_numbering.ijk_to_index(cell_i+1, cell_j+1, 1, vertex_size).flatten()
    cell_vtx[3::4] = s_numbering.ijk_to_index(cell_i,   cell_j+1, 1, vertex_size).flatten()
  elif dim == 3:
    cell_vtx = np.zeros(8*dn_cell, zone_S[1].dtype)
    cell_vtx_idx = 8*np.arange(0, dn_cell+1, dtype=np.int32)
    cell_vtx[0::8] = s_numbering.ijk_to_index(cell_i,   cell_j,   cell_k,   vertex_size).flatten()
    cell_vtx[1::8] = s_numbering.ijk_to_index(cell_i+1, cell_j,   cell_k,   vertex_size).flatten()
    cell_vtx[2::8] = s_numbering.ijk_to_index(cell_i+1, cell_j+1, cell_k,   vertex_size).flatten()
    cell_vtx[3::8] = s_numbering.ijk_to_index(cell_i,   cell_j+1, cell_k,   vertex_size).flatten()
    cell_vtx[4::8] = s_numbering.ijk_to_index(cell_i,   cell_j,   cell_k+1, vertex_size).flatten()
    cell_vtx[5::8] = s_numbering.ijk_to_index(cell_i+1, cell_j,   cell_k+1, vertex_size).flatten()
    cell_vtx[6::8] = s_numbering.ijk_to_index(cell_i+1, cell_j+1, cell_k+1, vertex_size).flatten()
    cell_vtx[7::8] = s_numbering.ijk_to_index(cell_i,   cell_j+1, cell_k+1, vertex_size).flatten()

  return cell_vtx_idx, cell_vtx

def _cell_vtx_connectivity(zone, comm):
  """
  Return cell_vtx connectivity for an input NGON Zone
  """
  if PT.Zone.has_ngon_elements(zone):
    ngon_node = PT.Zone.NGonNode(zone)
    face_vtx      = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
    face_vtx_idx  = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
    face_distri   = MT.get_distribution(ngon_node, 'Element')[1]
    _face_distri  = par_utils.partial_to_full_distribution(face_distri, comm)
    _face_vtx_idx = np.empty(face_vtx_idx.size, np.int32)
    np.subtract(face_vtx_idx, face_vtx_idx[0], out=_face_vtx_idx)
    if PT.Zone.has_nface_elements(zone):
      nface_node = PT.Zone.NFaceNode(zone)
      cell_face      = PT.get_child_from_name(nface_node, 'ElementConnectivity')[1]
      cell_distri    = MT.get_distribution(nface_node, 'Element')[1]
      _cell_distri   = par_utils.partial_to_full_distribution(cell_distri, comm)
      cell_face_idx  = PT.get_child_from_name(nface_node, 'ElementStartOffset')[1]
      _cell_face_idx = np.empty(cell_face_idx.size, np.int32)
      np.subtract(cell_face_idx, cell_face_idx[0], out=_cell_face_idx)

    else:
      assert PT.Element.Range(ngon_node)[0] == 1
      local_pe = indexing.get_ngon_pe_local(ngon_node).reshape(-1, order='C')
      cell_distri   = MT.get_distribution(zone, 'Cell')[1]
      _cell_distri  = par_utils.partial_to_full_distribution(cell_distri, comm)
      _cell_face_idx, cell_face = PDM_dfacecell_to_dcellface(comm, _face_distri, _cell_distri, local_pe)
      _cell_face_idx = np_utils.safe_int_cast(_cell_face_idx, np.int32)

    cell_vtx_idx, cell_vtx = PDM.dconnectivity_combine(comm, 
                                                      as_pdm_gnum(_cell_distri),
                                                      as_pdm_gnum(_face_distri),
                                                      _cell_face_idx,
                                                      as_pdm_gnum(cell_face),
                                                      _face_vtx_idx,
                                                      as_pdm_gnum(face_vtx),
                                                      False)
  else:
    raise NotImplementedError("Only NGON zones are managed")

  return cell_vtx_idx, cell_vtx

def _reduce_mean(vtx_id_idx, *arrays):
  vtx_id_n = np.diff(vtx_id_idx)
  return [np.add.reduceat(array, vtx_id_idx[:-1]) / vtx_id_n for array in arrays]


def _mean_coords_from_connectivity(vtx_id_idx, cx_expd, cy_expd, cz_expd):
  """ Coordinates should be repeted to match the size of vtx_id_idx """
  coords_mean = _reduce_mean(vtx_id_idx, cx_expd, cy_expd, cz_expd)
  return np_utils.interweave_arrays(coords_mean)

def _mean_coords_from_connectivity_cyl(vtx_id_idx, cr_expd, ctheta_expd, cz_expd):
  """ Coordinates should be repeted to match the size of vtx_id_idx """
  cx,cy,cz = _to_xyz(cr_expd, ctheta_expd, cz_expd)
  coords_mean = _reduce_mean(vtx_id_idx, cx, cy, cz)
  return np_utils.interweave_arrays(_to_rthetaz(*coords_mean))



def compute_face_normal(zone, comm):
  """Compute the face normal of a distributed zone.

  Input zone must have cartesian coordinates recorded under a unique
  GridCoordinates node.

  The normal is outward oriented and its norms equals the area of the faces.

  Args:
    zone (CGNSTree): Distributed 3D or 2D U-NGon CGNS Zone
  Returns:
    face_normal (array): Flat (interlaced) numpy array of face normal

  """
  coords = PT.Zone.coordinates(zone)
  dist_coords = dict((coords._fields[i], coords[i]) for i in range(len(coords)))
  vtx_distri = MT.getDistribution(zone, 'Vertex')[1]

  if PT.Zone.Type(zone) == "Unstructured":
    if PT.Zone.has_ngon_elements(zone):
      ngon_node = PT.Zone.NGonNode(zone)
      face_vtx_idx = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
      _face_vtx_idx = np.empty(face_vtx_idx.size, np.int32)
      np.subtract(face_vtx_idx, face_vtx_idx[0], out=_face_vtx_idx)
      face_vtx     = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
      part_data = EP.block_to_part(dist_coords, vtx_distri, [face_vtx], comm)
      coords = [part_data[key][0] for key in part_data.keys()]

      return cpart_algo.compute_face_normal_u(_face_vtx_idx, *coords)
  raise NotImplementedError("Only NGON zones are managed")

def compute_face_center(zone, comm):
  """Compute the face center of a distributed zone.

  Input zone must have cartesian coordinates recorded under a unique
  GridCoordinates node.

  Centers are computed using a basic average over the vertices of the faces.

  Args:
    zone (CGNSTree): Distributed 3D or 2D U-NGon CGNS Zone
  Returns:
    face_normal (array): Flat (interlaced) numpy array of face centers

  """
  # TODO Implementation for U/elts
  if PT.Zone.Type(zone) == "Structured":
    face_vtx_idx, face_vtx = _cell_vtx_connectivity_S(zone, PT.Zone.CellDimension(zone))
  else:
    if PT.Zone.has_ngon_elements(zone):
      ngon_node = PT.Zone.NGonNode(zone)
      _face_vtx_idx = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
      face_vtx_idx = np.empty(_face_vtx_idx.size, np.int32)
      np.subtract(_face_vtx_idx, _face_vtx_idx[0], out=face_vtx_idx)
      face_vtx     = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
    else:
      raise NotImplementedError("U/elt zones are not managed")

  coords = PT.Zone.coordinates(zone)
  dist_coords = dict((coords._fields[i], coords[i]) for i in range(len(coords)))
  vtx_distri = MT.getDistribution(zone, 'Vertex')[1]

  part_data = EP.block_to_part(dist_coords, vtx_distri, [face_vtx], comm)
  local_coords = [part_data[key][0] for key in part_data.keys()]

  if isinstance(coords, PT.CartesianCoordinates):
    return _mean_coords_from_connectivity(face_vtx_idx, *local_coords)
  elif isinstance(coords, PT.CylindricalCoordinates):
    return _mean_coords_from_connectivity_cyl(face_vtx_idx, *local_coords)

def compute_cell_center(zone, comm):
  # TODO Implementation for U/elts

  if PT.Zone.Type(zone) == "Structured":
    cell_vtx_idx, cell_vtx = _cell_vtx_connectivity_S(zone, PT.Zone.CellDimension(zone))
  else:
    cell_vtx_idx, cell_vtx = _cell_vtx_connectivity(zone, comm)

  coords = PT.Zone.coordinates(zone)
  dist_coords = dict((coords._fields[i], coords[i]) for i in range(len(coords)))
  vtx_distri = MT.getDistribution(zone, 'Vertex')[1]

  part_data = EP.block_to_part(dist_coords, vtx_distri, [cell_vtx], comm)
  local_coords = [part_data[key][0] for key in part_data.keys()]

  if isinstance(coords, PT.CartesianCoordinates):
    return _mean_coords_from_connectivity(cell_vtx_idx, *local_coords)
  elif isinstance(coords, PT.CylindricalCoordinates):
    return _mean_coords_from_connectivity_cyl(cell_vtx_idx, *local_coords)