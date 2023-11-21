import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.algo     import indexing
from maia.utils    import np_utils, par_utils, as_pdm_gnum
from maia.transfer import protocols as EP
from .ngon_tools   import PDM_dfacecell_to_dcellface

import cmaia.part_algo as cpart_algo

import Pypdm.Pypdm as PDM

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

def _mean_coords_from_connectivity(vtx_id_idx, cx_expd, cy_expd, cz_expd):
  """ Coordinates should be repeted to match the size of vtx_id_idx """

  vtx_id_n = np.diff(vtx_id_idx)

  mean_x = np.add.reduceat(cx_expd, vtx_id_idx[:-1]) / vtx_id_n
  mean_y = np.add.reduceat(cy_expd, vtx_id_idx[:-1]) / vtx_id_n
  mean_z = np.add.reduceat(cz_expd, vtx_id_idx[:-1]) / vtx_id_n

  return np_utils.interweave_arrays([mean_x, mean_y, mean_z])

@PT.check_is_label("Zone_t")
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
  cx, cy, cz  = PT.Zone.coordinates(zone)
  dist_coords = {'CoordinateX' : cx, 'CoordinateY': cy, 'CoordinateZ': cz}
  vtx_distri = MT.getDistribution(zone, 'Vertex')[1]

  if PT.Zone.Type(zone) == "Unstructured":
    if PT.Zone.has_ngon_elements(zone):
      ngon_node = PT.Zone.NGonNode(zone)
      face_vtx_idx = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
      _face_vtx_idx = np.empty(face_vtx_idx.size, np.int32)
      np.subtract(face_vtx_idx, face_vtx_idx[0], out=_face_vtx_idx)
      face_vtx     = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
      part_data = EP.block_to_part(dist_coords, vtx_distri, [face_vtx], comm)
      coords = [part_data[f'Coordinate{key}'][0] for key in ['X', 'Y', 'Z']]

      return cpart_algo.compute_face_normal_u(_face_vtx_idx, *coords)
  raise NotImplementedError("Only NGON zones are managed")

@PT.check_is_label("Zone_t")
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
  cx, cy, cz  = PT.Zone.coordinates(zone)
  dist_coords = {'CoordinateX' : cx, 'CoordinateY': cy, 'CoordinateZ': cz}
  vtx_distri = MT.getDistribution(zone, 'Vertex')[1]

  if PT.Zone.Type(zone) == "Unstructured":
    if PT.Zone.has_ngon_elements(zone):
      ngon_node = PT.Zone.NGonNode(zone)
      face_vtx_idx = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
      _face_vtx_idx = np.empty(face_vtx_idx.size, np.int32)
      np.subtract(face_vtx_idx, face_vtx_idx[0], out=_face_vtx_idx)
      face_vtx     = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
      part_data = EP.block_to_part(dist_coords, vtx_distri, [face_vtx], comm)
      coords = [part_data[f'Coordinate{key}'][0] for key in ['X', 'Y', 'Z']]

      return _mean_coords_from_connectivity(_face_vtx_idx, *coords)
  raise NotImplementedError("Only NGON zones are managed")

@PT.check_is_label("Zone_t")
def compute_cell_center(zone, comm):

  if PT.Zone.Type(zone) == "Structured":
    raise NotImplementedError("Only NGON zones are managed")

  cell_vtx_idx, cell_vtx = _cell_vtx_connectivity(zone, comm)

  cx, cy, cz  = PT.Zone.coordinates(zone)
  dist_coords = {'CoordinateX' : cx, 'CoordinateY': cy, 'CoordinateZ': cz}
  vtx_distri = MT.getDistribution(zone, 'Vertex')[1]

  part_data = EP.block_to_part(dist_coords, vtx_distri, [cell_vtx], comm)
  coords = [part_data[f'Coordinate{key}'][0] for key in ['X', 'Y', 'Z']]

  return _mean_coords_from_connectivity(cell_vtx_idx, *coords)