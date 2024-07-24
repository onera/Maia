import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.algo     import indexing
from maia.utils    import py_utils, np_utils, par_utils, s_numbering, as_pdm_gnum
from maia.transfer import protocols as EP

from .ngon_tools   import PDM_dfacecell_to_dcellface
from .s_to_u       import zonedims_to_ngon

from maia.utils import logging as mlog

from maia.algo.geometry_utils import DIM_TO_LOC, update_container

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

def _entity_vtx_connectivity_elt(zone, comm, dim, distri_global):
  """
  Exchange vtx ids to compute the cell_vtx table for a given dimension.
  All elements of same dim are concatenated in output.
  If distrib_global is True, this cell_vtx connectivity is redistributed to match
  the global distribution of all elements of the requested dim
  Otherwise, we just concatenate the data of each section
  Exemple : if distri TETRA = [0,5,9], distri PRISM = [0,3,7] and global distri CELL = [0,8,16]
  with local mode rank 0 get 5 tetra and 3 prism, rank 1 get 4 tetra and 4 prism
  with global mode rank 0 get 8 tetra and rank 1 get 1 tetra and 7 prism
  """
  all_cell_vtx_n = []
  all_cell_vtx = []

  if distri_global:
    assert PT.Zone.CellDimension(zone) == dim, "Redispatch only supported for native cell dimension"
    distri_cell = MT.getDistribution(zone, 'Cell')[1]
    start = 0

  for elt in PT.Zone.get_ordered_elements_per_dim(zone)[dim]:
    distri = MT.get_distribution(elt, 'Element')[1]
    ec = PT.get_child_from_name(elt, 'ElementConnectivity')[1]
    
    if distri_global:
      end = start + PT.Element.Size(elt)
      distri_out = distri.copy()
      # Here we restrict the total cell distribution to ElementRange (ignoring low order elts), 
      # then we shift it to make it start a 0
      distri_out[0] = max(min(distri_cell[0], end), start) - start
      distri_out[1] = max(min(distri_cell[1], end), start) - start
      btb = EP.BlockToBlock(distri, distri_out, comm)
      ec = btb.exchange(ec, PT.Element.NVtx(elt))
      ec_idx = PT.Element.NVtx(elt) * np.ones(distri_out[1] - distri_out[0], np.int32)
      start = end
    else:
      ec_idx = PT.Element.NVtx(elt) * np.ones(distri[1] - distri[0], np.int32)

    all_cell_vtx.append(ec)
    all_cell_vtx_n.append(ec_idx)

  cell_vtx_n = np.concatenate(all_cell_vtx_n, dtype=np.int32)
  cell_vtx = np.concatenate(all_cell_vtx)
  cell_vtx_idx = np_utils.sizes_to_indices(cell_vtx_n)

  return cell_vtx_idx, cell_vtx


def _cell_vtx_connectivity_ngon(zone, comm):
  """
  Return cell_vtx connectivity for an input NGON Zone
  """
  assert PT.Zone.Type(zone) == "Unstructured" and PT.Zone.CellDimension(zone) == 3
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
      local_pe = indexing.get_pe_local(ngon_node).reshape(-1, order='C')
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

def compute_edge_center(zone, comm):
  """Compute the edge centers of a distributed zone.

  Input zone must have cartesian coordinates or cylindrical coordinates recorded under a unique
  GridCoordinates node.
  Centers are computed using a basic average over the vertices of the edges.
  """
  if PT.Zone.Type(zone) == "Unstructured":
    global_distri = PT.Zone.CellDimension == 1
    edge_vtx_idx, edge_vtx = _entity_vtx_connectivity_elt(zone, comm, 1, global_distri)
  else:
    raise NotImplementedError("Only U zones are managed")

  coords = PT.Zone.coordinates(zone)

  dist_coords = dict((coords._fields[i], coords[i]) for i in range(len(coords)) if coords[i] is not None)
  vtx_distri = MT.getDistribution(zone, 'Vertex')[1]

  part_data = EP.block_to_part(dist_coords, vtx_distri, [edge_vtx], comm)
  local_coords = [part_data[key][0] for key in part_data.keys()]

  while len(local_coords) < 3 : #We are in phydim < 3 case, add Y and/or Z array
    local_coords.append(np.zeros_like(local_coords[0]))

  if isinstance(coords, PT.CartesianCoordinates):
    return _mean_coords_from_connectivity(edge_vtx_idx, *local_coords)
  elif isinstance(coords, PT.CylindricalCoordinates):
    return _mean_coords_from_connectivity_cyl(edge_vtx_idx, *local_coords)

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
  zone_dim = PT.Zone.CellDimension(zone)
  assert zone_dim >= 2, "CellDimension of zone must be >= 2 to compute face centers"

  # TODO Implementation for U/elts
  if PT.Zone.Type(zone) == "Structured":
    vtx_size = np.ones(3, zone[1].dtype) # This trick allows to call zonedims_to_ngon even on 2D meshes
    vtx_size[:zone_dim] = PT.Zone.VertexSize(zone)
    ngon_node = zonedims_to_ngon(vtx_size, comm)
    _face_vtx_idx = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
    face_vtx_idx = np.empty(_face_vtx_idx.size, np.int32)
    np.subtract(_face_vtx_idx, _face_vtx_idx[0], out=face_vtx_idx)
    face_vtx     = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
  else:
    if PT.Zone.has_ngon_elements(zone):
      ngon_node = PT.Zone.NGonNode(zone)
      _face_vtx_idx = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
      face_vtx_idx = np.empty(_face_vtx_idx.size, np.int32)
      np.subtract(_face_vtx_idx, _face_vtx_idx[0], out=face_vtx_idx)
      face_vtx     = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
    else:
      global_distri = PT.Zone.CellDimension(zone) == 2
      face_vtx_idx, face_vtx = _entity_vtx_connectivity_elt(zone, comm, 2, global_distri)


  coords = PT.Zone.coordinates(zone)
  dist_coords = dict((coords._fields[i], coords[i]) for i in range(len(coords)) if coords[i] is not None)
  vtx_distri = MT.getDistribution(zone, 'Vertex')[1]

  part_data = EP.block_to_part(dist_coords, vtx_distri, [face_vtx], comm)
  local_coords = [part_data[key][0] for key in part_data.keys()]

  if len(local_coords) == 2 : #We are in phydim==2, Add Z array
    local_coords.append(np.zeros_like(local_coords[0]))

  if isinstance(coords, PT.CartesianCoordinates):
    return _mean_coords_from_connectivity(face_vtx_idx, *local_coords)
  elif isinstance(coords, PT.CylindricalCoordinates):
    return _mean_coords_from_connectivity_cyl(face_vtx_idx, *local_coords)

def compute_cell_center(zone, comm):
  assert PT.Zone.CellDimension(zone) == 3, "CellDimension of zone must be == 3 to compute cell centers"

  if PT.Zone.Type(zone) == "Structured":
    cell_vtx_idx, cell_vtx = _cell_vtx_connectivity_S(zone, PT.Zone.CellDimension(zone))
  else:
    if PT.Zone.has_ngon_elements(zone):
      cell_vtx_idx, cell_vtx = _cell_vtx_connectivity_ngon(zone, comm)
    else:
      cell_vtx_idx, cell_vtx = _entity_vtx_connectivity_elt(zone, comm, 3, True)


  coords = PT.Zone.coordinates(zone)
  dist_coords = dict((coords._fields[i], coords[i]) for i in range(len(coords)))
  vtx_distri = MT.getDistribution(zone, 'Vertex')[1]

  part_data = EP.block_to_part(dist_coords, vtx_distri, [cell_vtx], comm)
  local_coords = [part_data[key][0] for key in part_data.keys()]

  if isinstance(coords, PT.CartesianCoordinates):
    return _mean_coords_from_connectivity(cell_vtx_idx, *local_coords)
  elif isinstance(coords, PT.CylindricalCoordinates):
    return _mean_coords_from_connectivity_cyl(cell_vtx_idx, *local_coords)


def _compute_zone_centers(zone, dim, comm):
  """Dispatch centers computing according to zone dimension and 
  requested dimension
  Return a raw interlaced array or None"""
  zone_dim = PT.Zone.CellDimension(zone)
  if dim == 'CellCenter':
    dim = zone_dim
  if dim == 3 and zone_dim >= 3:
    return compute_cell_center(zone, comm)
  elif dim == 2 and zone_dim >= 2:
    return compute_face_center(zone, comm)
  elif dim == 1 and zone_dim >= 1:
    return compute_edge_center(zone, comm)

def compute_zone_centers(zone, dim, comm):
  """ Implementation of maia.algo.compute_centers for a given distributed zone.
  See the above function for full documentation """

  cell_dim = PT.Zone.CellDimension(zone)
  rq_dim = cell_dim if dim == 'CellCenter' else dim
  interlaced_centers = _compute_zone_centers(zone, rq_dim, comm)
  if interlaced_centers is None:
    msg = f"Zone '{PT.get_name(zone)}' skipped during centers computing because "\
          f"its dimension is too low (cell_dim={cell_dim} < {rq_dim})"
    mlog.warning(msg)
  else:
    # Underlying function always return a concatenated array of size 3*n_entity
    # We must filter it if phy_dim is lower
    coords = PT.Zone.coordinates(zone)
    center_names = [s.replace('Coordinate', 'Center') for s in coords._fields]
    phy_dim  = len([c for c in coords if c is not None]) # 1, 2 or 3
    centers = {name : interlaced_centers[i::3] \
               for i,name in enumerate(center_names) if i < phy_dim}

    output_loc = DIM_TO_LOC[cell_dim][rq_dim]
    if PT.Zone.Type(zone) == 'Structured':
      if output_loc == 'FaceCenter':
        # Zone is 3D, and we computed FaceCenter --> We have to split it into I/J/KFaceCenter
        facesize = PT.Zone.FaceSize(zone)
        dirfacesizefunc = [PT.Zone.IFaceSize, PT.Zone.JFaceSize, PT.Zone.KFaceSize]

        #Distribué -> répartition I,J,K  car distribution des faces calculées sur n_face_tot
        face_distri = par_utils.dn_to_distribution(next(iter(centers.values())).size, comm)
        nfi, nfj, nfk = facesize
        dfacesize = [py_utils.overlap_size(face_distri[0], face_distri[1], 0      , nfi),
                     py_utils.overlap_size(face_distri[0], face_distri[1], nfi    , nfi+nfj),
                     py_utils.overlap_size(face_distri[0], face_distri[1], nfi+nfj, nfi+nfj+nfk)]
        start = 0
        for i,dir in enumerate(['I', 'J', 'K']):
          end = start + dfacesize[i]
          dircenter = {key: val[start:end] for key,val in centers.items()}
          container = update_container(zone, f'Geometry_{rq_dim}d_{dir}', f'{dir}{output_loc}', dircenter)
          MT.newDistribution({'Index' : par_utils.dn_to_distribution(dfacesize[i], comm)}, container)
          pr = np.ones((3,2), order='F', dtype=zone[1].dtype)
          pr[:,1] = dirfacesizefunc[i](zone)
          PT.new_IndexRange(value=pr, parent=container)
          start = end

      if output_loc == 'CellCenter':
        container = update_container(zone, f'Geometry_{rq_dim}d', output_loc, centers)

    else: # Unstructured
      container = update_container(zone, f'Geometry_{rq_dim}d', output_loc, centers)
      if output_loc in ['EdgeCenter', 'FaceCenter']: # PointList is supposed to be mandatory. Maybe we could make it optional in maia ?
        if PT.Zone.has_ngon_elements(zone):
          if output_loc == 'FaceCenter':
            ng = PT.Zone.NGonNode(zone)
          elif output_loc == 'EdgeCenter':
            assert PT.Zone.CellDimension(zone) == 2
            ng = MT.Zone.EdgeNode(zone)
          er = PT.Element.Range(ng)
          distri = MT.getDistribution(ng, 'Element')[1]
          pl = np.arange(distri[0]+er[0], distri[1]+er[0], dtype=er.dtype).reshape((1,-1), order='F')
        else: # Must collect faces or edge in same order than the one used to compute face centers
          subdim = 2 if output_loc == 'FaceCenter' else 1
          ordered_faces = PT.Zone.get_ordered_elements_per_dim(zone)[subdim]
          distribs = [MT.getDistribution(e, 'Element')[1] for e in ordered_faces]
          sizes =  [distri_elt[1] - distri_elt[0] for distri_elt in distribs]
          pl = np.empty((1, sum(sizes)), dtype=zone[1].dtype, order='F')
          start = 0
          for i,e in enumerate(ordered_faces):
            distri_elt = distribs[i]
            er = PT.Element.Range(e)
            pl[0,start:start+sizes[i]] = np.arange(distri_elt[0]+er[0], distri_elt[1]+er[0], dtype=er.dtype)
            start += sizes[i]
          distri = sum(distribs) # Compute global distrib

        PT.new_IndexArray('PointList', pl, container)
        PT.maia.newDistribution({'Index' : distri}, container)
