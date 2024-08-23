import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from   maia.utils     import np_utils
from   maia.utils     import logging as mlog

from   maia.algo.part import connectivity_utils as CU

from maia.algo.geometry_utils import DIM_TO_LOC, update_container

import cmaia.part_algo as cpart_algo

def compute_edge_measure(zone):
  coords = PT.Zone.coordinates(zone)
  assert isinstance(coords, PT.CartesianCoordinates), "Only cartesian coordinates are supported"

  if PT.Zone.Type(zone) == "Unstructured":
    if PT.Zone.has_ngon_elements(zone) and PT.Zone.CellDimension(zone) == 3:
      raise NotImplementedError("Only U-elts zones are managed")
    edge_vtx_idx, edge_vtx = CU.cell_vtx_connectivity(zone, dim=1)

    # Compute lenght : |L| = ||x2 - x1||
    first_vtx  = edge_vtx[0::2] - 1
    second_vtx = edge_vtx[1::2] - 1
    lenght = (coords[0][second_vtx] - coords[0][first_vtx])**2
    if (cy := coords[1]) is not None:
      lenght += (cy[second_vtx] - cy[first_vtx])**2
    if (cz := coords[2]) is not None:
      lenght += (cz[second_vtx] - cz[first_vtx])**2
    return np.sqrt(lenght)
  else:
    raise NotImplementedError("Only U-elts zones are managed")

def compute_face_measure(zone):

  coords = PT.Zone.coordinates(zone)
  assert isinstance(coords, PT.CartesianCoordinates), "Only cartesian coordinates are supported"
  zone_dim = PT.Zone.CellDimension(zone)
  assert zone_dim >= 2, "CellDimension of zone must be >= 2 to compute face centers"


  if PT.Zone.Type(zone) == "Unstructured":
    _coords = []
    for c in coords:
      _coords.append(c if c is not None else np.zeros_like(coords[0]))
    _coords = np_utils.interweave_arrays(_coords)

    from maia.algo.part.geometry import _compute_zone_centers
    face_center = _compute_zone_centers(zone, 2)

    _coords.shape = (-1, 3)
    face_center.shape = (-1, 3)

    if PT.Zone.has_ngon_elements(zone):
      ngon_node = PT.Zone.NGonNode(zone)
      face_vtx_idx = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
      face_vtx     = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
    else:
      face_vtx_idx, face_vtx = CU.cell_vtx_connectivity(zone, dim=2)

    face_vtx_next = np_utils.roll_once_by_stride(face_vtx_idx, face_vtx)
    # |K| = ½ || sum_i CV_i ⨯ CV_{i+1}|| (C := face center)
    reps = np_utils.repeated_arange(np.diff(face_vtx_idx)) # To access face center
    face_center_reps = face_center[reps]
    crossprod = np.cross(_coords[face_vtx-1] - face_center_reps, _coords[face_vtx_next-1] - face_center_reps)
    # Sum per face
    normalflux = 0.5*np.add.reduceat(crossprod, face_vtx_idx[:-1])
    measure = np.linalg.norm(normalflux, axis=1)
  else:
    vtx_size = [1,1,1]
    vtx_size[:zone_dim] = PT.Zone.VertexSize(zone)
    # Create cz if zone_dim == 2 & cz is None
    _cx = np.atleast_3d(coords[0]) # Auto expand arrays if zone_dim == 2
    _cy = np.atleast_3d(coords[1])
    if zone_dim == 2 and coords[2] is None:
      _cz = np.zeros(vtx_size, dtype=float, order='F')
    else:
      _cz = np.atleast_3d(coords[2])
    measure = cpart_algo.compute_area_face_s(*vtx_size, _cx, _cy, _cz)

  return measure

def _compute_elt_volume(elt_node, coords, out):
  assert out.size == PT.Element.Size(elt_node)
  elt_kind = PT.Element.CGNSName(elt_node)

  ec = PT.get_child_from_name(elt_node, 'ElementConnectivity')[1]

  if elt_kind == 'TETRA_4':
    vtxa = ec[0::4] - 1
    vtxb = ec[1::4] - 1
    vtxc = ec[2::4] - 1
    vtxd = ec[3::4] - 1
    a = [coords[i][vtxa] - coords[i][vtxd] for i in range(3)]
    b = [coords[i][vtxb] - coords[i][vtxd] for i in range(3)]
    c = [coords[i][vtxc] - coords[i][vtxd] for i in range(3)]
    out[:] = np.fabs(a[0]*b[1]*c[2] + b[0]*c[1]*a[2] + c[0]*a[1]*b[2] 
                   - c[0]*b[1]*a[2] - b[0]*a[1]*c[2] - a[0]*c[1]*b[2]) / 6.

  elif elt_kind == 'PYRA_5': # 1/3 * B *h
    _coords = np.stack(coords, axis=1)
    vtxa = ec[0::5] - 1
    vtxb = ec[1::5] - 1
    vtxc = ec[2::5] - 1
    vtxd = ec[3::5] - 1
    vtxe = ec[4::5] - 1
    normal = np.cross(_coords[vtxc] - _coords[vtxa], _coords[vtxd] - _coords[vtxb])
    tt = _coords[vtxa] - _coords[vtxe]
    dist = np.abs(tt[:,0]*normal[:,0] + tt[:,1]*normal[:,1] + tt[:,2]*normal[:,2])
    out[:] = dist / 6.

  elif elt_kind == 'PENTA_6': # B *h
    # Need : face_vtx_idx, face_vtx, cell_face_idx, cell_face
    _coords = np.stack(coords, axis=1)
    vtxa = ec[0::6] - 1
    vtxb = ec[1::6] - 1
    vtxc = ec[2::6] - 1
    vtxd = ec[3::6] - 1
    normal = np.cross(_coords[vtxc] - _coords[vtxa], _coords[vtxb] - _coords[vtxa])
    height = np.linalg.norm(_coords[vtxd] - _coords[vtxa], axis=1)
    out[:] = 0.5*np.linalg.norm(normal, axis=1)*height

  elif elt_kind == 'HEXA_8':
    vtxa = ec[0::8] - 1
    vtxb = ec[2::8] - 1
    vtxc = ec[7::8] - 1
    vtxd = ec[3::8] - 1
    a = [coords[i][vtxa] - coords[i][vtxd] for i in range(3)]
    b = [coords[i][vtxb] - coords[i][vtxd] for i in range(3)]
    c = [coords[i][vtxc] - coords[i][vtxd] for i in range(3)]

    out[:] = np.fabs(a[0]*b[1]*c[2] + b[0]*c[1]*a[2] + c[0]*a[1]*b[2] 
                   - c[0]*b[1]*a[2] - b[0]*a[1]*c[2] - a[0]*c[1]*b[2])

    
  
def compute_cell_measure(zone):
  coords = PT.Zone.coordinates(zone)
  assert isinstance(coords, PT.CartesianCoordinates), "Only cartesian coordinates are supported"
  assert PT.Zone.CellDimension(zone) == 3, "CellDimension of zone must be == 3 to compute cell centers"

  if PT.Zone.Type(zone) == "Unstructured":
    if PT.Zone.has_ngon_elements(zone):

      _coords = np.stack(coords, axis=1)
      ngon_node = PT.Zone.NGonNode(zone)
      face_vtx     = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
      face_vtx_idx = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
      face_vtx_n   = np.diff(face_vtx_idx)

      nface_node = PT.Zone.NFaceNode(zone)
      cell_face_idx = PT.get_child_from_name(nface_node, 'ElementStartOffset')[1]
      cell_face     = PT.get_child_from_name(nface_node, 'ElementConnectivity')[1]

      center = np.add.reduceat(_coords[face_vtx-1], face_vtx_idx[:-1]) / face_vtx_n.reshape((-1,1))

      # Compute mean normal flux on each face : ½ || sum_i CV_i ⨯ CV_{i+1}|| (C := face center)
      face_vtx_next = np_utils.roll_once_by_stride(face_vtx_idx, face_vtx)
      reps = np_utils.repeated_arange(face_vtx_n) # To access face center
      face_center_reps = center[reps]
      crossprod = np.cross(_coords[face_vtx-1] - face_center_reps, _coords[face_vtx_next-1] - face_center_reps)
      normalflux = np.add.reduceat(crossprod, face_vtx_idx[:-1])

      face_contrib = np.sum(center*normalflux, axis=1) # Scalar product face_center * normal_flux

      # Final assembly : for each cell, sum the quantities computed on each cell
      measure = (1/6.) * np.add.reduceat(np.sign(cell_face) * face_contrib[np.abs(cell_face)-1], cell_face_idx[:-1])

    else:
      measure = np.empty(PT.Zone.n_cell(zone))
      start = 0
      for elt in PT.Zone.get_ordered_elements_per_dim(zone)[3]:
        end = start + PT.Element.Size(elt)
        _compute_elt_volume(elt, coords, measure[start:end])
        start = end
  else:
    measure = cpart_algo.compute_volume_cell_s(*PT.Zone.CellSize(zone), *coords)

  return measure



def _compute_zone_measures(zone, dim):
  """Dispatch measures computing according to zone dimension and 
  requested dimension.
  Return a raw array or None"""
  zone_dim = PT.Zone.CellDimension(zone)
  if dim == 'CellCenter':
    dim = zone_dim
  if dim == 3 and zone_dim >= 3:
    return compute_cell_measure(zone)
  elif dim == 2 and zone_dim >= 2:
    return compute_face_measure(zone)
  elif dim == 1 and zone_dim >= 1:
    return compute_edge_measure(zone)

def compute_zone_measures(zone, dim):
  """ Implementation of maia.algo.compute_measures for a given partitioned zone.
  See the calling function for full documentation """

  cell_dim = PT.Zone.CellDimension(zone)
  rq_dim = cell_dim if dim == 'CellCenter' else dim
  measure = _compute_zone_measures(zone, rq_dim)
  if measure is None:
    msg = f"Zone '{PT.get_name(zone)}' skipped during measures computing because "\
          f"its dimension is too low (cell_dim={cell_dim} < {rq_dim})"
    mlog.warning(msg)

  elif measure.size > 0:
    output_loc = DIM_TO_LOC[cell_dim][rq_dim]

    if PT.Zone.Type(zone) == 'Structured':
      # Reshape is needed for S / part zones
      if output_loc == 'FaceCenter':
        # Zone is 3D, and we computed FaceCenter --> We have to split it into I/J/KFaceCenter
        facesize = PT.Zone.FaceSize(zone)
        dirfacesizefunc = [PT.Zone.IFaceSize, PT.Zone.JFaceSize, PT.Zone.KFaceSize]
        start = 0
        for i,dir in enumerate(['I', 'J', 'K']):
          end = start + facesize[i]
          newsize = dirfacesizefunc[i](zone)
          dirmeasure = measure[start:end].reshape(newsize, order='F')
          container = update_container(zone, f'Geometry_{rq_dim}d_{dir}', f'{dir}{output_loc}', {'Measure' : dirmeasure})
          start = end

      if output_loc == 'CellCenter':
        measure = measure.reshape(PT.Zone.CellSize(zone), order='F')
        container = update_container(zone, f'Geometry_{rq_dim}d', output_loc, {'Measure' : measure})

    else: # Unstructured
      container = update_container(zone, f'Geometry_{rq_dim}d', output_loc, {'Measure' : measure})
      if output_loc in ['EdgeCenter', 'FaceCenter']: # PointList is supposed to be mandatory. Maybe we could make it optional in maia ?
        if PT.Zone.has_ngon_elements(zone):
          if output_loc == 'FaceCenter':
            ng = PT.Zone.NGonNode(zone)
          elif output_loc == 'EdgeCenter':
            assert PT.Zone.CellDimension(zone) == 2
            ng = MT.Zone.EdgeNode(zone)
          er = PT.Element.Range(ng)
          pl = np.arange(er[0], er[1]+1, dtype=np.int32).reshape((1,-1), order='F')
          gnum =  PT.maia.getGlobalNumbering(ng, 'Element')[1]
        else: # Must collect faces or edge in same order than the one used to compute face area
          subdim = 2 if output_loc == 'FaceCenter' else 1
          ordered_faces = PT.Zone.get_ordered_elements_per_dim(zone)[subdim]
          sizes =  [PT.Element.Size(e) for e in ordered_faces]
          pl = np.empty((1, sum(sizes)), order='F', dtype=np.int32)
          start = 0
          for i,e in enumerate(ordered_faces):
            er = PT.Element.Range(e)
            pl[0,start:start+sizes[i]] = np.arange(er[0], er[1]+1, dtype=np.int32)
            start += sizes[i]
          # For gnum, we computed on all face or edge so Element/GlobalNumbering/Sections should be fine
          _, gnum = np_utils.concatenate_np_arrays([PT.maia.getGlobalNumbering(e, 'Sections')[1] for e in ordered_faces])

        PT.new_IndexArray('PointList', pl, container)
        PT.maia.newGlobalNumbering({'Index' : gnum}, container)


