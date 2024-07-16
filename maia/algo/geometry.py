import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT
from maia.algo.apply_function_to_nodes import zones_iterator

from .dist import geometry as dist_geometry
from .part import geometry as part_geometry

from maia.utils import logging as mlog
from maia.utils import np_utils, par_utils, py_utils


def _compute_vol_center(zone, comm=None):
  if MT.getDistribution(zone) is not None:
    assert comm is not None
    return dist_geometry.compute_cell_center(zone, comm)
  else:
    return part_geometry.compute_cell_center(zone)

def _compute_face_center(zone, comm=None):
  if MT.getDistribution(zone) is not None:
    assert comm is not None
    return dist_geometry.compute_face_center(zone, comm)
  else:
    return part_geometry.compute_face_center(zone)

def _compute_edge_center(zone, comm=None):
  if MT.getDistribution(zone) is not None:
    assert comm is not None
    return dist_geometry.compute_edge_center(zone, comm)
  else:
    return part_geometry.compute_edge_center(zone)

def _compute_centers(zone, dim, comm=None):
  """Dispatch centers computing according to zone dimension and 
  requested dimension """
  zone_dim = PT.Zone.CellDimension(zone)
  if dim == 'Cell':
    dim = zone_dim
  if dim == 3 and zone_dim >= 3:
    return _compute_vol_center(zone, comm)
  elif dim == 2 and zone_dim >= 2:
    return _compute_face_center(zone, comm)
  elif dim == 1 and zone_dim >= 1:
    return _compute_edge_center(zone, comm)
  



def compute_centers(t, dim, comm=None, out_fs_name='', method='mean'):
  """Compute the cell centers of a partitioned zone.

  Input zone must have cartesian or cylindrical coordinates recorded under a unique
  GridCoordinates node.
  Centers are computed using a basic average over the vertices of the cells.

  Args:
    t    (CGNSTree(s)): Tree (or sequences of) starting at Zone_t level or higher.
    dim  (int): XXXXX
    comm       (MPIComm) : MPI communicator, mandatory only for distributed trees

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #compute_cell_center@start
        :end-before: #compute_cell_center@end
        :dedent: 2
  """

  """       VolCenter     FaceCenter    EdgeCenter   | CellCenter
  dim 3        X              X             X        |     Volu
  dim 2                       X             X        |     Face
  dim 1                                     X        |     Edge
  
  Maillages (celldim / phydim): 3D(3), 2D(3), 2D(2), 1D(3), 1D(2), 1D(1) --> 6 choix
  Connectivity : Ungon / Uelt / S   ---> 3 choix
  Parallel : Dist / part --> 2 choix 
  Cartésien / Cylindrique --> 2 choix
  Total : 6*3*2*2 = 72 possibilité


  """

  # For each cell_dimension, list of output GridLocation depending of requested dim argument
  dim_to_loc = {3: ['Vertex', 'EdgeCenter', 'FaceCenter', 'CellCenter'],
                2: ['Vertex', 'EdgeCenter', 'CellCenter',  None],
                1: ['Vertex', 'CellCenter',  None,         None],}

  def get_or_create_container(zone, container_name, container_loc):
    """ Utility to retrieve a container from its name, or create it """
    container = PT.get_child_from_name(zone, container_name)
    if container is not None: # Container exists
      assert PT.Subset.GridLocation(container) == container_name, \
        f"Container {PT.get_name(container)} already exists in zone {PT.get_name(zone)} but has incompatible GridLocation"
    else:  # Create container
      container = PT.new_child(zone, container_name, 'DiscreteData_t')
      PT.new_GridLocation(container_loc, container)
    return container
  def feed_container(container, datas, names):
    for data, name in zip(datas, names):
      if data is not None:
        PT.update_child(container, name, 'DataArray_t', data)

  
  for zone in zones_iterator(t):
    
    is_distributed = MT.getDistribution(zone) is not None
    cell_dim = PT.Zone.CellDimension(zone)
    rq_dim = cell_dim if dim == 'Cell' else dim
    interlaced_centers = _compute_centers(zone, rq_dim, comm)
    if interlaced_centers is None:
      msg = f"Zone '{PT.get_name(zone)}' skipped in compute_centers because "\
            f"its dimension is too low (cell_dim={cell_dim} < {rq_dim})"
      mlog.warning(msg)
    else:
      # Underlying function always return a concatenated array of size 3*n_entity
      # We must filter it if phy_dim is lower
      coords = PT.Zone.coordinates(zone)
      center_names = [s.replace('Coordinate', 'Center') for s in coords._fields]
      phy_dim  = len([c for c in coords if c is not None]) # 1, 2 or 3
      centerx = interlaced_centers[0::3]
      centery = interlaced_centers[1::3] if phy_dim >= 2 else None
      centerz = interlaced_centers[2::3] if phy_dim >= 3 else None
      centers = [centerx, centery, centerz]

      output_loc = dim_to_loc[cell_dim][rq_dim]
      if PT.Zone.Type(zone) == 'Structured':
        # Reshape is needed for S / part zones
        if output_loc == 'FaceCenter':
          # Zone is 3D, and we computed FaceCenter --> We have to split it into I/J/KFaceCenter
          facesize = PT.Zone.FaceSize(zone)
          dirfacesizefunc = [PT.Zone.IFaceSize, PT.Zone.JFaceSize, PT.Zone.KFaceSize]
          if PT.Zone.coordinates(zone)[0].ndim > 1: # Only reshape if zone is partitionned
            start = 0
            for i,dir in enumerate(['I', 'J', 'K']):
              end = start + facesize[i]
              newsize = dirfacesizefunc[i](zone)
              dircenterx = centerx[start:end].reshape(newsize, order='F')
              dircentery = centery[start:end].reshape(newsize, order='F')
              dircenterz = centerz[start:end].reshape(newsize, order='F')
              container = get_or_create_container(zone, f'Geometry_{rq_dim}d_{dir}', f'{dir}{output_loc}')
              feed_container(container, [dircenterx, dircentery, dircenterz], center_names)
              start = end
          else: #Distribué -> répartition I,J,K  car distribution des faces calculées sur n_face_tot
            face_distri = par_utils.dn_to_distribution(centerx.size, comm)
            nfi, nfj, nfk = facesize
            dfacesize = [py_utils.overlap_size(face_distri[0], face_distri[1], 0      , nfi),
                         py_utils.overlap_size(face_distri[0], face_distri[1], nfi    , nfi+nfj),
                         py_utils.overlap_size(face_distri[0], face_distri[1], nfi+nfj, nfi+nfj+nfk)]
            start = 0
            for i,dir in enumerate(['I', 'J', 'K']):
              end = start + dfacesize[i]
              dircenterx = centerx[start:end]
              dircentery = centery[start:end]
              dircenterz = centerz[start:end]
              container = get_or_create_container(zone, f'Geometry_{rq_dim}d_{dir}', f'{dir}{output_loc}')
              feed_container(container, [dircenterx, dircentery, dircenterz], center_names)
              MT.newDistribution({'Index' : par_utils.dn_to_distribution(dircenterx.size, comm)}, container)
              pr = np.ones((3,2), order='F', dtype=zone[1].dtype)
              pr[:,1] = dirfacesizefunc[i](zone)
              PT.new_IndexRange(value=pr, parent=container)
              start = end


        if output_loc == 'CellCenter':
          if PT.Zone.coordinates(zone)[0].ndim > 1: # Only reshape if zone is partitionned
            for dir in range(len(centers)):
              if centers[dir] is not None:
                centers[dir] = centers[dir].reshape(PT.Zone.CellSize(zone), order='F')

          container = get_or_create_container(zone, f'Geometry_{rq_dim}d', output_loc)
          feed_container(container, centers, center_names)

      else: # Unstructured
        container = get_or_create_container(zone, f'Geometry_{rq_dim}d', output_loc)
        feed_container(container, centers, center_names)
        if output_loc in ['EdgeCenter', 'FaceCenter']: # PointList is supposed to be mandatory. Maybe we could make it optional in maia ?
          if PT.Zone.has_ngon_elements(zone):
            if output_loc == 'FaceCenter':
              ng = PT.Zone.NGonNode(zone)
            elif output_loc == 'EdgeCenter':
              assert PT.Zone.CellDimension(zone) == 2
              ng = MT.Zone.EdgeNode(zone)
            er = PT.Element.Range(ng)
            if is_distributed:
              distri = MT.getDistribution(ng, 'Element')[1]
              pl = np.arange(distri[0]+er[0], distri[1]+er[0], dtype=er.dtype).reshape((1,-1), order='F')
            else:
              pl = np.arange(er[0], er[1]+1, dtype=np.int32).reshape((1,-1), order='F')
              gnum =  PT.maia.getGlobalNumbering(ng, 'Element')[1]
          else: # Must collect faces or edge in same order than the one used to compute face centers
            subdim = 2 if output_loc == 'FaceCenter' else 1
            ordered_faces = PT.Zone.get_ordered_elements_per_dim(zone)[subdim]
            if is_distributed:
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
            else:
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
          if is_distributed:
            PT.maia.newDistribution({'Index' : distri}, container)
          else:
            PT.maia.newGlobalNumbering({'Index' : gnum}, container)
