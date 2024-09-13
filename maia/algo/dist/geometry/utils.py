import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.utils import py_utils, par_utils

from maia.algo.geometry_utils import DIM_TO_LOC, update_container

def place_in_container(zone, rq_dim, fields, comm):
  cell_dim = PT.Zone.CellDimension(zone)
  output_loc = DIM_TO_LOC[cell_dim][rq_dim]
  if PT.Zone.Type(zone) == 'Structured':
    if output_loc == 'FaceCenter':
      # Zone is 3D, and we computed FaceCenter --> We have to split it into I/J/KFaceCenter
      facesize = PT.Zone.FaceSize(zone)
      dirfacesizefunc = [PT.Zone.IFaceSize, PT.Zone.JFaceSize, PT.Zone.KFaceSize]

      #Distribué -> répartition I,J,K  car distribution des faces calculées sur n_face_tot
      face_distri = par_utils.dn_to_distribution(next(iter(fields.values())).size, comm)
      nfi, nfj, nfk = facesize
      dfacesize = [py_utils.overlap_size(face_distri[0], face_distri[1], 0      , nfi),
                    py_utils.overlap_size(face_distri[0], face_distri[1], nfi    , nfi+nfj),
                    py_utils.overlap_size(face_distri[0], face_distri[1], nfi+nfj, nfi+nfj+nfk)]
      start = 0
      for i,dir in enumerate(['I', 'J', 'K']):
        end = start + dfacesize[i]
        dircenter = {key: val[start:end] for key,val in fields.items()}
        container = update_container(zone, f'Geometry_{rq_dim}d_{dir}', f'{dir}{output_loc}', dircenter)
        distri = par_utils.dn_to_distribution(dfacesize[i], comm)
        pr = np.ones((3,2), order='F', dtype=zone[1].dtype)
        pr[:,1] = dirfacesizefunc[i](zone)
        existing_pr = PT.get_child_from_name(container, 'PointRange')
        if existing_pr:
          cur_pr = existing_pr[1]
          cur_distri = MT.getDistribution(container, 'Index')[1]
          if not (np.array_equal(cur_pr, pr) and np.array_equal(cur_distri, distri)):
            raise RuntimeError("Container already exists, but has incompatible PointRange or Distribution")
        else:
          PT.new_IndexRange(value=pr, parent=container)
          MT.newDistribution({'Index' : distri}, container)
        start = end

    if output_loc == 'CellCenter':
      container = update_container(zone, f'Geometry_{rq_dim}d', output_loc, fields)

  else: # Unstructured
    container = update_container(zone, f'Geometry_{rq_dim}d', output_loc, fields)
    if output_loc in ['EdgeCenter', 'FaceCenter']: # PointList is supposed to be mandatory. Maybe we could make it optional in maia ?
      if PT.Zone.has_ngon_elements(zone):
        if output_loc == 'FaceCenter':
          ng = PT.Zone.NGonNode(zone)
        elif output_loc == 'EdgeCenter':
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

      existing_pl = PT.get_child_from_name(container, 'PointList')
      if existing_pl is not None:
        cur_pl   = existing_pl[1]
        cur_distri = PT.maia.getDistribution(container, 'Index')[1]
        if not (np.array_equal(cur_pl, pl) and np.array_equal(cur_distri, distri)):
          raise RuntimeError("Container already exists, but has incompatible PointList or Distribution")
      else:
        PT.new_IndexArray('PointList', pl, container)
        PT.maia.newDistribution({'Index' : distri}, container)
