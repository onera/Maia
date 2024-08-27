import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.utils import np_utils

from maia.algo.geometry_utils import DIM_TO_LOC, update_container

def place_in_container(zone, rq_dim, fields):
  cell_dim = PT.Zone.CellDimension(zone)
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
        dirfields = {key: val[start:end].reshape(newsize, order='F') \
                      for key, val in fields.items()}
        container = update_container(zone, f'Geometry_{rq_dim}d_{dir}', f'{dir}{output_loc}', dirfields)
        start = end

    if output_loc == 'CellCenter':
      fields = {key: val.reshape(PT.Zone.CellSize(zone), order='F') \
                 for key, val in fields.items()}

      container = update_container(zone, f'Geometry_{rq_dim}d', output_loc, fields)

  else: # Unstructured
    container = update_container(zone, f'Geometry_{rq_dim}d', output_loc, fields)
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
      else: # Must collect faces or edge in same order than the one used to compute face centers
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

      existing_pl = PT.get_child_from_name(container, 'PointList')
      if existing_pl is not None:
        cur_pl   = existing_pl[1]
        cur_gnum = PT.maia.getGlobalNumbering(container, 'Index')[1]
        if not (np.array_equal(cur_pl, pl) and np.array_equal(cur_gnum, gnum)):
          raise RuntimeError("Container already exists, but has incompatible PointList or GlobalNumbering")
      else:
        PT.new_IndexArray('PointList', pl, container)
        PT.maia.newGlobalNumbering({'Index' : gnum}, container)

