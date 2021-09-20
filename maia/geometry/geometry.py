from cmaia.geometry.geometry import *

import numpy as np
import Converter.Internal       as I
import maia.sids.Internal_ext   as IE
import maia.sids.sids           as sids


def _compute_cell_center_s(cell_size, cx,cy,cz):
  """
  What would be function in numpy 
  """
  from maia.utils import py_utils

  ncx, ncy, ncz = cell_size

  cell_center_x =  cx[0:ncx, 0:ncy, 0:ncz] + cx[1:ncx+1, 0:ncy, 0:ncz] \
                 + cx[0:ncx, 1:ncy+1, 0:ncz] + cx[1:ncx+1, 1:ncy+1, 0:ncz] \
                 + cx[0:ncx, 0:ncy, 1:ncz+1] + cx[1:ncx+1, 0:ncy, 1:ncz+1] \
                 + cx[0:ncx, 1:ncy+1, 1:ncz+1] + cx[1:ncx+1, 1:ncy+1, 1:ncz+1]
  cell_center_x = 0.125*cell_center_x

  cell_center_y =  cy[0:ncx, 0:ncy, 0:ncz] + cy[1:ncx+1, 0:ncy, 0:ncz] \
                 + cy[0:ncx, 1:ncy+1, 0:ncz] + cy[1:ncx+1, 1:ncy+1, 0:ncz] \
                 + cy[0:ncx, 0:ncy, 1:ncz+1] + cy[1:ncx+1, 0:ncy, 1:ncz+1] \
                 + cy[0:ncx, 1:ncy+1, 1:ncz+1] + cy[1:ncx+1, 1:ncy+1, 1:ncz+1]
  cell_center_y = 0.125*cell_center_y

  cell_center_z =  cz[0:ncx, 0:ncy, 0:ncz] + cz[1:ncx+1, 0:ncy, 0:ncz] \
                 + cz[0:ncx, 1:ncy+1, 0:ncz] + cz[1:ncx+1, 1:ncy+1, 0:ncz] \
                 + cz[0:ncx, 0:ncy, 1:ncz+1] + cz[1:ncx+1, 0:ncy, 1:ncz+1] \
                 + cz[0:ncx, 1:ncy+1, 1:ncz+1] + cz[1:ncx+1, 1:ncy+1, 1:ncz+1]
  cell_center_z = 0.125*cell_center_z

  #Equivalent to : 
  # center_cell = np.empty(3*ncx*ncy*ncz)
  # idx = 0
  # for k in range(ncz):
    # for j in range(ncy):
      # for i in range(ncx):
        # center_cell[3*idx] = cell_center_x[i,j,k]
        # center_cell[3*idx+1] = cell_center_y[i,j,k]
        # center_cell[3*idx+2] = cell_center_z[i,j,k]
        # idx += 1
  center_cell_np = py_utils.interweave_arrays([cell_center_x.flatten('F'), 
                                               cell_center_y.flatten('F'),
                                               cell_center_z.flatten('F')])
  return center_cell_np
  

@IE.check_is_label("Zone_t")
def compute_cell_center(zone):
  """
  Compute the cell centers of a NGon unstructured zone and return it as a flat np array
  """
  cx, cy, cz = sids.coordinates(zone)

  if sids.Zone.Type(zone) == "Unstructured":
    n_cell     = sids.Zone.n_cell(zone)
    ngons  = [e for e in I.getNodesFromType1(zone, 'Elements_t') if sids.ElementCGNSName(e) == 'NGON_n']
    if len(ngons) != 1:
      raise NotImplementedError(f"Cell center computation is only available for NGON connectivity")
    face_vtx, face_vtx_idx, ngon_pe = sids.ngon_connectivity(zone)
    center_cell = compute_center_cell_u(n_cell, cx, cy, cz, face_vtx, face_vtx_idx, ngon_pe)
  else:
    center_cell = compute_center_cell_s(*sids.Zone.CellSize(zone), cx, cy, cz)

  return center_cell
