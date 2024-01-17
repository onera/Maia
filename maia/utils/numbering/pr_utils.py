import numpy as np

import maia.pytree as PT

def normal_index_shift(point_range, n_vtx, bnd_axis, input_loc, output_loc):
  """
  Return the value that should be added to pr[normal_index,:] to account for cell <-> face|vtx transformation :
    +1 if we move from cell to face|vtx and if it was the last plane of cells
    -1 if we move from face|vtx to cell and if it was the last plane of face|vtx
     0 in other cases
  """
  in_loc_is_cell  = (input_loc == 'CellCenter')
  out_loc_is_cell = (output_loc == 'CellCenter')
  normal_index_is_last = point_range[bnd_axis,0] == (n_vtx[bnd_axis] - int(in_loc_is_cell))
  correction_sign = -int(out_loc_is_cell and not in_loc_is_cell) \
                    +int(not out_loc_is_cell and in_loc_is_cell)
  return int(normal_index_is_last) * correction_sign

def transform_bnd_pr_size(point_range, input_loc, output_loc):
  """
  Predict a point_range defined at an input_location if it were defined at an output_location
  """
  size = np.abs(point_range[:,1] - point_range[:,0]) + 1

  if input_loc == 'Vertex' and 'Center' in output_loc:
    size -= (size != 1)
  elif 'Center' in input_loc and output_loc == 'Vertex':
    bnd_axis = PT.Subset.normal_axis(PT.new_BC(point_range=point_range, loc=input_loc))
    mask = np.arange(point_range.shape[0]) == bnd_axis
    size += (~mask)
  return size