import numpy as np

from maia.typing import *
import maia.pytree as PT

from .s_numbering_funcs import ijk_to_index_from_loc, ij_to_index_from_loc

def normal_index_shift(point_range: NDArray,
                       n_vtx: List[int],
                       bnd_axis: int,
                       input_loc: str, 
                       output_loc: str) -> int:
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

def transform_bnd_pr_size(point_range: NDArray,
                          input_loc: str,
                          output_loc: str) -> NDArray:
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

def unroll_pr(pr: NDArray) -> NDArray:
  """
  Create a structured pointList of size (idx_dim,N) spawning the same region than the input PR.
  Unrolling if done following cgns conventions : increasing i, then j, then k
  """
  inc = 2*(pr[:,0] <= pr[:,1]) - 1 #In each direction, 1 if pr[l,0] <= pr[l,1] else - 1

  # Here we build for each direction a looping array range(start, end+1) if pr is increasing
  # or range(start, end-1, -1) if pr is decreasing
  np_idx_arrays = []
  for l in range(pr.shape[0]):
    np_idx_arrays.append(np.arange(pr[l,0], pr[l,1] + inc[l], inc[l]))

  def cartesian_product(*arrays):
    #https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points/
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

  out_tmp = cartesian_product(*reversed(np_idx_arrays)).T
  out = np.empty_like(out_tmp)

  assert pr.shape[0] >= 2
  out[0,:] = out_tmp[-1,:]
  out[-1,:] = out_tmp[0,:]
  if pr.shape[0] == 3:
    out[1,:] = out_tmp[1,:]

  return out

def _ijk_to_func(idx_arrays:Sequence[NDArray], loc:str, n_vtx_S:List[int], order:str) -> NDArray:
  """
  Wraps the relevant ijk_to_func depening of dimension and location, and call it in a vectorial way
  """
  dim = len(n_vtx_S)
  assert len(idx_arrays) == len(n_vtx_S)
  assert order in ['F', 'C']

  if dim == 3:
    i_idx, j_idx, k_idx = idx_arrays
    if order == 'F':
      return ijk_to_index_from_loc(i_idx, j_idx.reshape(-1,1), k_idx.reshape(-1,1,1), loc, n_vtx_S).flatten()
    else: # Order = 'C'
      return ijk_to_index_from_loc(i_idx.reshape(-1,1,1), j_idx.reshape(-1,1), k_idx, loc, n_vtx_S).flatten()
  elif dim == 2:
    i_idx, j_idx = idx_arrays
    if order == 'F':
      return ij_to_index_from_loc(i_idx, j_idx.reshape(-1,1), loc, n_vtx_S).flatten()
    else: # Order = 'C'
      return ij_to_index_from_loc(i_idx.reshape(-1,1), j_idx, loc, n_vtx_S).flatten()
  elif dim == 1:
    i_idx = idx_arrays[0]
    return i_idx
  else:
    raise AssertionError(f"Invalid 'n_vtx_S' argument ({n_vtx_S})")


def compute_pointList_from_pointRanges(sub_pr_list: List[NDArray],
                                       n_vtx_S: List[int],
                                       loc: str,
                                       order: str = 'F') -> NDArray:
  """
  Transform a list of pointRange in a concatenated pointList array in order. The sub_pr_list must
  describe entity of kind loc, which can take the values '{I,J,K}FaceCenter', 'Vertex' or 'CellCenter'.
  The pointlist array will be output at the same location.
  Note that the pointRange intervals can be reverted (start > end) as it occurs in GC nodes.
  """


  sub_range_sizes = [(np.abs(pr[:,1] - pr[:,0]) + 1).prod() for pr in sub_pr_list]
  dtype = sub_pr_list[0].dtype if len(sub_pr_list) > 0 else int
  point_list = np.empty((1, sum(sub_range_sizes)), order='F', dtype=dtype)
  counter = 0

  for ipr, pr in enumerate(sub_pr_list):
    inc = 2*(pr[:,0] <= pr[:,1]) - 1 #In each direction, 1 if pr[l,0] <= pr[l,1] else - 1

    # Here we build for each direction a looping array range(start, end+1) if pr is increasing
    # or range(start, end-1, -1) if pr is decreasing
    np_idx_arrays = []
    for l in range(pr.shape[0]):
      np_idx_arrays.append(np.arange(pr[l,0], pr[l,1] + inc[l], inc[l]))

    point_list[0][counter:counter+sub_range_sizes[ipr]] = _ijk_to_func(np_idx_arrays, loc, n_vtx_S, order)
    counter += sub_range_sizes[ipr]

  return point_list