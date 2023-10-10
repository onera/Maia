import numpy as np
from cmaia.utils import layouts

def interweave_arrays(array_list):
  #https://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays
  first  = array_list[0]
  number = len(array_list)
  output = np.empty(number*first.size, first.dtype)
  for i,array in enumerate(array_list):
    output[i::number] = array
  return output

def single_dim_pr_to_pl(pr, distrib=None):
  assert pr.shape[0] == 1
  if distrib is not None:
    return np.arange(pr[0,0]+distrib[0], pr[0,0]+distrib[1], dtype=pr.dtype).reshape((1,-1), order='F')
  else:
    return np.arange(pr[0,0], pr[0,1]+1, dtype=pr.dtype).reshape((1,-1), order='F')

def compress(t):
  """
  Inverse of np.repeat. Go back to array with np.repeat(val, np.diff(idx))
  """
  assert t.size > 0
  diff = t[:-1] != t[1:]
  n_diff = diff.sum()
  idx = np.empty(n_diff+2, np.int32)
  idx[0] = 0
  idx[1:-1] = np.where(diff)[0] + 1
  idx[-1] = t.size
  val = t[idx[:-1]]
  return idx, val

def indexed_to_interlaced(idx, array):
  """ Create an interlaced array from two offset + data arrays (eg. cgns 3 from cgns 4)"""
  return layouts.indexed_to_interleaved_connectivity(idx, array)

def interlaced_to_indexed(n_elem, array):
  """ Create two offset + data arrays from an interlaced array (eg. cgns 4 from cgns 3)"""
  return layouts.interleaved_to_indexed_connectivity(n_elem, array)

def concatenate_np_arrays(arrays, dtype=None):
  """
  Merge the input array such that output array is F ordered and
  have CGNS coherent shape ( (N,) or (IndexDimension, N) ).
  Also return an idx array to indicate how the array has been concatenated

  If list is empty, a flat array (0,) of type dtype is returned.
  """
  if arrays == []:
    if dtype is None:
      raise ValueError("Can not concatenate empty list of arrays if dtype is not provided")
    return np.zeros(1, np.int32), np.empty(0, dtype)

  merged_idx = sizes_to_indices([array.shape[-1] for array in arrays], dtype=np.int32)
  stacked = np.hstack(arrays)
  if dtype is not None:
    stacked = safe_int_cast(stacked, dtype)
  return merged_idx, stacked

def concatenate_point_list(point_lists, dtype=None):
  """
  Merge all the PointList arrays in point_lists list
  into a flat 1d array and an index array
  """
  arrays = [pl[0,:] for pl in point_lists]
  return concatenate_np_arrays(arrays, dtype)

def sizes_to_indices(nb_array, dtype=None):
  """ Create and offset array from a size array """
  nptype = dtype if dtype else np.asarray(nb_array).dtype
  offset_array = np.empty(len(nb_array)+1, dtype=nptype)
  offset_array[0] = 0
  np.cumsum(nb_array, out=offset_array[1:])
  return offset_array

def shift_nonzeros(array, shift):
  """
  Add the scalar value shift to the element of array that are not
  equal to 0 (inplace)
  """
  array += shift * (array != 0)

def shift_absvalue(array, shift):
  """
  Add the scalar value shift to the element of array
  regardless of their sign
  """
  if shift == 0: return
  neg = array < 0
  np.abs(array, out=array)
  array += shift
  array[neg] *= -1

def shifted_to_local(array, offset):
  """ Assuming that offset describes intervals and array global
  values between offset[0]; offset[N], retrieve the
  interval + position within this interval of each value """
  interval_num = np.searchsorted(offset, array)
  output = array - offset[interval_num - 1]
  return output, interval_num.astype(np.int32)

def reverse_connectivity(ids, idx, array):
  """
  Reverse an strided array (idx+array) supported by some elements whose id is given by ids
  Return a strided array(r_idx+r_array) and the ids of (initially childs) elements
  supporting it
  """
  r_ids, counts = np.unique(array, return_counts=True)
  sort_idx = np.argsort(array)
  sizes = np.diff(idx)
  extended_ids = np.repeat(ids, sizes)
  r_array = extended_ids[sort_idx]
  r_idx = sizes_to_indices(counts)

  return (r_ids, r_idx, r_array)

def multi_arange(starts, stops):
  """
  Create concatenated np.arange of integers for multiple start/stop
  See https://codereview.stackexchange.com/questions/83018/
  vectorized-numpy-version-of-arange-with-multiple-start-stop

  This is equivalent to 
  np.concatenate([np.arange(start,stop) for start,stop in zip(starts,stops)])
  but much faster. Don't remplace it !

  """
  assert len(starts)==len(stops)
  stops = np.asarray(stops)
  l = stops - starts # Lengths of each range.
  return np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())

def arange_with_jumps(multi_interval,jumps):
  """
  Create an arange, but where sub-intervals are removed
  """
  multi_interval = np.asarray(multi_interval)
  jumps = np.asarray(jumps)
  return multi_arange(multi_interval[ :-1][~jumps],
                      multi_interval[1:  ][~jumps])

def jagged_extract(idx_array, array, ids):
  extracted_array = array[multi_arange(idx_array[ids], idx_array[ids+1])]
  sizes = np.diff(idx_array)
  extracted_sizes = sizes[ids]
  extracted_idx = sizes_to_indices(extracted_sizes)
  return extracted_idx, extracted_array

def jagged_merge(idx1, array1, idx2, array2):
  """
  Interwave two jagged arrays of same n_elt
  """
  assert array1.dtype == array2.dtype
  return layouts.jagged_merge(idx1, array1, idx2, array2)

def roll_from(array, start_idx = None, start_value = None, reverse = False):
  """
  Return a new array starting from given index (or value), in normal or reversed order
  """
  assert (start_idx is None) != (start_value is None)
  if start_idx is None:
    start_idx = np.where(array == start_value)[0][0]

  return np.roll(array, -start_idx) if not reverse else np.roll(array[::-1], start_idx + 1)

def others_mask(array, ids):
  """
  Return a mask usefull to access elements of array whose local index *are not* in ids array
  """
  mask = np.ones(array.size, dtype=bool)
  mask[ids] = False
  return mask

def any_in_range(array, start, end, strict=False):
  """
  Return True if any element of array is in interval
  [start, end]. In is large by defaut and strict is strict==True
  """
  np_array = np.asarray(array)
  return ((start <  np_array) & (np_array <  end)).any() if strict\
    else ((start <= np_array) & (np_array <= end)).any()

def all_in_range(array, start, end, strict=False):
  """
  Return True if all the elements of array are in interval
  [start, end]. In is large by defaut and strict is strict==True
  """
  np_array = np.asarray(array)
  return ((start <  np_array) & (np_array <  end)).all() if strict\
    else ((start <= np_array) & (np_array <= end)).all()

def transform_cart_matrix(vectors, translation=np.zeros(3), rotation_center=np.zeros(3), rotation_angle=np.zeros(3)):
  """
  Apply the defined cartesian transformation on concatenated components of vectors described by :
  [vx1 vx2 ... vxN]
  [vy1 vy2 ... vyN]
  [vz1 vz2 ... vzN]
  and return the modified components of the vectors in the same format
  """
  rotation_center = np.array(rotation_center).reshape((3,1))
  alpha, beta, gamma  = rotation_angle
  rotation_matx = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
  rotation_maty = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
  rotation_matz = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
  rotation_mat  = np.dot(rotation_matx, np.dot(rotation_maty, rotation_matz))
  rotated_vectors = ((np.dot(rotation_mat, vectors-rotation_center)+rotation_center).T + translation).T
  return rotated_vectors.astype(vectors.dtype,copy=False)

def transform_cart_vectors(vx, vy, vz, translation=np.zeros(3), rotation_center=np.zeros(3), rotation_angle=np.zeros(3)):
  """
  Apply the defined cartesian transformation on separated components of vectors and return a tuple with each of the modified components of the vectors
  """
  assert vx.shape == vy.shape == vz.shape
  if vx.ndim == 1:
    vectors = np.array([vx,vy,vz], order='F')
  else:
    vectors = np.array([vx.flatten('F'), vy.flatten('F'), vz.flatten('F')], order='F')
  
  modified_components = transform_cart_matrix(vectors, translation, rotation_center, rotation_angle)

  if vx.ndim == 1:
    return (modified_components[0], modified_components[1], modified_components[2])
  else:
    return (modified_components[0].reshape(vx.shape, order='F'),
            modified_components[1].reshape(vy.shape, order='F'),
            modified_components[2].reshape(vz.shape, order='F'))

def transform_cart_matrix_2d(vectors, translation=np.zeros(2), rotation_center=np.zeros(2), rotation_angle=0.):
  """
  Apply the defined cartesian transformation on concatenated components of vectors described by :
  [vx1 vx2 ... vxN]
  [vy1 vy2 ... vyN]
  and return the modified components of the vectors in the same format
  """
  theta = rotation_angle
  rotation_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
  _rotation_center = np.array(rotation_center).reshape((-1,1))
  _translation = np.array(translation).reshape((-1,1))
  rotated_vectors = (np.dot(rotation_mat, vectors-_rotation_center)+_rotation_center) + _translation
  return rotated_vectors.astype(vectors.dtype,copy=False)


def transform_cart_vectors_2d(vx, vy, translation=np.zeros(2), rotation_center=np.zeros(2), rotation_angle=0.):
  assert vx.shape == vy.shape
  if vx.ndim == 1:
    vectors = np.array([vx,vy], order='F')
  else:
    vectors = np.array([vx.flatten('F'), vy.flatten('F')], order='F')
  modified_components = transform_cart_matrix_2d(vectors, translation, rotation_center, rotation_angle)
  if vx.ndim == 1:
    return (modified_components[0], modified_components[1])
  else:
    return (modified_components[0].reshape(vx.shape, order='F'), modified_components[1].reshape(vy.shape, order='F'))


def safe_int_cast(array, dtype):
  """ Util function to perfom I4 <--> I8 conversions with bounds test """
  if array.dtype == dtype:
    return array
  if array.dtype == np.int32 and dtype == np.int64:
    return array.astype(np.int64)
  if array.dtype == np.int64 and dtype == np.int32:
    if np.abs(array).max(initial=0) <= np.iinfo(np.int32).max:
      return array.astype(np.int32)
    else:
      raise OverflowError("Can not cast array to int32 type")
  raise ValueError("Incompatibles dtypes for numpy cast")
