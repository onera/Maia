import numbers
import numpy as np

import cmaia.utils as cutils
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
  Return a strided array(r_idx+r_array) and the ids of (initially children) elements
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
  dtype = starts.dtype if isinstance(starts, np.ndarray) else int
  assert len(starts)==len(stops)
  stops = np.asarray(stops)
  l = stops - starts # Lengths of each range.
  return np.repeat(stops - l.cumsum(dtype=dtype), l) + np.arange(l.sum(), dtype=dtype)

def arange_with_jumps(multi_interval,jumps):
  """
  Create an arange, but where sub-intervals are removed
  """
  multi_interval = np.asarray(multi_interval)
  jumps = np.asarray(jumps)
  return multi_arange(multi_interval[ :-1][~jumps],
                      multi_interval[1:  ][~jumps])

def repeated_arange(counts, start=0, stop=None, step=1, dtype=None):
  if stop is None:
    stop = start+counts.size
  else:
    assert isinstance(counts, numbers.Integral) or stop-start == step*counts.size
  return np.repeat(np.arange(start, stop, step, dtype), counts)

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

def unique_sorted(sorted_array, return_counts=False):
  """ A faster implementation of np.unique() if input array
  is sorted
  """
  is_new = np.empty(sorted_array.size, bool)
  if sorted_array.size > 0:
    is_new[0] = True
    is_new[1:] = sorted_array[1:] != sorted_array[:-1]

  unique_array = sorted_array[is_new]

  if not return_counts:
    return unique_array
  
  counts_idx = np.empty(unique_array.size+1, int)
  counts_idx[:-1] = np.arange(is_new.size)[is_new]
  counts_idx[-1] = sorted_array.size

  counts = np.diff(counts_idx)

  return unique_array, counts


def is_unique_strided(array, stride, method='hash'):
  """
  For a cst strided array (eg. a connectivity), return a bool array indicating
  for each element if it appears only once (w/ considering ordering)
  """
  assert isinstance(stride, int), "Only constant stride is supported"
  n_elt = array.size // stride
  if method == 'hash':
    return cutils.is_unique_cst_stride_hash(n_elt, stride, array)
  elif method =='sort':
    return cutils.is_unique_cst_stride_sort(n_elt, stride, array)
  else:
    raise ValueError(f"Method must be one of ['hash', 'sort']")

def reverse_by_stride(array_idx, array, inplace=False):
  """
  Reverse each interval of an array.
  NB : the values are only sorted within each interval, there is no reverse between intervals.
  """
  if inplace:
    reversed_array = array
  else:
    reversed_array = array.copy()
  layouts.reverse_by_stride(array_idx, reversed_array)

  return reversed_array

def sort_by_stride(array_idx, array, inplace=False):
  """
  Sort each stride of an array.
  NB : the values are only sorted within each interval, there is no sorting between intervals.
  """
  if inplace:
    sorted_array = array
  else:
    sorted_array = array.copy()
  cutils.sort_by_stride(array_idx, sorted_array)
  #for i in range(idx.size-1):
  #  sorted_array[idx[i]:idx[i+1]].sort()
  return sorted_array

def make_unique_by_stride(array_idx, array):
  """
  Take a strided input array, and create a new one without repetitions
  within each interval.
  NB : the subintervals are not sorted ; input order is preserved
  """
  return cutils.make_unique_by_stride(array_idx, array)

def roll_once_by_stride(array_idx, array):
  """
  numpy.roll (with shift := -1) within each interval
  [34, 65, 33, 1,     39, 54, 2, 53, 3] --> [65, 33, 1, 34,     54, 2, 53, 3, 39]
  """
  values = array[array_idx[:-1]].copy()
  extended = np.insert(array, array_idx[1:], values)
  rm_idx = array_idx[:-1] + np.arange(array_idx.size-1)
  return np.delete(extended, rm_idx)

def take_strided2(array_idx, array, indices):
  """ Same as take_strided, but also return the idx array
  TODO : Other function should be removed
  """

  out_size = array_idx[indices+1] - array_idx[indices]
  out_idx  = sizes_to_indices(out_size, array_idx.dtype)

  out = np.empty(out_size.sum(), array.dtype)
  layouts.take_strided(array_idx, array, indices, out)

  return out_idx, out
def take_strided(array_idx, array, indices):
  """
  An equivalent to numpy.take (a[ind]), but with strided values in array
  Indices is the list of idx to extract; for each indices, the whole "grap" of strided
  values will be extracted
  Example:
  Given inputs:
    a_idx    = [0, 3, 4, 6]  (gather 3 values, then 1 value, then 2 values)
    a_val    = [10,11,12, 100, 1000, 1001] (input array)
    indices  = [2,0] (indices of groups that we want to take)
  We gather a_val according to a_idx: [[10,11,12], [100], [1000, 1001]]
  Then we return the groups at indices [2,0]
  So in the end, we have:
    take_strided(a_idx, a_val, indices) = [1000, 1001,  10,11,12]

  """
  out_size = (array_idx[indices+1] - array_idx[indices]).sum()
  out = np.empty(out_size, array.dtype)
  layouts.take_strided(array_idx, array, indices, out)

  return out

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

def matmul_cart_vectors(vectors, transform_matrix):
  """
  Apply the transformation matrix on another matrix composed with components of vectors and return each of the modified components of the vectors
  """
  assert all(v.shape == vectors[0].shape for v in vectors)
   
  _vectors = np.array([v.reshape(-1, order='F') for v in vectors], order='F')
  _res     = np.dot(transform_matrix, _vectors)
  
  return tuple(r.reshape(v.shape, order='F') for r,v in zip(_res, vectors))

def create_transform_matrix(revolution_axis=(0, 0, 1)):  
  """Create a transform matrix from any axis revolution and return the transformation matrix from the former basis toward the new basis.

  Input is any revolution axis but must have cartesian coordinates.
  Transform matrix is defined by a plane equation

  .. math::
     \\ ax + by + cz = 0

  where (a, b, c) is the direction vector of the plane equation.

  Args:
    revolution_axis (tuple, list, array) : Constant axis
                                           By default it set on z-axis.
  """
  assert not (np.array_equal(np.array(revolution_axis), np.zeros(3)))

  revolution_axis = np.asarray(revolution_axis)
  revolution_axis = revolution_axis / np.linalg.norm(revolution_axis)

  if revolution_axis[0] != 0:
    revolution_axis_bis = np.array([-revolution_axis[1]/revolution_axis[0], 1, 0])
  elif revolution_axis[1] != 0:
    revolution_axis_bis = np.array([0, -revolution_axis[2]/revolution_axis[1], 1])
  elif revolution_axis[2] != 0:
    revolution_axis_bis = np.array([1, 0, -revolution_axis[1]/revolution_axis[2]])
  
  revolution_axis_ter = np.cross(revolution_axis, revolution_axis_bis)

  transform_matrix = np.array([revolution_axis, revolution_axis_bis, revolution_axis_ter], order='F')
     
  return transform_matrix

def _homogeneous_matrix_to_transform(homo_matrix):
  """ Inverse of _transform_to_homogeneous_matrix : recompute rotation angle and translation from
  an homogeneous matrix. As for the opposite function, the order used to apply rotation angle corresponds to :
  - a Z, then Y, then X extrinsic rotation, or equivalently
  - a X, then Y, then Z intrinsic rotation
  """
  dim = homo_matrix.shape[0] - 1
  Rmat = homo_matrix[0:dim, 0:dim]
  translation = homo_matrix[0:dim, dim]
  rotation_center = np.zeros_like(translation)
  if dim == 3:
    rotation_angle = np.empty_like(translation)
    rotation_angle[0] = np.arctan2(-Rmat[1,2], Rmat[2,2])
    rotation_angle[1] = np.arcsin ( Rmat[0,2])
    rotation_angle[2] = np.arctan2(-Rmat[0,1], Rmat[0,0])
  elif dim == 2:
    rotation_angle = np.arcsin(Rmat[1,0])
  return translation, rotation_center, rotation_angle

def _transform_to_homogeneous_matrix(translation=np.zeros(3), rotation_center=np.zeros(3), rotation_angle=np.zeros(3)):
  """ Combine Transform data coming from CGNS (rotation_angle, rotation_center, translation) into 
  an homogeneous matrix of size 4x4 (in 3d). This matrix can be applied to a vector (vx, vy, vz, 1).
  # https://www.f-legrand.fr/scidoc/docmml/graphie/geometrie/affine/affine.html
  
  Important : if dim==3, the order used to apply rotation angle corresponds to :
  - a Z, then Y, then X extrinsic rotation, or equivalently
  - a X, then Y, then Z intrinsic rotation
  # https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
  """
  dim = len(translation)
  homo_matrix = np.zeros((dim+1,dim+1))
  if dim == 3:
    alpha, beta, gamma  = rotation_angle
    rotation_matx = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    rotation_maty = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    rotation_matz = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    rotation_mat  = np.dot(rotation_matx, np.dot(rotation_maty, rotation_matz))
  elif dim == 2: # Rotation angle is scalar
    theta = rotation_angle
    rotation_mat  = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
  homo_matrix[0:dim, 0:dim] = rotation_mat
  homo_matrix[0:dim,   dim] = rotation_center - np.dot(rotation_mat, rotation_center) + translation
  homo_matrix[dim,dim] = 1
  
  return homo_matrix
def transform_cart_matrix(vectors, translation=np.zeros(3), rotation_center=np.zeros(3), rotation_angle=np.zeros(3)):
  """
  Apply the defined cartesian transformation on concatenated components of vectors described by :
  [vx1 vx2 ... vxN]
  [vy1 vy2 ... vyN]
  [vz1 vz2 ... vzN]
  and return the modified components of the vectors in the same format
  """
  homo_matrix = _transform_to_homogeneous_matrix(translation, rotation_center, rotation_angle)
  homo_vector = np.ones((4, vectors.shape[1]))
  homo_vector[0:3,:] = vectors
  return np.dot(homo_matrix, homo_vector)[0:3,:]

def transform_cart_matrix_2d(vectors, translation=np.zeros(2), rotation_center=np.zeros(2), rotation_angle=0.):
  """
  Apply the defined cartesian transformation on 2D concatenated components of vectors described by :
  [vx1 vx2 ... vxN]
  [vy1 vy2 ... vyN]
  and return the modified components of the vectors in the same format
  """
  homo_matrix = _transform_to_homogeneous_matrix(translation, rotation_center, rotation_angle)
  homo_vector = np.ones((3, vectors.shape[1]))
  homo_vector[0:2,:] = vectors
  return np.dot(homo_matrix, homo_vector)[0:2,:]

def transform_cart_vectors(vx, vy, vz, translation=np.zeros(3), rotation_center=np.zeros(3), rotation_angle=np.zeros(3)):
  """
  Apply the defined cartesian transformation on separated components of vectors and return a tuple with each of the modified components of the vectors
  """
  assert vx.shape == vy.shape == vz.shape
  if vx.ndim == 1:
    vectors = np.array([vx,vy,vz,np.ones(vx.size)], order='F')
  else: #Manage structured blocks
    vectors = np.array([vx.flatten('F'), vy.flatten('F'), vz.flatten('F'), np.ones(vx.size)], order='F')
  
  homo_matrix = _transform_to_homogeneous_matrix(translation, rotation_center, rotation_angle)
  modified_components = np.dot(homo_matrix, vectors)[0:3,:]

  if vx.ndim == 1:
    return (modified_components[0], modified_components[1], modified_components[2])
  else:
    return (modified_components[0].reshape(vx.shape, order='F'),
            modified_components[1].reshape(vy.shape, order='F'),
            modified_components[2].reshape(vz.shape, order='F'))



def transform_cart_vectors_2d(vx, vy, translation=np.zeros(2), rotation_center=np.zeros(2), rotation_angle=0.):
  assert vx.shape == vy.shape
  if vx.ndim == 1:
    vectors = np.array([vx,vy,np.ones(vx.size)], order='F')
  else: #Manage structured blocks
    vectors = np.array([vx.flatten('F'), vy.flatten('F'), np.ones(vx.size)], order='F')
  homo_matrix = _transform_to_homogeneous_matrix(translation, rotation_center, rotation_angle)
  modified_components = np.dot(homo_matrix, vectors)[0:2,:]
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
