import pytest
import numpy as np


from maia.utils.ndarray import vstride as vs

###############################################################################
# Raw init, attributes and operators

def test_init():
  arr = vs.VStrideArray(None, np.array([0, 2, 5]), np.array([0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9]))
  assert arr.dtype == float and arr.counts.dtype == int and arr.displs.dtype == int

  arr = vs.VStrideArray(None, np.array([0, 2, 5], np.int32), np.array([0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9], np.float32))
  assert arr.dtype == np.float32 and arr.counts.dtype == np.int32 and arr.displs.dtype == np.int32

  counts = arr.counts
  assert np.array_equal(counts, [0,2,5]) and counts.flags.writeable == False
  displs = arr.displs
  assert np.array_equal(displs, [0,0,2,7]) and displs.flags.writeable == False
  assert len(arr) == 3 and arr.dsize == 7


  # Failures
  with pytest.raises(AssertionError): # No stride provided
    vs.VStrideArray(None, None, np.array([0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9], np.float32))
  with pytest.raises(AssertionError): # No value provided
    vs.VStrideArray(None, np.array([0,2,5]), None)
  with pytest.raises(AssertionError): # Inconsistent size
    vs.VStrideArray(None, np.array([0, 2, 4]), np.array([0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9]))
  with pytest.raises(AssertionError): # Inconsistent size
    vs.VStrideArray(np.array([0,2,8]), None, np.array([0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9]))
  with pytest.raises(AssertionError): # Wrong stride dtype
    vs.VStrideArray(None, np.array([0, 2.2, 5]), np.array([0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9]))

def test_getitem():
  arr = vs.from_counts([0, 2, 5], [0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9])
  block = arr[2]
  assert np.array_equal(block, [.1, .7, .2, .6, .9]) and block.dtype == arr.dtype
  block = arr[np.int32(2)]
  assert np.array_equal(block, [.1, .7, .2, .6, .9]) and block.dtype == arr.dtype
  block[3] = -999   # block is a view, arr should be updated
  assert arr.values[5] == -999

  with pytest.raises(IndexError):
    arr[3]
  with pytest.raises(TypeError):
    arr['a']

  arr[1] = [-8, -9] # Set item. Cast to arr.dtype
  assert np.array_equal(arr.values[0:2], [-8., -9.])
  arr[1] = -7 # Set item with broadcasting
  assert np.array_equal(arr.values[0:2], [-7., -7.])
  with pytest.raises(ValueError):
    arr[1] = [-7,-8,-9] # Set item with wrong size

  # Setting counts/displs is forbiden
  with pytest.raises(AttributeError):
    arr.counts = np.array([5,2,0])
  with pytest.raises(ValueError):
    arr.displs[0] += 2

def test_operators():
  a = vs.from_counts([0, 2, 5], [0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9])

  # Unary op
  b = -a
  c = abs(b)
  assert b._counts is a._counts and b._displs is a._displs # Share ref
  assert np.array_equal(b.values, -a.values) and b.dtype == a.dtype
  assert c._counts is a._counts and c._displs is a._displs # Share ref
  assert np.array_equal(c.values, a.values) and c.dtype == a.dtype

  # Binary Op
  c = a + b
  assert c._counts is a._counts and c._displs is a._displs # Share ref
  assert (c.values == 0).all() and c.dsize == a.dsize and c.dtype == a.dtype
  with pytest.raises(ValueError):
    _ = a ** vs.from_counts([7], np.arange(7))  # Different len
  with pytest.raises(ValueError):
    _ = a % vs.from_counts([5,2,0], np.arange(7))  # Different strides

  d = c + np.array([3,4,5]) # Extend
  assert (d.values == [4,4,5,5,5,5,5]).all() and d.dsize == c.dsize and d.dtype == c.dtype
  with pytest.raises(ValueError):
    _ = c / np.array([4,5]) # Wrong size

  d = c - 4 # Broadcast full
  assert (d.values == -4).all() and d.dsize == c.dsize and d.dtype == c.dtype

  with pytest.raises(TypeError):
    _ = c < {'will not' : 'work'}

  # Inplace Op
  values_bck = a.values
  a *= vs.from_counts([0, 2, 5], [10, -10, 10, -10, 10, -10, 10])
  assert a.values is values_bck and np.array_equal(a.values, [3.,-5,1,-7,2,-6,9])
  with pytest.raises(ValueError):
    a += vs.from_counts([7], np.arange(7))  # Different len
  with pytest.raises(ValueError):
    a -= vs.from_counts([5,2,0], np.arange(7))  # Different strides

  a /= 10.
  assert a.values is values_bck and np.array_equal(a.values, [.3,-.5,.1,-.7,.2,-.6,.9])

  a **= np.array([1,2,0])
  assert a.values is values_bck and np.array_equal(a.values, [.09,.25, 1,1,1,1,1])
  with pytest.raises(ValueError):
    a &= np.array([4,5]) # Wrong size

  with pytest.raises(TypeError):
    a <<= {'will not' : 'work'}

def test_restride():
  arr = vs.array([[0.3, 0.5], [0.1, 0.7, 0.2, 0.6, 0.9]])
  tmp1, tmp2 = arr.counts, arr.displs # Force computation
  assert arr._counts is not None and arr._displs is not None
  values_bck = arr.values
  arr.restride(counts=np.array([2,3,2]))
  assert arr._displs is None and np.array_equal(arr._counts, [2,3,2])
  assert len(arr) == 3
  assert arr.displs is not tmp2 and arr.values is values_bck

  with pytest.raises(AssertionError): # Incomptible stride
    arr.restride(displs=np.array([0, 8]))

def test_repr():
  # Basic
  arr = vs.array([[0.3, 0.5], [0.1, 0.7, 0.2, 0.6, 0.9]])
  assert arr.__repr__() == \
    'vsarray([\n  [0.3, 0.5],\n  [0.1, 0.7, 0.2, 0.6, 0.9],\n], dtype=float64)'

  # Alignement
  arr = vs.array([[0.3, 0.5], [0.1, 0.007, 0.2, 0.6, 0.9]], dtype=np.float32)
  assert arr.__repr__() == \
    'vsarray([\n  [0.3  , 0.5  ],\n  [0.1  , 0.007, 0.2  , 0.6  , 0.9  ],\n], dtype=float32)'

  # Linebreak
  arr = vs.array([np.arange(24), np.ones(5)], dtype=int)
  assert arr.__repr__() == \
    'vsarray([\n  [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n   ' \
    '17, 18, 19, 20, 21, 22, 23],\n  [ 1,  1,  1,  1,  1],\n], dtype=int64)'

  # Multiline
  arr = vs.array([np.arange(100*k, 100*(k+1)) for k in range(51)])
  assert arr.__repr__() == \
    'vsarray([\n  [   0,    1,    2, ... ,   97,   98,   99],\n  ' \
    '[ 100,  101,  102, ... ,  197,  198,  199],\n  '\
    '[ 200,  201,  202, ... ,  297,  298,  299],\n  ...,\n  ' \
    '[4800, 4801, 4802, ... , 4897, 4898, 4899],\n  '\
    '[4900, 4901, 4902, ... , 4997, 4998, 4999],\n  '\
    '[5000, 5001, 5002, ... , 5097, 5098, 5099],\n], dtype=int64)'

###############################################################################
# "Consummers methods"

def test_to_array_list():
  arr = vs.from_counts([0, 2, 5], [0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9])
  arrays = arr.to_array_list()
  assert len(arrays) == 3
  assert all(a.dtype == float for a in arrays)
  assert np.array_equal(arrays[0], [])
  assert np.array_equal(arrays[1], [0.3, 0.5])
  assert np.array_equal(arrays[2], [0.1, 0.7, 0.2, 0.6, 0.9])

def test_to_masked_array():
  arr = vs.from_counts([0, 2, 5], [0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9])
  ma = arr.to_masked_array()
  assert isinstance(ma, np.ma.masked_array)
  assert ma.ndim == 2 and ma.shape == (3,5) and ma.dtype == float
  assert np.array_equal(ma.mask, [[1,1,1,1,1], [0,0,1,1,1], [0,0,0,0,0]])
  assert np.array_equal(ma.data[~ma.mask], [0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9])

def test_reduce():
  arr = vs.from_counts([0, 2, 5], [0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9])
  red = arr.reduce(vs.ReduceOp.SUM)
  assert np.array_equal([0, .8, 2.5], red) and red.dtype == float

  # This case failed with numpy reduceat
  arr = vs.from_counts([0, 2, 5, 0, 0], [0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9])
  red = arr.reduce(vs.ReduceOp.SUM)
  assert np.array_equal([0, .8, 2.5, 0., 0.], red) and red.dtype == float

  arr = vs.from_counts([2, 3, 2], [3, 5, 1, 7, 2, 6, 9], dtype=np.int32)
  red = arr.reduce(vs.ReduceOp.MAX)
  assert np.array_equal([5, 7, 9], red) and red.dtype == np.int32
  red = arr.reduce(vs.ReduceOp.PROD)
  assert np.array_equal([3*5, 1*7*2, 6*9], red) and red.dtype == np.int32
  
  # Test with empty vals
  arr = vs.from_counts([0, 2, 5], [0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9])
  red = arr.reduce(vs.ReduceOp.MIN)
  assert np.array_equal([+np.inf, .3, .1], red) and red.dtype == float
  red = arr.reduce(vs.ReduceOp.MAX)
  assert np.array_equal([-np.inf, .5, .9], red) and red.dtype == float

  arr = vs.from_counts([0, 2, 5], [3, 5, 1, 7, 2, 6, 9], dtype=np.int32)
  red = arr.reduce(vs.ReduceOp.MIN)
  assert np.array_equal([np.iinfo(np.int32).max, 3, 1], red) and red.dtype == np.int32
  red = arr.reduce(vs.ReduceOp.MAX)
  assert np.array_equal([np.iinfo(np.int32).min, 5, 9], red) and red.dtype == np.int32

  arr = vs.from_counts([0, 2, 4], [True, True, False, True, True, False])
  red = arr.reduce(vs.ReduceOp.MIN)
  assert np.array_equal([True, True, False], red) and red.dtype == bool
  red = arr.reduce(vs.ReduceOp.MAX)
  assert np.array_equal([False, True, True], red) and red.dtype == bool

  with pytest.raises(ValueError):
    arr = vs.from_counts([0, 2, 5], [0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9])
    red = arr.reduce(vs.ReduceOp.BAND)



###############################################################################
# Constructeurs

def test_from_counts():
  a = vs.from_counts([0, 2, 5], [0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9])

  assert isinstance(a, vs.VStrideArray)
  assert a.dtype == float and a.counts.dtype == int
  assert np.array_equal(a.values, [0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9])
  assert np.array_equal(a.displs, [0, 0, 2, 7])

  # Cst counts
  b = vs.from_counts(3, [0.3, 0.5, 0.1, 0.7, 0.2, 0.6])
  assert b.dtype == float and b.counts.dtype == int
  assert np.array_equal(b.values, [0.3, 0.5, 0.1, 0.7, 0.2, 0.6])
  assert np.array_equal(b.displs, [0, 3, 6])

  a = vs.from_counts([0, 2, 5], [0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9], dtype=np.float32)
  assert a.dtype == np.float32 and a.counts.dtype == int

  a = vs.from_counts([], [], dtype=bool)
  assert len(a) == 0 and a.dtype == bool

  with pytest.raises(AssertionError):  # Unconsistent size / counts
    a = vs.from_counts([2,2], [1,2,3])
  with pytest.raises(AssertionError):  # Non integer type
    a = vs.from_counts(np.array([2,1], float), [1,2,3]) # Unconsistent size / counts
  with pytest.raises(AssertionError):  # Unconsistent scalar counts
    a = vs.from_counts(4, [1,2,3,4,5,6,7])

def test_from_displs():
  a = vs.from_displs([0, 0, 2, 7], [0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9])

  assert isinstance(a, vs.VStrideArray)
  assert a.dtype == float and a.displs.dtype == int
  assert np.array_equal(a.values, [0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9])
  assert np.array_equal(a.counts, [0, 2, 5])

  a = vs.from_displs([0, 0, 2, 7], [0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9], dtype=np.float32)
  assert a.dtype == np.float32 and a.counts.dtype == int

  a = vs.from_displs([0], [], dtype=bool)
  assert len(a) == 0 and a.dtype == bool

  with pytest.raises(AssertionError):  # Unconsistent size / displs
    a = vs.from_displs([0, 2, 4], [1,2,3])
  with pytest.raises(AssertionError):  # Non integer type
    a = vs.from_displs(np.array([0, 3], float), [1,2,3]) # Unconsistent size / counts

def test_array():
  # From list
  a = vs.array([[1,2], [3,4,5], [], [6]], dtype=int)
  assert len(a) == 4 and a.dtype == int and a.counts.dtype == int
  assert np.array_equal(a.counts, [2,3,0,1])
  assert np.array_equal(a.displs, [0,2,5,5,6])
  assert np.array_equal(a.values, [1,2,3,4,5,6])

  # From empty list(s)
  a = vs.array([[]])
  assert len(a) == 1 and a.dtype == float and a.counts.dtype == int

  a = vs.array([], dtype=int)
  assert len(a) == 0 and a.dtype == int and a.counts.dtype == int

  with pytest.raises(ValueError): # dtype is mandatory if empty list
    a = vs.array([])

  # From other VSarray
  a1 = vs.VStrideArray(None, np.array([2,2]), np.arange(4))
  a = vs.array(a1, dtype=float)
  assert len(a) == 2 and a.dtype == float
  a1.restride(displs=np.array([0,4], np.int32))
  a = vs.array(a1, dtype=float)
  assert len(a) == 1 and a.dtype == float and a.displs.dtype == int # New array -> int

  # From masked array
  ma = np.ma.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
              mask=[[0, 0, 1], [0, 0, 0], [0, 1, 1]])
  a = vs.array(ma)
  assert np.array_equal(a.counts, [2,3,1])
  assert np.array_equal(a.values, [1,2,4,5,6,7])

  ma.mask[2,:] = True
  a = vs.array(ma)
  assert np.array_equal(a.counts, [2,3,0])
  assert np.array_equal(a.values, [1,2,4,5,6])

  ma.mask = True
  a = vs.array(ma)
  assert np.array_equal(a.counts, [0,0,0])
  assert np.array_equal(a.values, [])
  assert a.dtype == ma.dtype

  # From full 2D array
  a = vs.array(np.array([[1,2], [3,4], [5,6], [7,8]], order='C'))
  assert a.dtype == np.int64
  assert np.array_equal(a.counts, [2,2,2,2])
  assert np.array_equal(a.values, [1,2,3,4,5,6,7,8])
  a = vs.array(np.array([[1,2], [3,4], [5,6], [7,8]], order='F'), dtype=np.int32)
  assert a.dtype == np.int32
  assert np.array_equal(a.counts, [2,2,2,2])
  assert np.array_equal(a.values, [1,2,3,4,5,6,7,8])

  # From random object
  with pytest.raises(ValueError):
    vs.array('will fail')


###############################################################################
# Indexing methods

def test_insert():
  a = vs.from_counts([0, 2, 5], [.3, .5, .1, .7, .2, .6, .9])
  
  assert vs.insert(a, 0, [.4, .2]) == vs.from_counts([2,0,2,5], [.4, .2, .3, .5, .1, .7, .2, .6, .9])
  assert vs.insert(a, 3, [])       == vs.from_counts([0,2,5,0], a.values)
  assert vs.insert(a, 2, [.8])     == vs.from_counts([0,2,1,5], [.3, .5, .8, .1, .7, .2, .6, .9])

  with pytest.raises(IndexError):
    vs.insert(a, 19, [.8])

  assert vs.insert(a, [0,2], vs.array([[.8, .9], [.2,.7,.4]])) == \
    vs.from_counts([2, 0, 2, 3, 5], [.8, .9, .3, .5, .2, .7, .4, .1, .7, .2, .6, .9])

  assert vs.insert(a, [], vs.array([], dtype=float)) == a
  assert vs.insert(a, [0,2], vs.array([[], []], dtype=float)) == vs.from_counts([0,0,2,0,5], a.values)

  with pytest.raises(IndexError):
    vs.insert(a, [0,2,8], vs.array([[], [], []], dtype=float))

  with pytest.raises(ValueError):
    vs.insert(a, [0,2,1], vs.array([[.8, .9], [.2,.7,.4]]))
  with pytest.raises(TypeError):
    a2 = vs.from_counts([0, 2, 5], [3, 5, 1, 7, 2, 6, 9])
    vs.insert(a2, [0,2], vs.array([[.8, .9], [.2,.7,.4]]))


def test_take():
  a = vs.take(vs.from_counts([3,1,2], [10,11,12,  100,  1000,1001]), [2,0])
  assert np.array_equal(a.displs, [0,2,5]) and np.array_equal(a.values, [1000,1001, 10,11,12])
  
  a = vs.take(vs.from_displs([0,3,4,6], [10,11,12,  100,  1000,1001]), [2,2,2,0])
  assert np.array_equal(a.values, [1000, 1001, 1000, 1001, 1000, 1001, 10,11,12])

  a = vs.take(vs.from_displs([0,3,3,3,4,6], [10,11,12,  100,  1000,1001]), [1,2])
  assert np.array_equal(a.counts, [0, 0]) and np.array_equal(a.values, []) and a.dtype == int

  a = vs.take(vs.from_displs([0,3,3,3,4,6], [10,11,12,  100,  1000,1001]), [2,4,0])
  assert np.array_equal(a.displs, [0, 0, 2, 5]) and np.array_equal(a.values, [1000, 1001, 10,11,12])

  a = vs.take(vs.from_counts([], np.empty(0, float)), [])
  assert len(a) == 0 and a.dsize == 0 and a.dtype == float

def test_put():
  a = vs.from_counts([3,1,2,2,3], [10,11,12,  100,  1000,1001, 2,4, 5,6,5], dtype=int)

  vals = vs.array([[-5,-5], [-10, -6], []], dtype=int)
  b = vs.put(a, [0,4, 2], vals)
  assert b.dtype == a.dtype and len(b) == len(a)
  assert np.array_equal(b.counts, [2,1,0,2,2]) and np.array_equal(b.values, [-5,-5,  100,  2,4,  -10,-6])

  vals = vs.array([[-5,-5,-5], [-4,-4], [-3,-3,-3]], dtype=int)
  b = vs.put(a, [0,0,0], vals)
  assert b.dtype == a.dtype and len(b) == len(a)
  assert np.array_equal(b.counts, [3,1,2,2,3])
  assert np.array_equal(b.values, [-3,-3,-3,  100,  1000,1001, 2,4, 5,6,5])
  
  b = vs.put(a, 1, [88,99])
  assert b.dtype == a.dtype and len(b) == len(a)
  assert np.array_equal(b.counts, [3,2,2,2,3])
  assert np.array_equal(b.values, [10,11,12,  88,99,  1000,1001, 2,4, 5,6,5])

  with pytest.raises(IndexError):
    b = vs.put(a, [23], vs.array([[6,5,4]]))
  with pytest.raises(IndexError):
    b = vs.put(a, 23, [6,5,4])
  with pytest.raises(ValueError):
    b = vs.put(a, [0], vs.array([[6],[5,4]]))
  with pytest.raises(TypeError):
    b = vs.put(a, [2], vs.array([[6.5,4]]))
  with pytest.raises(TypeError):
    b = vs.put(a, 2, np.array([6.5,4]))

def test_delete():
  a = vs.from_counts([3,1,2,2,3], [10,11,12,  100,  1000,1001, 2,4, 5,6,5], dtype=int)

  b = vs.delete(a, [0,1,4])
  assert vs.array_equal(b, vs.array([[1000,1001], [2,4]])) and b.dtype == int

  b = vs.delete(a, [4,0,1,0,1,4])
  assert vs.array_equal(b, vs.array([[1000,1001], [2,4]])) and b.dtype == int

  b = vs.delete(a, np.empty(0, int))
  assert vs.array_equal(b, a) and b.dtype == int

  b = vs.delete(a, 2)
  assert vs.array_equal(b, vs.array([[10,11,12], [100], [2,4], [5,6,5]])) and b.dtype == int

  with pytest.raises(IndexError):
    b = vs.delete(a, [2,8,0])


###############################################################################
# Inner / Outer Algorithms


def test_flip():
  a = vs.from_counts([3, 3, 4, 3, 0, 1],
                      [34,22,191,  29,32,53,  43,93,22,95, 633,92,5,   4])
  reversed = vs.flip(a, vs.INNER_AXIS)
  assert np.array_equal(reversed.values, [191,22,34,  53,32,29,  95,22,93,43, 5,92,633,   4])

  reversed = vs.flip(a, vs.OUTER_AXIS)
  assert np.array_equal(reversed.counts, [1,0,3,4,3,3])
  assert np.array_equal(reversed.values, [4, 633,92,5, 43,93,22,95, 29,32,53, 34,22,191])

  a = vs.flip(vs.from_counts([],    np.empty(0, float)), vs.INNER_AXIS)
  a = vs.flip(vs.from_counts([0,0], np.empty(0, float)), vs.INNER_AXIS)

  with pytest.raises(ValueError):
    vs.flip(a, 'WrongAxis')

  
  # Test inner flip with mask
  a = vs.from_counts([3, 3, 4, 3, 0, 1],
                      [34,22,191,  29,32,53,  43,93,22,95, 633,92,5,   4])
  a._inner_flip(mask=np.array([False, True, False, False, True, True]))
  assert (a.values == [34,22,191, 53,32,29, 43,93,22,95, 633,92,5, 4]).all()

def test_unique():
  a = vs.from_counts([0, 3, 5], [.3, .1, .1,  .2, .7, .2, .2, .9])
  b = vs.unique(a, vs.INNER_AXIS)
  assert len(b) == 3 and b.dtype == float
  assert np.array_equal(b.counts, [0, 2, 3])
  assert np.array_equal(b.values, [.3,.1,  .2, .7, .9,])

  b = vs.unique(a, vs.OUTER_AXIS)
  assert len(b) == 3 and b.dtype == float
  assert vs.array_equal(b, vs.array([[], [.2,.7,.2,.2,.9], [.3,.1,.1]]))

  a = vs.from_counts([0, 0, 0], np.empty(0, np.int32))
  a = vs.unique(a, vs.INNER_AXIS)
  assert len(a) == 3 and a.dtype == np.int32
  assert np.array_equal(a.counts, [0,0,0]) and a.dsize == 0

  a = vs.from_counts([], np.empty(0, np.int64))
  a = vs.unique(a, vs.INNER_AXIS)
  assert len(a) == 0 and a.dtype == np.int64

  a = vs.array([[3,5], [5,1], [1,5], [5,1], [3,5,4], [3,5], [5,1]])
  b = vs.unique(a, vs.OUTER_AXIS)
  assert len(b) == 4 and b.dtype == int
  assert vs.array_equal(b, vs.array([[1,5], [3,5], [3,5,4], [5,1]]))

  a = vs.array([], dtype=float)
  assert vs.array_equal(vs.unique(a, vs.OUTER_AXIS), a)

  with pytest.raises(ValueError):
    vs.unique(a, 'WrongAxis')


def test_sort():
  a = vs.from_counts([0, 3, 5], [.3, .1, .1,  .2, .7, .2, .5, .9])
  b = vs.sort(a, vs.INNER_AXIS)
  assert len(b) == 3 and a.dtype == float
  assert np.array_equal(b.counts, [0, 3, 5])
  assert np.array_equal(b.values, [.1,.1,.3,  .2,.2,.5,.7,.9])

  b = vs.sort(a, vs.OUTER_AXIS)
  assert len(b) == 3 and a.dtype == float
  assert np.array_equal(b.counts, [0, 5, 3])
  assert np.array_equal(b.values, [.2, .7, .2, .5, .9,   .3, .1, .1])

  a = vs.from_counts([0, 0, 0], np.empty(0, np.int32))
  a = vs.sort(a, vs.INNER_AXIS)
  assert len(a) == 3 and a.dtype == np.int32
  assert np.array_equal(a.counts, [0,0,0]) and a.dsize == 0

  a = vs.from_counts([], np.empty(0, np.int64))
  a = vs.sort(a, vs.INNER_AXIS)
  assert len(a) == 0 and a.dtype == np.int64

  with pytest.raises(ValueError):
    vs.sort(a, 'WrongAxis')

def test_roll():
  a = vs.from_counts([2, 3, 5, 4], [1,2,   3,1,1,    2,7,2,5,9,   6,4,4,2])

  assert np.array_equal(vs.roll(a, 0, vs.OUTER_AXIS).values, a.values)
  assert np.array_equal(vs.roll(a, 20, vs.OUTER_AXIS).values, a.values)
  assert np.array_equal(vs.roll(a, 0, vs.INNER_AXIS).values, a.values)

  b = vs.roll(a, 2, vs.OUTER_AXIS)
  assert np.array_equal(b.counts, [5,4,2,3]) and np.array_equal(b.values, [2,7,2,5,9,  6,4,4,2,  1,2,  3,1,1])

  b = vs.roll(a, -5, vs.OUTER_AXIS)
  assert np.array_equal(b.counts, [3,5,4,2]) and np.array_equal(b.values, [3,1,1,   2,7,2,5,9,   6,4,4,2,  1,2])

  b = vs.roll(a, 2, vs.INNER_AXIS)
  assert np.array_equal(b.counts, [2,3,5,4]) and np.array_equal(b.values, [1,2,  1,1,3,  5,9,2,7,2,  4,2,6,4])

  b = vs.roll(a, -1, vs.INNER_AXIS)
  assert np.array_equal(b.counts, [2,3,5,4]) and np.array_equal(b.values, [2,1,  1,1,3,  7,2,5,9,2,  4,4,2,6])

  b = vs.roll(vs.from_counts([0,0], values=np.empty(0, float)), 2, vs.OUTER_AXIS)
  assert np.array_equal(b.counts, [0,0]) and b.dsize == 0 and b.dtype == float
  b = vs.roll(vs.from_counts([0,0], values=np.empty(0, float)), 2, vs.INNER_AXIS)
  assert np.array_equal(b.counts, [0,0]) and b.dsize == 0 and b.dtype == float
  b = vs.roll(vs.from_counts([], values=np.empty(0, float)), -2, vs.OUTER_AXIS)
  assert len(b) == 0 and b.dsize == 0 and b.dtype == float

  with pytest.raises(ValueError):
    vs.roll(a, 1, 'WrongAxis')

def test_concatenate():
  a = vs.from_counts([2, 3, 0, 4], [1,2,      3,1,1,       6,4,4,2])
  b = vs.from_counts([0, 5, 0, 1], [          2,7,2,5,9,   15])
  c = vs.from_counts([4, 2, 0, 3], [6,4,4,2,  13,17,       1,1,4])

  d = vs.concatenate([a], vs.INNER_AXIS)
  assert np.array_equal(d.counts, a.counts) and np.array_equal(d.values, a.values)
  d = vs.concatenate([a], vs.OUTER_AXIS)
  assert np.array_equal(d.counts, a.counts) and np.array_equal(d.values, a.values)

  d = vs.concatenate([a,b,c], vs.INNER_AXIS)
  assert len(d) == 4 and d.dtype == int and d.displs.dtype == int
  assert np.array_equal(d.counts, [6, 10, 0, 8])
  assert np.array_equal(d.values, [1,2,6,4,4,2,  3,1,1,2,7,2,5,9,13,17,  6,4,4,2,15,1,1,4])

  d = vs.concatenate([a,b,c], vs.OUTER_AXIS)
  assert len(d) == 3*4 and d.dtype == int and d.displs.dtype == int
  assert np.array_equal(d.counts, [2,3,0,4,0,5,0,1,4,2,0,3])
  assert np.array_equal(d.values, [1,2,      3,1,1,       6,4,4,2,           
                                             2,7,2,5,9,   15,
                                   6,4,4,2,  13,17,       1,1,4])

  with pytest.raises(ValueError):
    vs.concatenate([], vs.OUTER_AXIS)
  with pytest.raises(ValueError):
    a = vs.from_counts([2, 3, 1], np.empty(6))
    b = vs.from_counts([0, 5  ],  np.empty(5))
    vs.concatenate([a,b], vs.INNER_AXIS)
  with pytest.raises(ValueError):
    vs.concatenate([a,b], 'WrongAxis')


# Additional operators

def test_sign():
  a = vs.from_counts([2,3,1], [1,2,-3,4,0,-6])
  assert vs.array_equal(vs.sign(a), vs.from_counts([2,3,1], [1,1,-1,1,0,-1]))

  a = vs.from_counts([2,3,1], [1,2,-3,4,0,-6], dtype=float)
  b = vs.sign(a)
  assert vs.array_equal(b, vs.from_counts([2,3,1], [1,1,-1,1,0,-1])) and b.dtype == float
  b = vs.sign(a, dtype=np.int32)
  assert vs.array_equal(b, vs.from_counts([2,3,1], [1,1,-1,1,0,-1])) and b.dtype == np.int32

  b = vs.sign(vs.array([np.array([4.4, -6.53e18, 0])]), dtype=np.int16)
  assert np.array_equal(b.values, [1,-1,0]) and b.dtype == np.int16

def test_strides_equal():
  assert     vs.strides_equal(vs.from_counts([2,3,1], [1,2,3,4,5,6]),
                              vs.from_counts([2,3,1], [6,5,4,3,2,1]))
  assert     vs.strides_equal(vs.from_counts([2,3,1], [1,2,3,4,5,6]),
                              vs.from_displs([0,2,5,6], [6,5,4,3,2,1]))
  assert not vs.strides_equal(vs.from_counts([2,3,1], [1,2,3,4,5,6]),
                              vs.from_counts([3,2,1], [1,2,3,4,5,6]))
  assert not vs.strides_equal(vs.from_counts([2,3],   [1,2,3,4,5]),
                              vs.from_counts([2,3,1], [1,2,3,4,6,5]))

def test_array_equal():
  assert     vs.array_equal(vs.from_counts([2,3,1], [1,2,3,4,5,6]),
                            vs.from_counts([2,3,1], [1,2,3,4,5,6]))
  assert not vs.array_equal(vs.from_counts([2,3,1], [1,2,3,4,5,6]),
                            vs.from_counts([3,2,1], [1,2,3,4,5,6]))
  assert not vs.array_equal(vs.from_counts([2,3,1], [1,2,3,4,5,6]),
                            vs.from_counts([2,3,1], [1,2,3,4,6,5]))
  assert not vs.array_equal(vs.from_counts([2,3],   [1,2,3,4,5]),
                            vs.from_counts([2,3,1], [1,2,3,4,6,5]))
def test_array_close():
    assert vs.array_close(vs.from_counts([2,3,1], [1.,2,3,4,5,6]),
                          vs.from_counts([2,3,1], [1.,2,3,4,5,6+1e-9]))
    
    assert not vs.array_close(vs.from_counts([2,3,1], [1.,2,3,4,5,6]),
                              vs.from_counts([2,3,1], [1.,2,3,4,5,6.2]))
    