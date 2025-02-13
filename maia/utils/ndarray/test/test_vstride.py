import pytest
import numpy as np


from maia.utils.ndarray import vstride as vs

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

###############################################################################
# Constructeurs

def test_from_counts():
  a = vs.from_counts([0, 2, 5], [0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9])

  assert isinstance(a, vs.VStrideArray)
  assert a.dtype == float and a.counts.dtype == int
  assert np.array_equal(a.values, [0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9])
  assert np.array_equal(a.displs, [0, 0, 2, 7])

  a = vs.from_counts([0, 2, 5], [0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9], dtype=np.float32)
  assert a.dtype == np.float32 and a.counts.dtype == int

  a = vs.from_counts([], [], dtype=bool)
  assert len(a) == 0 and a.dtype == bool

  with pytest.raises(AssertionError):  # Unconsistent size / counts
    a = vs.from_counts([2,2], [1,2,3])
  with pytest.raises(AssertionError):  # Non integer type
    a = vs.from_counts(np.array([2,1], float), [1,2,3]) # Unconsistent size / counts

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

  # From other Jagged
  a1 = vs.VStrideArray(None, np.array([2,2]), np.arange(4))
  a = vs.array(a1, dtype=float)
  assert len(a) == 2 and a.dtype == float

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
  with pytest.raises(TypeError):
    b = vs.put(a, [2], vs.array([[6.5,4]]))


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

def test_unique():
  a = vs.from_counts([0, 3, 5], [.3, .1, .1,  .2, .7, .2, .2, .9])
  a = vs.unique(a, vs.INNER_AXIS)
  assert len(a) == 3 and a.dtype == float
  assert np.array_equal(a.counts, [0, 2, 3])
  assert np.array_equal(a.values, [.3,.1,  .2, .7, .9,])

  a = vs.from_counts([0, 0, 0], np.empty(0, np.int32))
  a = vs.unique(a, vs.INNER_AXIS)
  assert len(a) == 3 and a.dtype == np.int32
  assert np.array_equal(a.counts, [0,0,0]) and a.dsize == 0

  a = vs.from_counts([], np.empty(0, np.int64))
  a = vs.unique(a, vs.INNER_AXIS)
  assert len(a) == 0 and a.dtype == np.int64


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

## SPECS
##
## Class name is VStrideArray
## repr displays vsarray
## constructeur is array
## module name vstride, aliased as vs in maia / doc
##  keep axis = vs.INNER / vs.OUTER 
## 
## enlever les valeurs non initialisés dans le reduce

