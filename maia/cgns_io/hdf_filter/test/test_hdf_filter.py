import Converter.Internal as I
from maia.cgns_io.hdf_filter import utils, range_to_slab
import numpy              as np

def test_pl_or_pr_size_pr():
  bc = I.newBC(name='BC', pointRange=[[16,25], [5,5], [1,5]], btype='BCFarfield')
  size = utils.pl_or_pr_size(bc)
  assert np.all(size == [10, 1, 5])

def test_pl_or_pr_size_pl():
  bc = I.newBC(name='BC', pointList=[[5,10,15,20,25,30]], btype='BCFarfield')
  I.newIndexArray('PointList#Size', [1, 6], parent=bc)
  size = utils.pl_or_pr_size(bc)
  assert np.all(size == [1, 6])

def test_pl_or_pr_size_pl_with_distri():
  bc = I.newBC(name='BC', pointList=[[5,10,15,20,25,30]], btype='BCFarfield')
  ud_node = I.createChild(bc, ':CGNS#Distribution', 'UserDefinedData_t')
  I.newDataArray('Distribution', [1, 6, 6], parent=ud_node)
  size = utils.pl_or_pr_size(bc)
  assert np.all(size == [1, 6])

def test_compute_slabs():
  slabs = range_to_slab.compute_slabs([5,5,5], [19, 62])
  assert len(slabs) == 5
  assert slabs[0] == [[4, 5], [3, 4], [0, 1]]
  assert slabs[1] == [[0, 5], [4, 5], [0, 1]]
  assert slabs[2] == [[0, 5], [0, 5], [1, 2]]
  assert slabs[3] == [[0, 5], [0, 2], [2, 3]]
  assert slabs[4] == [[0, 2], [2, 3], [2, 3]]

def test_compute_slabs_combine():
  slabs = range_to_slab.compute_slabs([7,3,9], [105, 161])
  assert len(slabs) == 2
  assert slabs[0] == [[0, 7], [0, 3], [5, 7]]
  assert slabs[1] == [[0, 7], [0, 2], [7, 8]]
