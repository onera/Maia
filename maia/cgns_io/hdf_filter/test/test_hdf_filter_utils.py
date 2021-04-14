import Converter.Internal as I
from maia.cgns_io.hdf_filter import utils
import maia.sids.Internal_ext as IE

def test_pl_or_pr_size():
  bc_pr = I.newBC(name='BC', pointRange=[[16,25], [5,5], [1,5]], btype='BCFarfield')
  size = utils.pl_or_pr_size(bc_pr)
  assert (size == [10, 1, 5]).all()

  bc_pl = I.newBC(name='BC', pointList=[[5,10,15,20,25,30]], btype='BCFarfield')
  I.newIndexArray('PointList#Size', [1, 6], parent=bc_pl)
  size = utils.pl_or_pr_size(bc_pl)
  assert (size == [1, 6]).all()

  bc_ud = I.newBC(name='BC', pointList=[[5,10,15,20,25,30]], btype='BCFarfield')
  IE.newDistribution({'Index' : [1,6,6]}, parent=bc_ud)
  size = utils.pl_or_pr_size(bc_ud)
  assert (size == [1, 6]).all()

def test_apply_dataspace_to_arrays():
  hdf_filter = dict()
  node = I.createNode('Parent', 'AnyType_t')
  for child in ['child1', 'child2', 'child3']:
    I.newDataArray(child, parent=node)
  utils.apply_dataspace_to_arrays(node, "path/to/Parent", "my_data_space", hdf_filter)
  for child in ['child1', 'child2', 'child3']:
     assert hdf_filter['path/to/Parent/' + child] == "my_data_space"

def test_apply_dataspace_to_pointlist():
  hdf_filter = dict()
  bc = I.createNode('wall', 'BC_t', children=[I.newIndexArray('PointList')])
  gc = I.createNode('wall', 'GC_t', children=[I.newIndexArray('PointList'),
                                              I.newIndexArray('PointListDonor')])
  utils.apply_dataspace_to_pointlist(bc, "path/to/bc", "my_data_space", hdf_filter)
  utils.apply_dataspace_to_pointlist(gc, "path/to/gc", "my_data_space", hdf_filter)
  assert hdf_filter["path/to/bc/PointList"] == "my_data_space"
  assert hdf_filter["path/to/gc/PointList"] == "my_data_space"
  assert hdf_filter["path/to/gc/PointListDonor"] == "my_data_space"
