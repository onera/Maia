import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.io.hdf import utils

def test_pl_or_pr_size():
  bc_pr = PT.new_BC(name='BC', point_range=[[16,25], [5,5], [1,5]], type='BCFarfield')
  size = utils.pl_or_pr_size(bc_pr)
  assert (size == [10, 1, 5]).all()

  bc_ud = PT.new_BC(name='BC', point_list=[[5,10,15,20,25,30]], type='BCFarfield')
  MT.newDistribution({'Index' : [1,6,6]}, parent=bc_ud)
  size = utils.pl_or_pr_size(bc_ud)
  assert (size == [1, 6]).all()

def test_apply_dataspace_to_arrays():
  hdf_filter = dict()
  node = PT.new_node('Parent', 'UserDefined_t')
  for child in ['child1', 'child2', 'child3']:
    PT.new_DataArray(child, None, parent=node)
  utils.apply_dataspace_to_arrays(node, "path/to/Parent", "my_data_space", hdf_filter)
  for child in ['child1', 'child2', 'child3']:
     assert hdf_filter['path/to/Parent/' + child] == "my_data_space"

def test_apply_dataspace_to_pointlist():
  hdf_filter = dict()
  bc = PT.new_node('wall', 'BC_t', children=[PT.new_PointList('PointList')])
  gc = PT.new_node('wall', 'GridConnectivity_t', children=[PT.new_PointList('PointList'),
                                              PT.new_PointList('PointListDonor')])
  utils.apply_dataspace_to_pointlist(bc, "path/to/bc", "my_data_space", hdf_filter)
  utils.apply_dataspace_to_pointlist(gc, "path/to/gc", "my_data_space", hdf_filter)
  assert hdf_filter["path/to/bc/PointList"] == "my_data_space"
  assert hdf_filter["path/to/gc/PointList"] == "my_data_space"
  assert hdf_filter["path/to/gc/PointListDonor"] == "my_data_space"
