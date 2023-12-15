import maia.pytree        as PT

from maia.io.hdf import utils

def test_apply_dataspace_to_arrays():
  hdf_filter = dict()
  node = PT.new_node('Parent', 'UserDefinedData_t')
  for child in ['child1', 'child2', 'child3']:
    PT.new_DataArray(child, None, parent=node)
  utils.apply_dataspace_to_arrays(node, "path/to/Parent", "my_data_space", hdf_filter)
  for child in ['child1', 'child2', 'child3']:
     assert hdf_filter['path/to/Parent/' + child] == "my_data_space"

