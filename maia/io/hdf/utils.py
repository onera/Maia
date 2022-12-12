import maia.pytree        as PT

def apply_dataspace_to_arrays(node, node_path, data_space, hdf_filter):
  """
  Fill the hdf_filter with the specified data_space for all the DataArray_t nodes
  below the parent node node
  """
  for data_array in PT.iter_children_from_label(node, 'DataArray_t'):
    path = node_path+"/"+data_array[0]
    hdf_filter[path] = data_space

