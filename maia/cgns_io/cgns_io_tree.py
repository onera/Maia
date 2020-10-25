import Converter.Internal as     I
import Converter.PyTree   as     C
from typing import Dict

def load_tree_from_filter(filename, dist_tree, comm, hdf_filter):
  """
  """
  print("load_tree_from_filter")
  # hdf_filter_with_dim  = dict(filter(lambda x: isinstance(x[1], list), hdf_filter.items() ))
  # hdf_filter_with_func = dict(filter(lambda x: not isinstance(x[1], list), hdf_filter.items() ))

  hdf_filter_with_dim  = {key: value for (key, value) in hdf_filter.items() if isinstance(value, list)}
  hdf_filter_with_func = {key: value for (key, value) in hdf_filter.items() if not isinstance(value, list)}

  partial_dict_load = C.convertFile2PartialPyTreeFromPath(filename, hdf_filter_with_dim, comm)

  print("hdf_filter_with_dim :: ")
  for key, val in hdf_filter_with_dim.items():
    print(key, val)

  print("hdf_filter_with_func :: ")
  for key, val in hdf_filter_with_func.items():
    print(key, val)

  for path, data in partial_dict_load.items():
    Node = I.getNodeFromPath(dist_tree, path)
    Node[1] = data

  # Update
  next_hdf_filter = dict()
  for key, f in hdf_filter_with_func.items():
    f(next_hdf_filter)

  partial_dict_load = C.convertFile2PartialPyTreeFromPath(filename, next_hdf_filter, comm)

  for path, data in partial_dict_load.items():
    Node = I.getNodeFromPath(dist_tree, path)
    Node[1] = data
