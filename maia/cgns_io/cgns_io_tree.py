import Converter.Internal as     I
import Converter.PyTree   as     C
from typing import Dict

def update_tree_with_partial_load_dict(dist_tree, partial_dict_load):
  """
  """
  for path, data in partial_dict_load.items():
    Node = I.getNodeFromPath(dist_tree, path)
    Node[1] = data


def load_tree_from_filter(filename, dist_tree, comm, hdf_filter):
  """
  """
  print("load_tree_from_filter")
  hdf_filter_with_dim  = {key: value for (key, value) in hdf_filter.items() if isinstance(value, list)}

  partial_dict_load = C.convertFile2PartialPyTreeFromPath(filename, hdf_filter_with_dim, comm)
  update_tree_with_partial_load_dict(dist_tree, partial_dict_load)

  hdf_filter_with_func = {key: value for (key, value) in hdf_filter.items() if not isinstance(value, list)}
  unlock_at_least_one = True
  while(len(hdf_filter_with_func) > 0 and unlock_at_least_one ):
    # Update if you can
    next_hdf_filter = dict()
    unlock_at_least_one = False
    for key, f in hdf_filter_with_func.items():
      try:
        f(next_hdf_filter)
        unlock_at_least_one = True
      except RuntimeError: # Not ready yet
        pass
    partial_dict_load = C.convertFile2PartialPyTreeFromPath(filename, next_hdf_filter, comm)

    update_tree_with_partial_load_dict(dist_tree, partial_dict_load)
    hdf_filter_with_func = {key: value for (key, value) in next_hdf_filter.items() if not isinstance(value, list)}

  if(unlock_at_least_one is False):
    raise RuntimeError("Something strange in the loading process")
