import Converter.Internal as     I
import Converter.PyTree   as     C
from typing import Dict

def update_tree_with_partial_load_dict(dist_tree, partial_dict_load):
  """
  """
  for path, data in partial_dict_load.items():
    Node = I.getNodeFromPath(dist_tree, path)
    Node[1] = data


def filtering_filter(dist_tree, hdf_filter, skip_type_ancestors):
  """
  """
  cur_hdf_filter = dict()

  for i, st in enumerate(skip_type_ancestors):
    if(st[-1] == "/"): skip_type_ancestors[i] = st[:len(st)-1]
    if(st[0]  == "/"): skip_type_ancestors[i] = st[1:]

  print("skip_type_ancestors::", skip_type_ancestors)
  n_ancestor = 0
  for skip_type in skip_type_ancestors:
    ancestors_type = skip_type.split("/")
    n_ancestor     = len(ancestors_type)
    print("ancestors_type::", ancestors_type)

  # > We filter the hdf_filter to keep only the unskip data
  for path, data in hdf_filter.items():
    split_path = path.split("/")
    print(split_path)
    # > We should remove from dictionnary all entry
    first_path = ''
    first_n = 0
    for idx in range(len(split_path)-n_ancestor-1):
      first_path += "/"+split_path[idx]
      first_n    += 1
    prev_node = I.getNodeFromPath(dist_tree, first_path[1:])

    # Now we need to skip if matching with ancestors type
    ancestors_type = ''
    for idx in range(first_n, len(split_path)-1):
      next_name = split_path[idx]
      next_node = I.getNodeFromName1(prev_node, next_name)
      if(idx == first_n):
        ancestors_type += next_node[3]
      else:
        ancestors_type += "/"+next_node[3]
      # print("next_name:: ", next_name)
      prev_node = next_node

    print("ancestors_type::", ancestors_type)

    if(ancestors_type not in skip_type_ancestors):
      cur_hdf_filter[path] = data

  return cur_hdf_filter


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
