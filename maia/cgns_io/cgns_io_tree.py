import Converter.Internal as     I
import Converter.PyTree   as     C
from typing import Dict
import fnmatch

def update_tree_with_partial_load_dict(dist_tree, partial_dict_load):
  """
  """
  for path, data in partial_dict_load.items():
    Node = I.getNodeFromPath(dist_tree, path)
    Node[1] = data


def filtering_filter(dist_tree, hdf_filter, name_or_type_list, skip=True):
  """
  """
  cur_hdf_filter = dict()

  # print("name_or_type_list::", name_or_type_list)
  n_ancestor = 0
  for skip_type in name_or_type_list:
    ancestors_type = skip_type # .split("/")
    n_ancestor     = len(ancestors_type)
    # print("ancestors_type::", ancestors_type)

  # > We filter the hdf_filter to keep only the unskip data
  for path, data in hdf_filter.items():
    split_path = path.split("/")
    # print(split_path)
    # > We should remove from dictionnary all entry
    first_path = ''
    first_n = 0
    for idx in range(len(split_path)-n_ancestor):
      first_path += "/"+split_path[idx]
      first_n    += 1
    prev_node = I.getNodeFromPath(dist_tree, first_path[1:]) # Remove //

    # Now we need to skip if matching with ancestors type
    ancestors_name = []
    ancestors_type = []
    for idx in range(first_n, len(split_path)):
      next_name = split_path[idx]
      next_node = I.getNodeFromName1(prev_node, next_name)
      ancestors_name.append(next_node[0])
      ancestors_type.append(next_node[3])
      prev_node = next_node

    # print("ancestors_type::", ancestors_type)
    # print("ancestors_name::", ancestors_name)

    keep_path = skip
    for skip_type in name_or_type_list:
      n_ancestor = len(skip_type)
      n_match_name_or_type = 0
      for idx in reversed(range(n_ancestor)):
        # print(f"idx::{idx} with skip_type={skip_type[idx]} compare to {ancestors_name[idx]} and {ancestors_type[idx]} ")
        # print("fnmatch::", fnmatch.fnmatch(ancestors_name[idx], skip_type[idx]))
        if( fnmatch.fnmatch(ancestors_name[idx], skip_type[idx])):
          n_match_name_or_type += 1
        elif( (skip_type[idx] == ancestors_name[idx] ) or
              (skip_type[idx] == ancestors_type[idx] )):
          n_match_name_or_type += 1
      if(n_match_name_or_type == len(skip_type)):
        keep_path = not skip
      # print("n_match_name_or_type::", n_match_name_or_type, "/", n_ancestor)
      # print("******************************", path, keep_path)

    if(keep_path):
      cur_hdf_filter[path] = data

  return cur_hdf_filter


def load_tree_from_filter(filename, dist_tree, comm, hdf_filter):
  """
  """
  # print("load_tree_from_filter")
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


def save_tree_from_filter(filename, dist_tree, comm, hdf_filter):
  """
  """
  # print("load_tree_from_filter")
  hdf_filter_with_dim  = {key: value for (key, value) in hdf_filter.items() if isinstance(value, list)}
  hdf_filter_with_func = {key: value for (key, value) in hdf_filter.items() if not isinstance(value, list)}

  next_hdf_filter = dict()
  for key, f in hdf_filter_with_func.items():
    f(hdf_filter_with_dim)

  # print("**********************")
  # for key, val in hdf_filter_with_dim.items():
  #   print(key, val)
  # print("**********************")


  C.convertPyTree2FilePartial(dist_tree, filename, comm, hdf_filter_with_dim, ParallelHDF=True)
