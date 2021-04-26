import Converter.Internal as     I
import Converter.PyTree   as     C
from   maia.distribution.distribution_tree import add_distribution_info, clean_distribution_info
from   .load_collective_size_tree          import load_collective_size_tree
from   .hdf_filter.tree                    import create_tree_hdf_filter

def update_tree_with_partial_load_dict(dist_tree, partial_dict_load):
  """
  """
  for path, data in partial_dict_load.items():
    Node = I.getNodeFromPath(dist_tree, path)
    Node[1] = data


def load_tree_from_filter(filename, dist_tree, comm, hdf_filter):
  """
  """
  # print("load_tree_from_filter")
  hdf_filter_with_dim  = {key: value for (key, value) in hdf_filter.items() \
      if isinstance(value, (list, tuple))}

  partial_dict_load = C.convertFile2PartialPyTreeFromPath(filename, hdf_filter_with_dim, comm)
  update_tree_with_partial_load_dict(dist_tree, partial_dict_load)

  # > Match with callable
  hdf_filter_with_func = {key: value for (key, value) in hdf_filter.items() \
      if not isinstance(value, (list, tuple))}
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
    hdf_filter_with_func = {key: value for (key, value) in next_hdf_filter.items() \
        if not isinstance(value, (list, tuple))}

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

  #Dont save distribution info, but work on a copy to keep it for further use
  saving_dist_tree = I.copyRef(dist_tree)
  clean_distribution_info(saving_dist_tree)

  C.convertPyTree2FilePartial(saving_dist_tree, filename, comm, hdf_filter_with_dim, ParallelHDF=True)

def file_to_dist_tree(filename, comm, distribution_policy='uniform'):
  """
  Distributed load of filename. Return a dist_tree.
  """
  dist_tree = load_collective_size_tree(filename, comm)
  add_distribution_info(dist_tree, comm, distribution_policy)

  hdf_filter = dict()
  create_tree_hdf_filter(dist_tree, hdf_filter)

  load_tree_from_filter(filename, dist_tree, comm, hdf_filter)

  return dist_tree

def dist_tree_to_file(dist_tree, filename, comm, hdf_filter = None):
  """
  Distributed write of cgns_tree into filename.
  """
  if hdf_filter is None:
    hdf_filter = dict()
    create_tree_hdf_filter(dist_tree, hdf_filter)
  save_tree_from_filter(filename, dist_tree, comm, hdf_filter)
