import Converter.Internal as     I
import Converter.PyTree   as     C
from typing import Dict

def load_tree_from_filter(filename, dist_tree, comm, hdf_filter):
  """
  """
  print("load_tree_from_filter")
  partial_dict_load = C.convertFile2PartialPyTreeFromPath(filename, hdf_filter, comm)

  for path, data in partial_dict_load.items():
    Node = I.getNodeFromPath(dist_tree, path)
    Node[1] = data
