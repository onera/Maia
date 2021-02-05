import numpy as np
import Converter.Internal as I

from maia.sids  import sids
from maia.utils import py_utils

def collect_distributed_pl(dist_zone, type_paths, filter_loc=None):
  """
  Search and collect all the pointList values found under the given
  types pathes.
  If a 1d PR is used, it is converted to a contiguous
  pointlist using the distribution node.
  If filter_loc list is not None, select only the pointLists of given
  GridLocation.
  """
  point_lists = []
  for type_path in type_paths:
    for node in py_utils.getNodesFromTypePath(dist_zone, type_path):
      if filter_loc is None or sids.GridLocation(node) in filter_loc:
        pl_n = I.getNodeFromName1(node, 'PointList')
        pr_n = I.getNodeFromName1(node, 'PointRange')
        if pl_n is not None:
          point_lists.append(I.getValue(pl_n))
        elif pr_n is not None and I.getValue(pr_n).shape[0] == 1:
          pr = I.getValue(pr_n)
          distrib_n = I.getNodeFromPath(node, ':CGNS#Distribution/Index')
          distrib   = I.getValue(distrib_n)
          point_lists.append(np.arange(pr[0,0]+distrib[0], pr[0,0]+distrib[1], dtype=pr.dtype))
        # else:
          # point_lists.append(np.empty((1,0), dtype=np.int32, order='F'))
  return point_lists

