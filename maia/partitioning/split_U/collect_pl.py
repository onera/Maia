import numpy as np
import Converter.Internal as I

from maia.utils import py_utils

def collect_distributed_pl(dist_zone, type_path):
  """
  Search all the pointList nodes from bcs and concatenate them using
  concatenate_point_list. If pointlist node is none, a fake pl is created
  from the distribution.
  """
  point_lists = []
  for node in py_utils.getNodesFromTypePath(dist_zone, type_path):
    if I.getNodeFromName1(node, 'PointList') is not None:
      point_lists.append(I.getNodeFromName1(node, 'PointList')[1])
    elif I.getNodeFromName1(node, 'PointRange') is not None:
      pr = I.getNodeFromName1(node, 'PointRange')[1]
      distrib_n = I.getNodeFromPath(node, ':CGNS#Distribution/Index')
      distrib   = I.getValue(distrib_n)
      point_lists.append(np.arange(pr[0,0]+distrib[0], pr[0,0]+distrib[1], dtype=pr.dtype))
    else:
      point_lists.append(np.empty((1,0), dtype=np.int32, order='F'))
  return point_lists
