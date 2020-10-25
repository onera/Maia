import Converter.Internal as I
import numpy as NPY

from .distribution_function import create_distribution_node

def compute_zone_subregion_distribution(zone_subregion, comm):
  """
  """

  pr_n = I.getNodeFromName1(zone_subregion, 'PointRange')
  pl_n = I.getNodeFromName1(zone_subregion, 'PointList')

  if(pr_n):
    raise NotImplemented

  if(pl_n):
    pls_n   = I.getNodeFromName1(zone_subregion, 'PointList#Size')
    pl_size = NPY.prod(pls_n[1])
    create_distribution_node(pl_size, comm, 'Distribution', zone_subregion)





