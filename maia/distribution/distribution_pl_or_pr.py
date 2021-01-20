import Converter.Internal as I
import numpy as NPY

import maia.sids.sids as SIDS
from .distribution_function import create_distribution_node

def compute_plist_or_prange_distribution(node, comm):
  """
  Compute the distribution for a given node using its PointList or PointRange child
  If a PointRange node is found, the total lenght is getted from the product
  of the differences for each direction (cgns convention (cgns convention : 
  first and last are included).
  If a PointList node is found, the total lenght is getted from the product of
  PointList#Size arrays, which store the size of the PL in each direction.
  """

  pr_n = I.getNodeFromName1(node, 'PointRange')
  pl_n = I.getNodeFromName1(node, 'PointList')

  if(pr_n):
    pr_lenght = SIDS.point_range_n_elt(pr_n)
    create_distribution_node(pr_lenght, comm, 'Distribution', node)

  if(pl_n):
    pls_n   = I.getNodeFromName1(node, 'PointList#Size')
    pl_size = NPY.prod(pls_n[1])
    create_distribution_node(pl_size, comm, 'Distribution', node)





