import numpy              as NPY
import Converter.Internal as I

from .distribution_function    import create_distribution_node

def compute_distribution_bc_dataset(bcds, comm):
  """
  """
  pr_n = I.getNodeFromName1(bcds, 'PointRange')
  pl_n = I.getNodeFromName1(bcds, 'PointList')

  if(pr_n):
    raise NotImplemented

  if(pl_n):
    pls_n   = I.getNodeFromName1(bc, 'PointList#Shape')
    pl_size = NPY.prod(pls_n[1])
    create_distribution_node(pl_size, comm, 'distrib_elmt', bcds)
