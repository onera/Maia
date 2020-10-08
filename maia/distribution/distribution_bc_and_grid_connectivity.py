import numpy              as NPY
import Converter.Internal as I

from .distribution_function    import create_distribution_node
from .distribution_bc_dataset  import compute_distribution_bc_dataset

def compute_distribution_bc(bc, comm):
  """
  """
  pr_n = I.getNodeFromName1(bc, 'PointRange')
  pl_n = I.getNodeFromName1(bc, 'PointList')

  if(pr_n):
    pr_size = 1
    print("*"*10, pr_n[1])
    for idx in xrange(len(pr_n[1])):
      print(pr_n[1][idx][0], pr_n[1][idx][1])
    raise NotImplemented

  if(pl_n):
    pls_n   = I.getNodeFromName1(bc, 'PointList#Shape')
    pl_size = NPY.prod(pls_n[1])
    create_distribution_node(pl_size, comm, 'distrib_elmt', bc)

  for bcds in I.getNodesFromType1(bc, 'BCDataSet_t'):
    compute_distribution_bc_dataset(bcds, comm)


def compute_distribution_grid_connectivity(join, comm):
  """
  """
  pr_n = I.getNodeFromName1(join, 'PointRange')
  pl_n = I.getNodeFromName1(join, 'PointList')

  if(pr_n):
    raise NotImplemented

  if(pl_n):
    pls_n   = I.getNodeFromName1(join, 'PointList#Shape')
    pl_size = NPY.prod(pls_n[1])
    create_distribution_node(pl_size, comm, 'distrib_elmt', join)

  # prd_n = I.getNodeFromName1(join, 'PointRangeDonor')
  # pld_n = I.getNodeFromName1(join, 'PointListDonor')

  for bcds in I.getNodesFromType1(join, 'BCDataSet_t'):
    compute_distribution_bc_dataset(bcds, comm)
