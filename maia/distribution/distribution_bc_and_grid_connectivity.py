import numpy              as NPY
import Converter.Internal as I

import maia.sids.sids as SIDS
from .distribution_function    import create_distribution_node
from .distribution_bc_dataset  import compute_distribution_bc_dataset

def compute_distribution_bc(bc, comm):
  """
  """
  pr_n = I.getNodeFromName1(bc, 'PointRange')
  pl_n = I.getNodeFromName1(bc, 'PointList')

  if(pr_n):
    pr_lenght = SIDS.point_range_lenght(pr_n)
    create_distribution_node(pr_lenght, comm, 'Distribution', bc)

  if(pl_n):
    pls_n   = I.getNodeFromName1(bc, 'PointList#Size')
    pl_size = NPY.prod(pls_n[1])
    create_distribution_node(pl_size, comm, 'Distribution', bc)

  for bcds in I.getNodesFromType1(bc, 'BCDataSet_t'):
    compute_distribution_bc_dataset(bcds, comm)


def compute_distribution_grid_connectivity(join, comm):
  """
  """
  pr_n = I.getNodeFromName1(join, 'PointRange')
  pl_n = I.getNodeFromName1(join, 'PointList')

  if(pr_n):
    pr_lenght = SIDS.point_range_lenght(pr_n)
    create_distribution_node(pr_lenght, comm, 'Distribution', join)

  if(pl_n):
    pls_n   = I.getNodeFromName1(join, 'PointList#Size')
    pl_size = NPY.prod(pls_n[1])
    create_distribution_node(pl_size, comm, 'Distribution', join)

  # prd_n = I.getNodeFromName1(join, 'PointRangeDonor')
  # pld_n = I.getNodeFromName1(join, 'PointListDonor')

