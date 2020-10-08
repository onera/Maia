import numpy              as NPY
import Converter.Internal as I

from .distribution_function                 import create_distribution_node

def compute_distribution_bc(bc, comm):
  """
  """
  pr_n = I.getNodeFromName1(bc, 'PointRange')
  pl_n = I.getNodeFromName1(bc, 'PointList')

  if(pr_n):
    raise NotImplemented

  if(pl_n):
    pls_n   = I.getNodeFromName1(zone_subregion, 'PointList#Shape')
    pl_size = NPY.prod(pls_n[1])
    create_distribution_node(pl_size, comm, 'distrib_elmt', bc)


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
