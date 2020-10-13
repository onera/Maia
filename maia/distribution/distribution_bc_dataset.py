import numpy              as NPY
import Converter.Internal as I

from .distribution_function import create_distribution_node

def compute_distribution_bc_dataset(bcds, comm):
  """
  """
  pr_n = I.getNodeFromName1(bcds, 'PointRange')
  pl_n = I.getNodeFromName1(bcds, 'PointList')

  if(pr_n):
    pr_size = 1
    grid_location_n = I.getNodeFromType1(bcds, 'GridLocation_t')
    if(grid_location_n[1].tostring() == b'Vertex'):
      shift = 0
    elif(grid_location_n[1].tostring() == b'CellCenter'):
      shift = -1
    for idx in range(len(pr_n[1])):
      pr_size *= (pr_n[1][idx][1]-pr_n[1][idx][0]+shift)
    create_distribution_node(pr_size, comm, 'distrib_elmt', bc)

  if(pl_n):
    pls_n   = I.getNodeFromName1(bc, 'PointList#Size')
    pl_size = NPY.prod(pls_n[1])
    create_distribution_node(pl_size, comm, 'distrib_elmt', bcds)
