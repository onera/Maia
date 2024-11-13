import maia
import maia.pytree as PT
from   maia.utils  import par_utils

import numpy as np

def generate_dist_line(start, end, n_point, comm):
  """
  Generate distributed straight line between start and end coordinates
  discretized with n_points.

  Args:
    start   (float array) : Line start coordinates
    end     (float array) : Line end coordinates
    n_point (int)         : Point number in line
    comm    (MPIComm)     : MPI communicator
  Returns:
    CGNSTree: Line mesh (distributed)

  Example:
    .. literalinclude:: snippets/test_factory.py
      :start-after: #generate_dist_line@start
      :end-before:  #generate_dist_line@end
      :dedent: 2
  """
  dist_tree = PT.new_CGNSTree()
  dist_base = PT.new_CGNSBase(parent=dist_tree)

  # > Vertices
  length      = end-start
  delta       = length/(n_point-1)
  vtx_distrib = par_utils.uniform_distribution(n_point, comm)
  dn_vtx      = vtx_distrib[1]-vtx_distrib[0]
  x = np.arange(vtx_distrib[0], vtx_distrib[1], dtype=np.float64)*delta[0]+start[0]
  y = np.arange(vtx_distrib[0], vtx_distrib[1], dtype=np.float64)*delta[1]+start[1]
  z = np.arange(vtx_distrib[0], vtx_distrib[1], dtype=np.float64)*delta[2]+start[2]
  
  # > Edge connectivity
  bar_distrib = par_utils.uniform_distribution(n_point-1, comm)
  dn_bar      = bar_distrib[1]-bar_distrib[0]
  a = np.arange(bar_distrib[0]+1, bar_distrib[1]+1, dtype=np.int32)
  b = np.arange(bar_distrib[0]+2, bar_distrib[1]+2, dtype=np.int32)
  ec = np.zeros(2*dn_bar, dtype=np.int32)
  ec[0::2] = a
  ec[1::2] = b

  # > Create zone
  coords={'CoordinateX' : x, 'CoordinateY' : y, 'CoordinateZ' : z}
  dist_zone = PT.new_Zone(f'Line', type='Unstructured', size=[[n_point,n_point-1,0]], parent=dist_base)
  PT.new_GridCoordinates(fields=coords, parent=dist_zone)
  elmt_n = PT.new_Elements('BAR_2', type='BAR_2', erange=[1,n_point-1], econn=ec, parent=dist_zone)
  PT.maia.newDistribution({"Element":bar_distrib}, parent=elmt_n)
  PT.maia.newDistribution({"Vertex":vtx_distrib, "Cell":bar_distrib}, parent=dist_zone)

  return dist_tree