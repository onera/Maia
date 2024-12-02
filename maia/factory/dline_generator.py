import maia
import maia.pytree as PT
from   maia.utils  import par_utils
from   maia        import npy_pdm_gnum_dtype as pdm_gnum_dtype

import numpy as np

def generate_dist_line(n_point, start, end, comm):
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
  phy_dim     = len(start)

  dist_tree = PT.new_CGNSTree()
  dist_base = PT.new_CGNSBase(cell_dim=1, phy_dim=phy_dim, parent=dist_tree)

  # > Vertices
  length      = np.array(end)-np.array(start)
  delta       = length/(n_point-1)
  vtx_distrib = par_utils.uniform_distribution(n_point, comm)
  dn_vtx      = vtx_distrib[1]-vtx_distrib[0]
  coords = {}
  for i in range(phy_dim):
    key = 'Coordinate' + 'XYZ'[i]
    coords[key] = np.arange(vtx_distrib[0], vtx_distrib[1], dtype=float)*delta[i]+start[i]
  
  # > Edge connectivity
  bar_distrib = par_utils.uniform_distribution(n_point-1, comm)
  dn_bar      = bar_distrib[1]-bar_distrib[0]
  a = np.arange(bar_distrib[0]+1, bar_distrib[1]+1, dtype=pdm_gnum_dtype)
  b = np.arange(bar_distrib[0]+2, bar_distrib[1]+2, dtype=pdm_gnum_dtype)
  ec = np.zeros(2*dn_bar, dtype=pdm_gnum_dtype)
  ec[0::2] = a
  ec[1::2] = b

  # > Create zone
  dist_zone = PT.new_Zone(f'Line', type='Unstructured', size=np.array([[n_point,n_point-1,0]], dtype=pdm_gnum_dtype), parent=dist_base)
  PT.new_GridCoordinates(fields=coords, parent=dist_zone)
  elmt_n = PT.new_Elements('BAR_2', type='BAR_2', erange=[1,n_point-1], econn=ec, parent=dist_zone)
  PT.maia.newDistribution({"Element":bar_distrib}, parent=elmt_n)
  PT.maia.newDistribution({"Vertex":vtx_distrib, "Cell":bar_distrib}, parent=dist_zone)

  return dist_tree