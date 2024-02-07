from mpi4py import MPI
import numpy as np

from cmaia import dist_algo as cdist_algo
from maia.algo.apply_function_to_nodes import apply_to_zones

import maia
import maia.pytree as PT
from maia.utils import require_cpp20

@require_cpp20
def ngons_to_elements(t,comm):
  """
  Transform a polyedric (NGon) based connectivity into a standard nodal
  connectivity.
  
  Tree is modified in place : polyedric element, which are supposed to describe
  only standard elements (tris, quads, tets, pyras, prisms and hexa)
  are removed from the zones and Pointlist (under the BC_t nodes) are updated.

  Warning: 
    This function has not been parallelized yet. Tree is internally gathered
    to a single process, which can cause memory or performance issues on large cases.

  Args:
    disttree   (CGNSTree): Tree with connectivity described by NGons
    comm       (`MPIComm`) : MPI communicator

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #ngons_to_elements@start
        :end-before: #ngons_to_elements@end
        :dedent: 2
  """
  # Function require NFACE + NGON with PE
  for zone in PT.get_all_Zone_t(t):
    if not PT.Zone.has_nface_elements(zone):
      maia.algo.pe_to_nface(zone, comm)
    ng = PT.Zone.NGonNode(zone)
    if PT.get_child_from_name(ng, 'ParentElements') is None:
      maia.algo.nface_to_pe(zone, comm)

  if comm.Get_size() > 1:
    maia.algo.dist.redistribute_tree(t, 'gather.0', comm)
  
  zone_to_elts = {}
  if comm.Get_rank() == 0:
    apply_to_zones(cdist_algo.convert_zone_to_std_elements, t)

    # This is to broadcast new element names to other ranks
    for zone_path in PT.predicates_to_paths(t, 'CGNSBase_t/Zone_t'):
      zone = PT.get_node_from_path(t, zone_path)
      elts = PT.get_children_from_label(zone, 'Elements_t')
      zone_to_elts[zone_path] = []
      for elt in elts:
        void_elt = PT.shallow_copy(elt)
        cnt = PT.get_child_from_name(void_elt, 'ElementConnectivity')
        cnt[1] = np.empty_like(cnt[1], shape=(0,))
        zone_to_elts[zone_path].append(void_elt)

  if comm.Get_size() > 1:
    zone_to_elts = comm.bcast(zone_to_elts, root=0)
    if comm.Get_rank() != 0:
      for zone_path, elts in zone_to_elts.items():
        zone = PT.get_node_from_path(t, zone_path)
        PT.rm_nodes_from_label(t, 'Elements_t')
        for elt in elts:
          distri = PT.maia.getDistribution(elt, 'Element')[1]
          distri[:] = distri[2]
          PT.add_child(zone, elt)

    maia.algo.dist.redistribute_tree(t, 'uniform', comm)

