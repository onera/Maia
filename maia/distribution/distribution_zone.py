import Converter.Internal as I
import maia.sids.sids as SIDS

from .distribution_function                 import create_distribution_node
from .distribution_elements                 import compute_elements_distribution
from .distribution_pl_or_pr                 import compute_plist_or_prange_distribution


def compute_zone_distribution(zone, comm):
  """
  """
  n_vtx  = SIDS.zone_n_vtx (zone)
  n_cell = SIDS.zone_n_cell(zone)

  distrib_vtx  = create_distribution_node(n_vtx  , comm, 'Vertex', zone)
  distrib_cell = create_distribution_node(n_cell , comm, 'Cell'  , zone)

  compute_elements_distribution(zone, comm)

  for zone_subregion in I.getNodesFromType1(zone, 'ZoneSubRegion_t'):
    compute_plist_or_prange_distribution(zone_subregion, comm)

  for zone_bc in I.getNodesFromType1(zone, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zone_bc, 'BC_t'):
      compute_plist_or_prange_distribution(bc, comm)
      for bcds in I.getNodesFromType1(bc, 'BCDataSet_t'):
        compute_plist_or_prange_distribution(bcds, comm)

  for zone_gc in I.getNodesFromType1(zone, 'ZoneGridConnectivity_t'):
    gcs = I.getNodesFromType1(zone_gc, 'GridConnectivity_t') + I.getNodesFromType1(zone_gc, 'GridConnectivity1to1_t')
    for gc in gcs:
      compute_plist_or_prange_distribution(gc, comm)


