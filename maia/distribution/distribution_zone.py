import Converter.Internal as I
import maia.sids.sids as SIDS

from .distribution_function                 import create_distribution_node
from .distribution_elements                 import compute_elements_distribution
from .distribution_zone_subregion           import compute_zone_subregion_distribution
from .distribution_bc_and_grid_connectivity import compute_distribution_bc, compute_distribution_grid_connectivity


def compute_zone_distribution(zone, comm):
  """
  """
  n_vtx  = SIDS.zone_n_vtx (zone)
  n_cell = SIDS.zone_n_cell(zone)

  distrib_vtx  = create_distribution_node(n_vtx  , comm, 'distribution_vtx' , zone)
  distrib_cell = create_distribution_node(n_cell , comm, 'distribution_cell', zone)

  compute_elements_distribution(zone, comm)

  for zone_subregion in I.getNodesFromType1(zone, 'ZoneSubRegion_t'):
    compute_zone_subregion(zone_subregion, comm)

  for zone_bc in I.getNodesFromType1(zone, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zone_bc, 'BC_t'):
      compute_distribution_bc(bc, comm) # Caution manage vtx/face - Caution BCDataSet can be Vertex

  for zone_gc in I.getNodesFromType1(zone, 'ZoneGridConnectivity_t'):
    gcs = I.getNodesFromType1(zone_gc, 'GridConnectivity_t') + I.getNodesFromType1(zone_gc, 'GridConnectivity1to1_t')
    for gc in gcs:
      compute_distribution_grid_connectivity(gc, comm)


