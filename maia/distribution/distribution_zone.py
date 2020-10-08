import Converter.PyTree   as C
import Converter.Internal as I
import maia.utils         as UTL

from   .distribution_function import uniform_distribution


# def compute_distribution_zone_face(zone_tree, comm):
#   """
#   For one zone setup distribution for faces if possible
#   """
#   # compute_proc_indices()

# def compute_distribution_zone_cell(zone_tree, comm):
#   """
#   For one zone setup distribution for faces if possible
#   """
#   # compute_proc_indices()
#   ncell = UTL.get_zone_nb_cell(zone_tree)
#   distrib_cell = NPY.zeros(3, order='C', dtype='int32')
#   UTL.compute_proc_indices(distrib_cell, ncell, i_active, n_active)


def compute_zone_distribution(zone, comm):
  """
  """
  nvtx  = UTL.get_zone_nb_vtx (zone)
  ncell = UTL.get_zone_nb_cell(zone)

  print("nvtx ::", nvtx)
  print("ncell::", ncell)

  distrib_vtx  = uniform_distribution(nvtx , comm)
  distrib_cell = uniform_distribution(ncell, comm)

  # FlowSolution : Avec ghost cells
  #    --> [ --------- , rind ]

  # > TODO put in tree
  compute_elements_distribution(zone, comm)

  for zone_subregion in I.getNodesFromType1(zone, 'ZoneSubRegion_t'):
    compute_zone_subregion(zone_subregion)

  for zone_bc in I.getNodesFromType1(zone, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zone, 'BC_t'):
      compute_distribution_bc(bc) # Caution manage vtx/face - Caution BCDataSet can be Vertex

  for zone_gc in I.getNodesFromType1(zone, 'ZoneGridConnectivity_t'):
    gcs = I.getNodesFromType1(zone_gc, 'GridConnectivity_t') + I.getNodesFromType1(zone_gc, 'GridConnectivity1to1_t')
    for gc in gcs:
      compute_distribution_grid_connectivity(gc) # Caution manage vtx/face


