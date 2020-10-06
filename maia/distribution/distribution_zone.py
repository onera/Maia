import Converter.PyTree   as C
import Converter.Internal as I


def compute_distribution_zone_face(zone_tree, comm):
  """
  For one zone setup distribution for faces if possible
  """
  # compute_proc_indexes()






def compute_distribution_zone_cell(zone_tree, comm):
  """
  For one zone setup distribution for faces if possible
  """
  # compute_proc_indexes()

  ncell = UTL.getZoneNbCell(zone_tree)

  distrib_cell = NPY.zeros(3, order='C', dtype='int32')
  UTL.compute_proc_indexes(distrib_cell, ncell, i_active, n_active)
