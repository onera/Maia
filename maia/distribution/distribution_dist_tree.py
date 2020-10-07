from   mpi4py             import MPI
import Converter.Internal as     I
import Converter.PyTree   as     C

from . import distribution_base as DBASE
from . import distribution_zone as DZONE


#
# uniform = compute_proc_indices

# def master_of_each_node():
#   sub_comm = master_of_each_node(comm)
#   compute_proc_indices(sub_comm)
#   #
# Heterogene : Element / BC / Join


def enrich_with_dist_info(dist_tree, distribution_policy='uniform', comm):
  """
  """

  # f_distrib_zone = distrib_function["Zone_t"]
  # f_distrib_bc   = distrib_function["BC_t"]

  # tuple(function:) [Zone_t] --> function
  # distrib = function(n_elements, comm)

  # DFAM.compute_family_distribution(dist_tree)
  for zone in I.getZones(dist_tree):
    DZONE.compute_zone_distribution(zone)


