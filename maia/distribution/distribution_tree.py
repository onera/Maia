from   mpi4py             import MPI
import Converter.Internal as     I
import Converter.PyTree   as     C

from . import distribution_base as DBASE
from . import distribution_zone import compute_zone_distribution


#
# uniform = compute_proc_indices

# def master_of_each_node():
#   sub_comm = master_of_each_node(comm)
#   compute_proc_indices(sub_comm)
#   #
# Heterogene : Element / BC / Join


def add_distribution_info(dist_tree, comm, distribution_policy='uniform'):
  """
  """
  for base in I.getNodesFromType1(dist_tree, 'CGNSBase_t'):
  # DFAM.compute_family_distribution(dist_tree)
    for zone in I.getNodesFromType1(base, 'Zone_t'):
      compute_zone_distribution(zone, comm)


