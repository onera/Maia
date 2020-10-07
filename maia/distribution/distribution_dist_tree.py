from   mpi4py             import MPI
import Converter.Internal as     I
import Converter.PyTree   as     C

from . import distribution_base as DBASE
from . import distribution_zone as DZONE


def enrich_with_dist_info(dist_tree, distribution_policy='uniform', comm):
  """
  """

  # DFAM.compute_family_distribution(dist_tree)
  for zone in I.getZones(dist_tree):
    DZONE.compute_zone_distribution(zone)


