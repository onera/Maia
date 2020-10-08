from mpi4py import MPI
import logging as LOG

# ------------------------------------------------------------------------
# > Initilise MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------------------------------------------------------
fmt = '%(levelname)s:%(message)s '.format(rank, size)
LOG.basicConfig(filename = '{0}.{1}.log'.format('maia_workflow_log', rank),
                level    = 10,
                format   = fmt,
                filemode = 'w')
# ---------------------------------------------------------

# > Import PyPart
from pypart                 import LazyLoadConfiguration   as LLC
from pypart                 import DistributionBase        as DBA
from pypart                 import DistributionZone        as DZO
# from pypart                 import DistributedLoad         as DLO
from pypart                 import PyPartMulti             as PPM
from pypart                 import TransfertTreeData       as TTD
from pypart                 import SaveTree                as SVT
# import etc.transform as trf

from maia.cgns_io        import lazy_load as LZL
import maia.distribution as     MDI

import Converter.PyTree   as C
import Converter.Internal as I
import numpy              as NPY
import sys

# ------------------------------------------------------------------------
# > Pick a file
inputfile    = '/home/bmaugars/dev/dev-Tools/etc/test/pypart/data/CaseU_C1_Cube.hdf'
inputfile    = '/home/bmaugars/dev/dev-Tools/etc/test/pypart/data/CaseU_C1_Cube_BCDataSet.hdf'
# inputfile    = '/home/bmaugars/dev/dev-Tools/etc/test/pypart/data/CaseS_C1_Cube.hdf'

# ------------------------------------------------------------------------
# > Load only the list of zone and sizes ...
dist_tree = LZL.load_collective_pruned_tree(inputfile, comm)

# > ParaDiGM : dcube_gen() --> A faire

MDI.add_distribution_info(dist_tree, comm, distribution_policy='uniform')

I.printTree(dist_tree)

# > To copy paste in new algorithm
# dzone_to_proc = compute_distribution_of_zones(dist_tree, distribution_policy='uniform', comm)


# > dZoneToWeightedParts --> Proportion de la zone initiale qu'on souhate après partitionnement
# > dLoadingProcs        --> Proportion de la zone initiale avant le partitionnement (vision block)

#
# > ... and this is suffisent to predict your partitions sizes
dZoneToWeightedParts = DBA.computePartitioningWeights(dist_tree, comm)

print(dZoneToWeightedParts)

dLoadingProcs = dict()
for zone in I.getZones(dist_tree):
  dLoadingProcs[zone[0]] = list(range(comm.Get_size()))

print(dLoadingProcs)

# pruned_tree         = LZL.load_collective_pruned_tree(inputfile, comm, ['CGNSBase_t/Zone_t',
#                                                                        'CGNSBase_t/Family_t'/*])
