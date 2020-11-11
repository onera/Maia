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

import Converter.PyTree   as C
import Converter.Internal as I
import numpy              as NPY
import sys

# > Import PyPart
from pypart                 import DistributionBase        as DBA

from maia.cgns_io            import load_collective_size_tree       as LST
from maia.cgns_io            import cgns_io_tree                    as IOT
from maia.cgns_io.hdf_filter import elements                        as HEF
from maia.cgns_io.hdf_filter import tree                            as HTF
from maia.connectivity       import generate_ngon_from_std_elements as FTH

import maia.distribution                                      as MDI


# ------------------------------------------------------------------------
# > Pick a file
inputfile    = '/home/bmaugars/dev/dev-Tools/etc/test/pypart/Cube_ANSAd/Cube_hyb_sep.hdf'
inputfile    = '/home/castillo/ELSA_HYBRIDE/CUBES_POUR_BRUNO/cube1a.cgns'

# ------------------------------------------------------------------------
# > Load only the list of zone and sizes ...
dist_tree = LST.load_collective_size_tree(inputfile, comm)

# > ParaDiGM : dcube_gen() --> A faire

MDI.add_distribution_info(dist_tree, comm, distribution_policy='uniform')

I.printTree(dist_tree)

hdf_filter = dict()
HTF.create_tree_hdf_filter(dist_tree, hdf_filter)

for key, val in hdf_filter.items():
  print("*****", type(key))
  print("*****", type(val))
  print(key, val)

IOT.load_tree_from_filter(inputfile, dist_tree, comm, hdf_filter)

FTH.generate_ngon_from_std_elements(dist_tree, comm)

# I.printTree(dist_tree)
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

# size_tree         = LST.load_collective_size_tree(inputfile, comm, ['CGNSBase_t/Zone_t',
#                                                                        'CGNSBase_t/Family_t'/*])
