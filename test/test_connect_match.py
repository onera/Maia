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
from maia.cgns_io            import save_part_tree                  as SPT
from maia.cgns_io.hdf_filter import elements                        as HEF
from maia.cgns_io.hdf_filter import tree                            as HTF
from maia.connectivity       import generate_ngon_from_std_elements as FTH
from maia.partitioning       import part                            as PPA
from maia.geometry           import connect_match                   as CMA
import maia.distribution                                            as MDI

from   Converter import cgnskeywords as CGK

# ------------------------------------------------------------------------
# > Pick a file
inputfile    = '/home/bmaugars/dev/dev-Tools/etc/test/pypart/data/CaseU_C11_TwoCubes_NoJoin.hdf'

# ------------------------------------------------------------------------
# > Load only the list of zone and sizes ...
dist_tree = LST.load_collective_size_tree(inputfile, comm)

# > ParaDiGM : dcube_gen() --> A faire

MDI.add_distribution_info(dist_tree, comm, distribution_policy='uniform')

# I.printTree(dist_tree)

hdf_filter = dict()
HTF.create_tree_hdf_filter(dist_tree, hdf_filter)

# for key, val in hdf_filter.items():
#   print(key, val)

# skip_type_ancestors = ["Zone_t/FlowSolution_t/"]
# skip_type_ancestors = [[CGK.Zone_t, "FlowSolution#EndOfRun"], ["ZoneSubRegion_t", "VelocityY"]]
# skip_type_ancestors = [[CGK.Zone_t, "FlowSolution#EndOfRun", "*"], ["Zone_t", "ZoneSubRegion_t", "VelocityY"]]
skip_type_ancestors = [[CGK.Zone_t, "FlowSolution#EndOfRun", "Momentum*"],
                       ["Zone_t", "ZoneSubRegion_t", "Velocity*"]]
# hdf_filter_wo_fs = IOT.filtering_filter(dist_tree, hdf_filter, skip_type_ancestors, skip=True)
# # IOT.load_tree_from_filter(inputfile, dist_tree, comm, hdf_filter)

# for key, val in hdf_filter_wo_fs.items():
#   print(key, val)
# IOT.load_tree_from_filter(inputfile, dist_tree, comm, hdf_filter_wo_fs)
IOT.load_tree_from_filter(inputfile, dist_tree, comm, hdf_filter)

# FTH.generate_ngon_from_std_elements(dist_tree, comm)

# C.convertPyTree2File(dist_tree, "dist_tree_{0}.hdf".format(rank))

# I.printTree(dist_tree)
# > To copy paste in new algorithm
# dzone_to_proc = compute_distribution_of_zones(dist_tree, distribution_policy='uniform', comm)
# > dzone_to_weighted_parts --> Proportion de la zone initiale qu'on souhate après partitionnement
# > dloading_procs        --> Proportion de la zone initiale avant le partitionnement (vision block)
#
# > ... and this is suffisent to predict your partitions sizes
dzone_to_weighted_parts = DBA.computePartitioningWeights(dist_tree, comm)

# print(dzone_to_weighted_parts)

dloading_procs = dict()
for zone in I.getZones(dist_tree):
  dloading_procs[zone[0]] = list(range(comm.Get_size()))
# print(dloading_procs)

part_tree = PPA.partitioning(dist_tree, dzone_to_weighted_parts,
                             comm,
                             split_method=2,
                             part_weight_method=1,
                             reorder_methods=["NONE", "NONE"])

CMA.connect_match_from_family(part_tree, ['JOIN_1', 'JOIN_2'], comm,
                              match_type = ['FaceCenter'], rel_tol=1e-5)

# I.printTree(part_tree)
SPT.save_part_tree(part_tree, 'part_tree', comm)
# C.convertPyTree2File(part_tree, "part_tree_{0}.hdf".format(rank))

# size_tree         = LST.load_collective_size_tree(inputfile, comm, ['CGNSBase_t/Zone_t',
#                                                                        'CGNSBase_t/Family_t'/*])
