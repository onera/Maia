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
from maia.partitioning       import part                            as PPA
from maia.cgns_io            import save_part_tree                  as SPT
from maia.cgns_registry      import tree                            as CGT # Not bad :D

import maia.distribution                                      as MDI


# ------------------------------------------------------------------------
# > Pick a file
inputfile    = '/home/bmaugars/dev/dev-Tools/etc/test/pypart/Cube_ANSAd/Cube_hyb_sep.hdf'
inputfile    = '/home/bmaugars/dev/dev-Tools/maia/unit_tests_case/CUBES_POUR_BRUNO/cube8.cgns'
# inputfile    = '/home/castillo/ELSA_HYBRIDE/CUBES_POUR_BRUNO/cube1a.cgns'

# ------------------------------------------------------------------------
# > Load only the list of zone and sizes ...
dist_tree = LST.load_collective_size_tree(inputfile, comm)

# > ParaDiGM : dcube_gen() --> A faire

MDI.add_distribution_info(dist_tree, comm, distribution_policy='uniform')

# I.printTree(dist_tree)

hdf_filter = dict()
HTF.create_tree_hdf_filter(dist_tree, hdf_filter)

# for key, val in hdf_filter.items():
#   print("*****", type(key))
#   print("*****", type(val))
#   print(key, val)

IOT.load_tree_from_filter(inputfile, dist_tree, comm, hdf_filter)

FTH.generate_ngon_from_std_elements(dist_tree, comm)

# I.printTree(dist_tree)
# C.convertPyTree2File(dist_tree, "dist_tree_{0}.hdf".format(rank))
hdf_filter = dict()
HTF.create_tree_hdf_filter(dist_tree, hdf_filter, mode='write')
I.printTree(dist_tree)

IOT.save_tree_from_filter("dist_tree.hdf", dist_tree, comm, hdf_filter)

dzone_to_weighted_parts = {}
for zone in I.getZones(dist_tree):
    dzone_to_weighted_parts[zone[0]] = [1./comm.Get_size()]

# print(dzone_to_weighted_parts)

dloading_procs = dict()
for zone in I.getZones(dist_tree):
  dloading_procs[zone[0]] = list(range(comm.Get_size()))
# print(dloading_procs)

cgr = CGT.add_cgns_registry_information(dist_tree, comm)
part_tree = PPA.partitioning(dist_tree, dzone_to_weighted_parts,
                             comm,
                             split_method=2,
                             part_weight_method=1,
                             reorder_methods=["NONE", "NONE"])

# size_tree         = LST.load_collective_size_tree(inputfile, comm, ['CGNSBase_t/Zone_t',
#                                                                        'CGNSBase_t/Family_t'/*])

SPT.save_part_tree(part_tree, 'part_tree', comm)
# C.convertPyTree2File(dist_tree, "dist_tree_{0}.hdf".format(rank))
