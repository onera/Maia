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


from pypart                 import DistributionBase        as DBA

from maia.cgns_registry      import tree             as CGT # Not bad :D
from maia.generate           import dplane_generator as DPG
from maia.cgns_io.hdf_filter import tree             as HTF
from maia.cgns_io            import cgns_io_tree     as IOT
from maia.partitioning       import part             as PPA
from maia.cgns_io            import save_part_tree   as SPT

# 200 / 20proc fail
# 100 / 20proc fail
# 100 / 10proc fail
# 10  / 9 porc fail
xmin = 0.
xmax = 1.
ymin = 0.
ymax = 1.
have_random = 0
init_random = 1
nx          = 8
ny          = 8

dist_tree = DPG.dplane_generate(xmin, xmax, ymin, ymax, have_random, init_random, nx, ny, comm)

cgr = CGT.add_cgns_registry_information(dist_tree, comm)

hdf_filter = dict()
HTF.create_tree_hdf_filter(dist_tree, hdf_filter)

for zone in I.getZones(dist_tree):
  fs_n = I.newFlowSolution(name='FlowSolution#Centers', gridLocation='CellCenter', parent=zone)
  zone_dim = I.getZoneDim(zone)
  n_cell   = zone_dim[2]
  N = I.newDataArray('cell_num', NPY.linspace(1, n_cell, num=n_cell), parent=fs_n)

IOT.save_tree_from_filter("dplane_mesh.hdf", dist_tree, comm, hdf_filter)

# for key, val in hdf_filter.items():
#   print(key, val)
# I.printTree(dist_tree)

dzone_to_weighted_parts = DBA.computePartitioningWeights(dist_tree, comm)

print(dzone_to_weighted_parts)

dloading_procs = dict()
for zone in I.getZones(dist_tree):
  dloading_procs[zone[0]] = list(range(comm.Get_size()))
print(dloading_procs)

part_tree = PPA.partitioning(dist_tree, dzone_to_weighted_parts,
                             comm,
                             split_method=2,
                             part_weight_method=1,
                             reorder_methods=["NONE", "NONE"])
# C.convertPyTree2File(dist_tree, "dcube_gen_{0}.hdf".format(rank))
I._rmNodesFromName(part_tree, "ZoneGridConnectivity#Vertex")
# I.printTree(part_tree)
SPT.save_part_tree(part_tree, 'part_tree', comm)

