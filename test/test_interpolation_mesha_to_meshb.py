from mpi4py import MPI
import logging as LOG
import numpy     as np
import maia.sids.sids     as SIDS
from maia.sids import Internal_ext as IE

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

from maia.partitioning                import part                    as PPA
from maia.distribution                import distribution_function   as DF
from maia.generate                    import dcube_generator         as DCG
from maia.partitioning.load_balancing import setup_partition_weights as DBA
from maia.interpolation               import interpolate          as ITP
from maia.cgns_io                     import save_part_tree          as SPT
from maia.tree_exchange.dist_to_part import data_exchange as MBTP

# ---------------------------------------------------------
n_vtx       = 4
edge_length = 1.
origin      = [0., 0., 0.]
dist_tree_src    = DCG.dcube_generate(n_vtx, edge_length, origin, comm)
zone = I.getZones(dist_tree_src)[0]
d_fs = I.newFlowSolution("FlowSolution#Init", gridLocation='CellCenter', parent=zone)
distri = DF.uniform_distribution((n_vtx-1)**3, comm)
da = I.newDataArray("Density", np.arange(distri[0], distri[1], dtype=np.float64)+1, parent=d_fs)

# ---------------------------------------------------------
n_vtx       = 5
edge_length = 1.
origin      = [0., 0., 0.]
# origin      = [-0.75, 0., 0.]
# origin      = [-1.05, -1.05, -1.05]
# origin      = [-1.05, 0., 0.]
origin      = [0.85, 0.5, 0.]
dist_tree_target = DCG.dcube_generate(n_vtx, edge_length, origin, comm)

#Simplify
I._rmNodesByName(dist_tree_src, 'ZoneBC')
I._rmNodesByName(dist_tree_target, 'ZoneBC')
# ---------------------------------------------------------
dzone_to_weighted_parts_src    = DBA.npart_per_zone(dist_tree_src   , comm, 2)
dzone_to_weighted_parts_target = DBA.npart_per_zone(dist_tree_target, comm, 3)

# ---------------------------------------------------------
part_tree_src    = PPA.partitioning(dist_tree_src   , comm, zone_to_parts=dzone_to_weighted_parts_src   )
part_tree_target = PPA.partitioning(dist_tree_target, comm, zone_to_parts=dzone_to_weighted_parts_target)
#Simplify
I._rmNodesByName(part_tree_src, ':CGNS#Ppart')
I._rmNodesByName(part_tree_target, ':CGNS#Ppart')

#Transfert source flow sol so we have a sol independant from parallelism for dbg
MBTP.dist_sol_to_part_sol(zone, I.getZones(part_tree_src), comm)
for part in I.getZones(part_tree_target):
  p_fs = I.newFlowSolution("FlowSolution", gridLocation='CellCenter', parent=part)

# ---------------------------------------------------------

# SPT.save_part_tree(part_tree_src   , 'part_tree_src'   , comm)
# SPT.save_part_tree(part_tree_target, 'part_tree_target', comm)

# ---------------------------------------------------------
ITP.interpolate_from_part_trees(part_tree_src, part_tree_target, comm, ['FlowSolution#Init'], 'CellCenter') 

for part_zone_target in I.getZones(part_tree_target):
  gnum = IE.getGlobalNumbering(I.getZones(part_zone_target)[0], 'Cell')
  sol  = I.getNodeFromName(part_zone_target, 'Density')
  #print(comm.Get_rank(), part_zone_target[0], gnum, sol)

# ---------------------------------------------------------
# I.printTree(part_tree_target, "part_tree_target_{0}.txt".format(comm.Get_rank()))
SPT.save_part_tree(part_tree_src   , 'part_tree_src'   , comm)
SPT.save_part_tree(part_tree_target, 'part_tree_target', comm)


# I.printTree(dist_tree)

# C.convertPyTree2File(dist_tree, "dcube_gen_{0}.hdf".format(rank))
