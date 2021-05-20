from mpi4py import MPI
import logging as LOG
import numpy     as np
import maia.sids.sids     as SIDS

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
from maia.generate                    import dcube_generator         as DCG
from maia.partitioning.load_balancing import setup_partition_weights as DBA
from maia.interpolation               import mesha_to_meshb          as ITP
from maia.cgns_io                     import save_part_tree          as SPT

# ---------------------------------------------------------
n_vtx       = 3
edge_length = 2.
origin      = [0., 0., 0.]
dist_tree_src    = DCG.dcube_generate(n_vtx, edge_length, origin, comm)

# ---------------------------------------------------------
n_vtx       = 3
edge_length = 2.
origin      = [0., 0., 0.]
dist_tree_target = DCG.dcube_generate(n_vtx, edge_length, origin, comm)

# ---------------------------------------------------------
dzone_to_weighted_parts_src    = DBA.npart_per_zone(dist_tree_src   , comm, 1)
dzone_to_weighted_parts_target = DBA.npart_per_zone(dist_tree_target, comm, 1)

# ---------------------------------------------------------
part_tree_src    = PPA.partitioning(dist_tree_src   , comm, zone_to_parts=dzone_to_weighted_parts_src   )
part_tree_target = PPA.partitioning(dist_tree_target, comm, zone_to_parts=dzone_to_weighted_parts_target)

# ---------------------------------------------------------
# > Create a flow solution
for zone in I.getZones(part_tree_src):
  n_cell = SIDS.zone_n_cell(zone)
  fs = I.newFlowSolution("FlowSolution#Init", gridLocation='CellCenter', parent=zone)
  da = I.newDataArray("Density", np.linspace(1., 2., num=n_cell), parent=fs)


# SPT.save_part_tree(part_tree_src   , 'part_tree_src'   , comm)
# SPT.save_part_tree(part_tree_target, 'part_tree_target', comm)

# # ---------------------------------------------------------
# ITP.mesha_to_meshb(part_tree_src, part_tree_target, comm, order=0)


# # ---------------------------------------------------------
# SPT.save_part_tree(part_tree_src   , 'part_tree_src'   , comm)
# SPT.save_part_tree(part_tree_target, 'part_tree_target', comm)

# I.printTree(dist_tree)

# C.convertPyTree2File(dist_tree, "dcube_gen_{0}.hdf".format(rank))
