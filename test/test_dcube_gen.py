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


from maia.generate           import dcube_generator                 as DCG

n_vtx       = 3
edge_length = 2.
origin      = [0., 0., 0.]

dist_tree = DCG.dcube_generate(n_vtx, edge_length, origin, comm)

# I.printTree(dist_tree)

C.convertPyTree2File(dist_tree, "dcube_gen_{0}.hdf".format(rank))
