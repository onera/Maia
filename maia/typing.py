from maia.pytree.typing import *
from mpi4py import MPI

# MPI related types
MPIComm = MPI.Comm

# Specialized tree types
CGNSDistTree = CGNSTree  # Tree with distribution information
CGNSPartTree = CGNSTree  # Tree without distribution information
