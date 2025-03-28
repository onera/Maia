from maia.pytree.typing import *
from typing import NewType
from mpi4py import MPI


MPIComm = MPI.Comm

CGNSDistTree = NewType('CGNSDistTree', CGNSTree)  
CGNSPartTree = NewType('CGNSPartTree', CGNSTree)  
