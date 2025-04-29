import typing

# Standard types
from typing import (
  Any,
  Callable,
  Dict,
  Iterable,
  Iterator,
  List,
  Literal,
  NamedTuple,
  Optional,
  Sequence,
  Set,
  Tuple,
  TypeVar,
  Union
)

# Third party types
from os           import PathLike
from mpi4py.MPI   import Intracomm as MPIComm
from numpy.typing import NDArray, ArrayLike, DTypeLike

# Reexport Pytree types
from maia.pytree.typing import CGNSTree, CGNSPath

# Define maia specific types
CGNSDistTree = typing.NewType('CGNSDistTree', CGNSTree)  
CGNSPartTree = typing.NewType('CGNSPartTree', CGNSTree)  
