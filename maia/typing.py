import typing
from typing import (
    Tuple, List, Dict, Optional, Any, Callable, Union, Iterator, NamedTuple,
    TypeVar, Generic, Type, Protocol, Sequence, Set, Literal,
    TextIO, Iterable)

from os import PathLike
from mpi4py.MPI import Comm as MPIComm
from maia.pytree.typing import CGNSTree, CGNSPath
from numpy.typing import NDArray


import numpy as np
try: #Require numpy >= 1.20
  from numpy.typing import ArrayLike, DTypeLike
except ImportError:  #pragma: no cover
  ArrayLike = Any
  DTypeLike = Any

CGNSDistTree = typing.NewType('CGNSDistTree', CGNSTree)  
CGNSPartTree = typing.NewType('CGNSPartTree', CGNSTree)  
