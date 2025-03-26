from typing import (
    Tuple, List, Dict, Optional, Any, Callable, Union, Iterator, NamedTuple,
    TypeVar, Generic, Type, Protocol, runtime_checkable, Sequence, Set, Literal,
    TextIO, Iterable
)
from os import PathLike
import numpy as np

try: #Require numpy >= 1.20
  from numpy.typing import ArrayLike, DTypeLike
except ImportError:  #pragma: no cover
  ArrayLike = Any
  DTypeLike = Any

# Base CGNS types
CGNSTree = Tuple[str, Optional[np.ndarray], List["CGNSTree"], str]

# Path and filter types
CGNSPath = str
