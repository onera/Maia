from typing import Tuple, List, Dict, Optional, Any, Callable, Union, Iterator, NamedTuple
import numpy as np

try: #Require numpy >= 1.20
  from numpy.typing import ArrayLike, DTypeLike
except ImportError:
  ArrayLike = Any
  DTypeLike = Any

CGNSTree = Tuple[str, Optional[np.ndarray], List["CGNSTree"], str]