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
  TextIO,
  Tuple,
  Union
)

# Third party types
from numpy.typing import NDArray, ArrayLike, DTypeLike

# Define maia.pytree specific types
CGNSTree = Tuple[str, Optional[NDArray], List["CGNSTree"], str]
CGNSPath = str
