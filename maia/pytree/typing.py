# Standard types
from typing import (
  Any,
  Callable,
  Dict,
  Iterable,
  Iterator,
  List,
  Literal,
  Mapping,
  NamedTuple,
  Optional,
  Sequence,
  TextIO,
  Tuple,
  Type,
  Union
)

# Third party types
from numpy.typing import NDArray, ArrayLike, DTypeLike

# Define maia.pytree specific types
CGNSTree = Tuple[str, Optional[NDArray], List["CGNSTree"], str]
CGNSPath = str

Predicate = Union[str, Callable[[CGNSTree], bool]]
Predicates = Union[str, List[Predicate]]
