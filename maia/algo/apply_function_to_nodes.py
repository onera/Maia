import maia.pytree as PT
from maia.typing import CGNSTree, Callable, Iterator, Any, TypeVar

T = TypeVar('T')

def apply_to_bases(f: Callable,
                   t: CGNSTree,
                   *args: Any) -> None:
  """Apply a function to all bases in a CGNS tree.
  
  Args:
    f: Function to apply to each base
    t: CGNS tree or base node
    *args: Additional arguments to pass to the function
  
  Raises:
    Exception: If t is not a CGNSBase_t or CGNSTree_t node
  """
  if PT.get_label(t)=="CGNSBase_t":
    b_iter = [t]
  elif PT.get_label(t)=="CGNSTree_t":
    b_iter = PT.iter_children_from_label(t, "CGNSBase_t")
  else:
    raise Exception("function \""+f.__name__+"\"" \
                    " can only be applied to a \"CGNSBase_t\" or on a complete \"CGNSTree_t\"," \
                    " not on a node of type \""+PT.get_label(t)+"\".")
  for b in b_iter:
    f(b, *args)

def zones_iterator(t: CGNSTree) -> Iterator[CGNSTree]:
  """
  Helper iterator to loop over zones from a tree, a base or a zone node
  
  Args:
    t: CGNS tree, base or zone node
  
  Returns:
    Iterator over zone nodes
  
  Raises:
    ValueError: If t is not a valid node type for zone iteration
  """
  if PT.get_label(t)=="Zone_t":
    yield t
  elif PT.get_label(t)=="CGNSBase_t":
    yield from PT.iter_children_from_label(t, "Zone_t")
  elif PT.get_label(t)=="CGNSTree_t":
    yield from PT.iter_children_from_labels(t, ["CGNSBase_t", "Zone_t"])
  else:
    raise ValueError("Unvalid object for zones iterator")

def apply_to_zones(f: Callable,
                   t: CGNSTree,
                   *args: Any) -> None:
  """Apply a function to all zones in a CGNS tree.
  
  Args:
    f: Function to apply to each zone
    t: CGNS tree, base or zone node
    *args: Additional arguments to pass to the function
  
  Raises:
    Exception: If t is not a Zone_t, CGNSBase_t or CGNSTree_t node
  """
  if PT.get_label(t) not in ["Zone_t", "CGNSBase_t", "CGNSTree_t"]:
    raise Exception("function \""+f.__name__+"\"" \
                    " can only be applied to a \"Zone_t\", a \"CGNSBase_t\" or on a complete \"CGNSTree_t\"," \
                    " not on a node of type \""+PT.get_label(t)+"\".")
  else:
    for z in zones_iterator(t):
      f(z, *args)

