import types
from functools import wraps

from maia.pytree.typing import *
import maia.pytree as PT


class CGNSNodeFromPredicateNotFoundError(Exception):
    """
    Attributes:
        node (List): CGNS node
        name (str): Name of the CGNS Name
    """
    def __init__(self, node: List, predicate):
        self.node = node
        self.predicate = predicate
        super().__init__()

    def __str__(self):
        return f"Unable to find the predicate '{self.predicate}' from the CGNS node '[n:{PT.get_name(self.node)}, ..., l:{PT.get_label(self.node)}]"

class CGNSLabelNotEqualError(Exception):
    """
    Attributes:
        node (List): CGNS node
        label (str): Name of the CGNS Label
    """
    def __init__(self, node: List, label: str):
        self.node  = node
        self.label = label
        super().__init__()

    def __str__(self):
        return f"Expected a CGNS node with label '{self.label}', '[n:{PT.get_name(self.node)}, ..., l:{PT.get_label(self.node)}]' found here."


def check_is_label(label, n=0):
  def _check_is_label(f):
    @wraps(f)
    def wrapped_method(*args, **kwargs):
      node = args[n]
      if PT.get_label(node) != label:
        raise CGNSLabelNotEqualError(node, label)
      return f(*args, **kwargs)
    return wrapped_method
  return _check_is_label

def check_in_labels(labels, n=0):
  def _check_in_labels(f):
    @wraps(f)
    def wrapped_method(*args, **kwargs):
      node = args[n]
      if PT.get_label(node) not in labels:
        raise CGNSLabelNotEqualError(node, labels)
      return f(*args, **kwargs)
    return wrapped_method
  return _check_in_labels

def for_all_methods(decorator):
  """
  This is a class decorator which take a function decorator as argument and
  apply it to all the functions and static methods of the class
  https://stackoverflow.com/questions/35292547/how-to-decorate-class-or-static-methods
  https://is.gd/wWcG5U
  """
  def _cls_decorator(cls):
    for name, member in vars(cls).items():
      # Good old function object, just decorate it
      if isinstance(member, (types.FunctionType, types.BuiltinFunctionType)):
          setattr(cls, name, decorator(member))
          continue
      # Static and class methods: do the dark magic
      if isinstance(member, (classmethod, staticmethod)):
        inner_func = member.__func__
        method_type = type(member)
        setattr(cls, name, method_type(decorator(inner_func)))
        continue
    return cls

  return _cls_decorator