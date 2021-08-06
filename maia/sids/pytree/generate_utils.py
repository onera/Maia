import inspect
from functools import partial
import sys
import numpy as np
import Converter.Internal as I
import maia.utils.py_utils as PYU

from .predicate import match_name, match_value, match_label, \
    match_name_value, match_name_label, match_value_label, match_name_value_label

module_object = sys.modules["maia.sids.Internal_ext"] #Todo JC : replace with __name__
MAXDEPTH = 10

allfuncs = {
  'Name' : (match_name,  ('name',)),
  'Value': (match_value, ('value',)),
  'Label': (match_label, ('label',)),
  'NameAndValue' : (match_name_value,  ('name', 'value',)),
  'NameAndLabel' : (match_name_label,  ('name', 'label',)),
  'ValueAndLabel': (match_value_label, ('value', 'label',)),
  'NameValueAndLabel': (match_name_value_label, ('name', 'value', 'label',)),
}


def _overload_depth(function, depth):
  """
  Return a new function (with update doc and name) build from function with fixed parameter depth=depth
  """
  #Partial functions does not have __name__, but .func.__name__ Manage both with try/except
  try:
    input_name = function.__name__
  except AttributeError:
    input_name = function.func.__name__

  func = partial(function, depth=depth)
  func.__name__ = f"{input_name}{depth}"
  func.__doc__  = f"Specialization of {input_name} with depth={depth}"
  return func

def _overload_predicate(function, suffix, predicate_signature):
  """
  Return a new function (with update doc and name) build from function with fixed predicate
  Suffix is will replace 'Predicate' in the name of the generated function
  predicate_signature is the tuple (predicate_function, predicate_name_of_arguments) where
  predicate_name_of_arguments does not include node
  """
  input_name = function.__name__
  predicate, nargs = predicate_signature
  predicate_info = f"{predicate.__name__}{inspect.signature(predicate)}"

  #Creation of the specialized function : arguments are the new predicate function + the name of the
  #arguements of this predicated function
  def create_specialized_func(predicate, nargs):
    def _specialized(root, *args, **kwargs):
      pkwargs = dict([(narg, arg,) for narg, arg in zip(nargs, args)])
      #At execution, replace the generic predicate with the specialized predicate function and 
      # pass runtime arguments as named arguments to the specialized predicate
      # Other kwargs are directly passed to the specialized function
      return function(root, partial(predicate, **pkwargs), **kwargs)
    return _specialized

  func = create_specialized_func(predicate, nargs)
  func.__name__ = input_name.replace('Predicate', suffix)
  func.__doc__   = f"Specialization of {input_name} with embedded predicate\n  {predicate_info}"
  return func

def generate_functions(function, maxdepth=MAXDEPTH, easypredicates=allfuncs):
  """
  From a XXXFromPredicate function, generate and register in module 
    - the depth variants XXXFromPredicateN from 1 to maxdepth
    - the 'easy predicate' variants XXXFromName, XXXFromLabel, etc. depending on easy_predicates
    - the easy predicate + depth variants XXXFromNameN
    - the snake case functions
  """
  #Generate Predicate function with specific level
  for depth in range(1,maxdepth+1):
    func = _overload_depth(function, depth) 
    setattr(module_object, func.__name__, func)
    setattr(module_object, PYU.camel_to_snake(func.__name__), func)

  for suffix, predicate_signature in easypredicates.items():

    #Generate Name,Type,etc function without specific level ...
    func = _overload_predicate(function, suffix, predicate_signature)
    setattr(module_object, func.__name__, func)
    setattr(module_object, PYU.camel_to_snake(func.__name__), func)

    # ... and with specific level
    for depth in range(1,maxdepth+1):
      dfunc = _overload_depth(func, depth) 
      setattr(module_object, dfunc.__name__, dfunc)
      setattr(module_object, PYU.camel_to_snake(dfunc.__name__), dfunc)

#Generation for cgns names
# --------------------------------------------------------------------------
#
#   getAcoustic, ..., getCoordinateX, ..., getZoneSubRegionPointers
#
# --------------------------------------------------------------------------
def create_functions_name(create_function, name):
  snake_name = PYU.camel_to_snake(name)

  # Generate getAcoustic, ..., getCoordinateX, ..., getZoneSubRegionPointers
  funcname = f"get{name}"
  func = create_function(match_name, ('name',), (name,))
  func.__name__ = funcname
  func.__doc__  = """get the CGNS node with name {0}.""".format(name)
  setattr(module_object, funcname, partial(func, search='dfs'))
  # Generate get_acoustic, ..., get_coordinate_x, ..., get_zone_sub_region_pointers
  funcname = f"get_{snake_name}"
  func = create_function(match_name, ('name',), (name,))
  func.__name__ = funcname
  func.__doc__  = """get the CGNS node with name {0}.""".format(name)
  setattr(module_object, funcname, partial(func, search='dfs'))

  for depth in range(1,MAXDEPTH+1):
    # Generate getAcoustic1, ..., getCoordinateX1, ..., getZoneSubRegionPointers1
    funcname = f"get{name}{depth}"
    func = create_function(match_name, ('name',), (name,))
    func.__name__ = funcname
    func.__doc__  = """get the CGNS node with name {0} with depth={1}""".format(name, depth)
    setattr(module_object, funcname, partial(func, search='dfs', depth=depth))
    # Generate get_acoustic1, ..., get_coordinateX1, ..., get_zone_sub_region_pointers1
    funcname = f"get_{snake_name}{depth}"
    func = create_function(match_name, ('name',), (name,))
    func.__name__ = funcname
    func.__doc__  = """get the CGNS node with name {0} with depth={1}""".format(name, depth)
    setattr(module_object, funcname, partial(func, search='dfs', depth=depth))
