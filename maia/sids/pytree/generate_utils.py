import inspect
from   functools import partial
import numpy as np

from maia.utils.py_utils import camel_to_snake

from .predicate import match_name, match_value, match_label, \
    match_name_value, match_name_label, match_value_label, match_name_value_label

MAXDEPTH = 3

allfuncs = {
  'Name' : (match_name,  ('name',)),
  'Value': (match_value, ('value',)),
  'Label': (match_label, ('label',)),
  'NameAndLabel' : (match_name_label,  ('name', 'label',)),
  #'NameAndValue' : (match_name_value,  ('name', 'value',)),
  #'ValueAndLabel': (match_value_label, ('value', 'label',)),
  #'NameValueAndLabel': (match_name_value_label, ('name', 'value', 'label',)),
}

def _func_name(function):
  #Partial functions does not have __name__, but .func.__name__ Manage both with try/except
  try:
    return function.__name__
  except AttributeError:
    return function.func.__name__

def _overload_depth(function, depth):
  """
  Return a new function (with update doc and name) build from function with fixed parameter depth=depth
  """
  input_name = _func_name(function)

  func = partial(function, depth=depth)
  func.__name__ = f"{input_name}{depth}"
  func.__doc__  = f"Specialization of {input_name} with depth={depth}"
  return func

def _overload_depth1(function):
  """
  Return a new function (with update doc and name) build from function with fixed parameter depth=1
  """
  input_name = _func_name(function)
  func_name = input_name.replace("Nodes", "Children")
  func_name = func_name.replace("nodes", "children")
  func_name = func_name.replace("Node", "Child")
  func_name = func_name.replace("node", "child")

  func = partial(function, depth=1)
  func.__name__ = func_name
  func.__doc__  = f"Specialization of {input_name} with depth=1"
  return func

def _overload_predicate(function, suffix, predicate_signature):
  """
  Return a new function (with update doc and name) build from function with fixed predicate
  Suffix is will replace 'Predicate' in the name of the generated function
  predicate_signature is the tuple (predicate_function, predicate_name_of_arguments) where
  predicate_name_of_arguments does not include node
  """
  input_name = _func_name(function)
  predicate, nargs = predicate_signature
  predicate_info = f"{_func_name(predicate)}{inspect.signature(predicate)}"

  #Creation of the specialized function : arguments are the new predicate function + the name of the
  #arguements of this predicated function
  def create_specialized_func(predicate, nargs):
    if "predicates" in _func_name(function):
      def _specialized(root, *args, **kwargs):
        assert len(args) == 1, "Specialized versions of from_predicates accepts only predicate args"
        npredicate  = len(args[0])
        predicates = [partial(predicate, **{nargs[0] : args[0][i]}) for i in range(npredicate)]
        return function(root, predicates, **kwargs)
    else:
      def _specialized(root, *args, **kwargs):
        pkwargs = dict([(narg, arg,) for narg, arg in zip(nargs, args)])
        #At execution, replace the generic predicate with the specialized predicate function and 
        # pass runtime arguments as named arguments to the specialized predicate
        # Other kwargs are directly passed to the specialized function
        return function(root, partial(predicate, **pkwargs), **kwargs)

    return _specialized

  func = create_specialized_func(predicate, nargs)
  if 'predicate' in input_name:
    func.__name__ = input_name.replace('predicate', camel_to_snake(suffix))
  elif 'Predicate' in input_name:
    func.__name__ = input_name.replace('Predicate', suffix)
  func.__doc__   = f"Specialization of {input_name} with embedded predicate\n  {predicate_info}"
  return func

def generate_functions(function, maxdepth=MAXDEPTH, child=True, easypredicates=allfuncs):
  """
  From a XXX_from_predicate function, generate :
    - the depth variants XXX_from_predicateN from 1 to maxdepth
    - the 'easy predicate' variants XXX_from_name, XXX_from_label, etc. depending on easy_predicates
    - the easy predicate + depth variants XXX_from_nameN
    - the "child" versions (equivalent to depth=1)
  Return a dictionnary containing name of generated functions and generated functions
  """
  generated = {}
  #Generate Predicate function with specific level
  for depth in range(1,maxdepth+1):
    func = _overload_depth(function, depth) 
    generated[func.__name__] = func

  if child:
    func = _overload_depth1(function)
    generated[func.__name__] = func

  for suffix, predicate_signature in easypredicates.items():

    #Generate Name,Type,etc function without specific level ...
    func = _overload_predicate(function, suffix, predicate_signature)
    generated[func.__name__] = func

    # ... and with specific level
    for depth in range(1,maxdepth+1):
      dfunc = _overload_depth(func, depth) 
      generated[dfunc.__name__] = dfunc

    if child:
      dfunc = _overload_depth1(func)
      generated[dfunc.__name__] = dfunc

  return generated
