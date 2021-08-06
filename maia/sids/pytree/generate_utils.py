import inspect
from   functools import partial
import numpy as np

from .predicate import match_name, match_value, match_label, \
    match_name_value, match_name_label, match_value_label, match_name_value_label

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
  From a XXXFromPredicate function, generate :
    - the depth variants XXXFromPredicateN from 1 to maxdepth
    - the 'easy predicate' variants XXXFromName, XXXFromLabel, etc. depending on easy_predicates
    - the easy predicate + depth variants XXXFromNameN
    - the snake case functions
  Return a dictionnary containing name of generated functions and generated functions
  """
  generated = {}
  #Generate Predicate function with specific level
  for depth in range(1,maxdepth+1):
    func = _overload_depth(function, depth) 
    generated[func.__name__] = func

  for suffix, predicate_signature in easypredicates.items():

    #Generate Name,Type,etc function without specific level ...
    func = _overload_predicate(function, suffix, predicate_signature)
    generated[func.__name__] = func

    # ... and with specific level
    for depth in range(1,maxdepth+1):
      dfunc = _overload_depth(func, depth) 
      generated[dfunc.__name__] = dfunc

  return generated
