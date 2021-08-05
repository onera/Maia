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


def generate_functions(function, create_function, search, funcs, mesg):
  snake_name = PYU.camel_to_snake(function.__name__)
  prefix = function.__name__.replace('Predicate', '')
  # print(f"function          : {function}")
  # print(f"function.__name__ : {function.__name__}")
  # print(f"prefix          : {prefix}")
  # print(f"snake_name      : {snake_name}")

  for depth in range(1,MAXDEPTH+1):
    doc = """{0} from a predicate with depth={1}""".format(mesg, depth)
    # Generate getXXXFromPredicate1, getXXXFromPredicate2, ..., getXXXFromPredicate{MAXDEPTH}
    funcname = f"{function.__name__}{depth}"
    func = partial(function, search='dfs', depth=depth)
    func.__name__ = funcname
    func.__doc__  = doc
    setattr(module_object, funcname, func)
    # Generate get_xxx_from_predicate1, get_xxx_from_predicate2, ..., get_xxx_from_predicate{MAXDEPTH}
    funcname = f"{snake_name}{depth}"
    func = partial(function, search='dfs', depth=depth)
    func.__name__ = funcname
    func.__doc__  = doc
    setattr(module_object, funcname, func)

  for what, item in funcs.items():
    dwhat = ' '.join(PYU.camel_to_snake(what).split('_'))
    predicate, nargs = item

    # Generate getXXXFromName, getXXXFromValue, ..., getXXXFromNameValueAndLabel
    funcname = f"{prefix}{what}"
    func = create_function(predicate, nargs)
    func.__name__ = funcname
    func.__doc__  = """{0} from a {1}""".format(mesg, dwhat)
    setattr(module_object, funcname, partial(func, search=search))
    # Generate get_xxx_from_name, get_xxx_from_value, ..., get_xxx_from_name_value_and_label
    funcname = PYU.camel_to_snake(f"{prefix}{what}")
    # print(f"function.__name__ = {function.__name__}, funcname = {funcname}")
    func = create_function(predicate, nargs)
    func.__name__ = funcname
    func.__doc__  = """{0} from a {1}""".format(mesg, dwhat)
    setattr(module_object, funcname, partial(func, search=search))

    for depth in range(1,MAXDEPTH+1):
      # Generate getXXXFromName1, getXXXFromName2, ..., getXXXFromName{MAXDEPTH}
      # Generate getXXXFromValue1, getXXXFromValue2, ..., getXXXFromValue{MAXDEPTH}
      #   ...
      # Generate getXXXFromNameValueAndLabel1, getXXXFromNameValueAndLabel2, ..., getXXXFromNameValueAndLabel{MAXDEPTH}
      funcname = f"{prefix}{what}{depth}"
      func = create_function(predicate, nargs)
      func.__name__ = funcname
      func.__doc__  = """{0} from a {1} with depth={2}""".format(mesg, dwhat, depth)
      setattr(module_object, funcname, partial(func, search='dfs', depth=depth))
      # Generate get_xxx_from_name1, get_xxx_from_name2, ..., get_xxx_from_name{MAXDEPTH}
      # Generate get_xxx_from_value1, get_xxx_from_value2, ..., get_xxx_from_value{MAXDEPTH}
      #   ...
      # Generate get_xxx_from_name_value_and_label1, get_xxx_from_name_value_and_label2, ..., get_xxx_from_name_value_and_label{MAXDEPTH}
      funcname = "{0}{1}".format(PYU.camel_to_snake(f"{prefix}{what}"), depth)
      func = create_function(predicate, nargs)
      func.__name__ = funcname
      func.__doc__  = """{0} from a {1} with depth={2}""".format(mesg, dwhat, depth)
      setattr(module_object, funcname, partial(func, search='dfs', depth=depth))


# RM nodes
def generate_rmkeep_functions(function, create_function, funcs, mesg):
  snake_name = PYU.camel_to_snake(function.__name__)
  prefix = function.__name__.replace('Predicate', '')

  for what, item in funcs.items():
    dwhat = ' '.join(PYU.camel_to_snake(what).split('_'))
    predicate, nargs = item

    # Generate xxxChildrenFromName, xxxChildrenFromValue, ..., xxxChildrenFromNameValueAndLabel
    funcname = f"{prefix}{what}"
    func = create_function(predicate, nargs)
    func.__name__ = funcname
    func.__doc__  = """{0} from a {1}""".format(mesg, dwhat)
    setattr(module_object, funcname, func)
    # Generate xxx_children_from_name, xxx_children_from_value, ..., xxx_children_from_name_value_and_label
    funcname = PYU.camel_to_snake(f"{prefix}{what}")
    func = create_function(predicate, nargs)
    func.__name__ = funcname
    func.__doc__  = """{0} from a {1}""".format(mesg, dwhat)
    setattr(module_object, funcname, func)

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
