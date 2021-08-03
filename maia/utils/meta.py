import types

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

