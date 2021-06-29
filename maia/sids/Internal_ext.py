from typing import List, Optional, NoReturn, Union, Tuple, Callable, Any
from functools import wraps
from functools import partial
import sys
import pathlib
import fnmatch
import queue
import numpy as np
import Converter.Internal as I
from maia.sids.cgns_keywords import Label as CGL
import maia.sids.cgns_keywords as CGK
import maia.utils.py_utils as PYU

# Declare a type alias for type hint checkers
# For Python>=3.9 it is possible to set the MaxLen
# from typing import Annotated
# TreeNode = Annotated[List[Union[str, Optional[numpy.ndarray], List['TreeNode']]], MaxLen(4)]
TreeNode = List[Union[str, Optional[np.ndarray], List["TreeNode"]]]

# Keys to access TreeNode values
__NAME__     = 0
__VALUE__    = 1
__CHILDREN__ = 2
__LABEL__    = 3

# Default Containers naming
__GridCoordinates__ = "GridCoordinates"
__FlowSolutionNodes__ = "FlowSolution"
__FlowSolutionCenters__ = "FlowSolution#Centers"

module_object = sys.modules[__name__]

# --------------------------------------------------------------------------
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
        return f"Unable to find the predicate '{self.predicate}' from the CGNS node '[n:{I.getName(self.node)}, ..., l:{I.getType(self.node)}]', see : \n{I.printTree(self.node)}."

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
        return f"Expected a CGNS node with label '{self.label}', '[n:{I.getName(self.node)}, ..., l:{I.getType(self.node)}]' found here."

class NotImplementedForElementError(NotImplementedError):
    """
    Attributes:
        zone_node (List): CGNS Zone_t node
        element_node (List): CGNS Elements_t node
    """
    def __init__(self, zone_node: List, element_node: List):
        self.zone_node    = zone_node
        self.element_node = element_node
        super().__init__()

    def __str__(self):
        return f"Unstructured CGNS Zone_t named '{I.getName(self.zone_node)}' with CGNS Elements_t named '{SIDS.ElementCGNSName(self.element_node)}' is not yet implemented."

# --------------------------------------------------------------------------
def is_valid_name(name: str):
  """
  Return True if name is a valid Python/CGNS name
  """
  return isinstance(name, str) #and len(label) < 32

def is_valid_value(value):
  """
  Return True if label is a valid Python/CGNS Label
  """
  return isinstance(value, np.ndarray) and np.isfortran(value) if value.ndim > 1 else True

def is_valid_children(children):
  """
  Return True if label is a valid Python/CGNS Children
  """
  return isinstance(children, (list, tuple))

def is_valid_label(label):
  """
  Return True if label is a valid Python/CGNS Label
  """
  return isinstance(label, str) and ((label.endswith('_t') and label in CGK.Label.__members__) or label == '')

# --------------------------------------------------------------------------
def check_name(name: str):
  if is_valid_name(name):
    return name
  raise TypeError(f"Invalid Python/CGNS name '{name}'")

def check_value(value):
  if is_valid_value(value):
    return value
  raise TypeError(f"Invalid Python/CGNS value '{value}'")

def check_children(children):
  if is_valid_children(children):
    return children
  raise TypeError(f"Invalid Python/CGNS children '{children}'")

def check_label(label):
  if is_valid_label(label):
    return label
  raise TypeError(f"Invalid Python/CGNS label '{label}'")

# --------------------------------------------------------------------------
def is_valid_node(node):
  if not isinstance(node, list) and len(node) != 4 and \
      is_valid_name(I.getName(node))         and \
      is_valid_value(I.getValue(node))       and \
      is_valid_children(I.getChildren(node)) and \
      is_valid_label(I.getName(node)) :
    return False
  return True

# --------------------------------------------------------------------------
def check_is_label(label, n=0):
  def _check_is_label(f):
    @wraps(f)
    def wrapped_method(*args, **kwargs):
      node = args[n]
      if I.getType(node) != label:
        raise CGNSLabelNotEqualError(node, label)
      return f(*args, **kwargs)
    return wrapped_method
  return _check_is_label

# --------------------------------------------------------------------------
def is_same_name(n0: TreeNode, n1: TreeNode):
  return n0[0] == n1[0]

def is_same_label(n0: TreeNode, n1: TreeNode):
  return n0[3] == n1[3]

def is_same_value(n0: TreeNode, n1: TreeNode):
  return np.array_equal(n0[1], n1[1])

# --------------------------------------------------------------------------
def match_name(n, name: str):
  return fnmatch.fnmatch(n[__NAME__], name)

def match_value(n, value):
  return np.array_equal(n[__VALUE__], value)

def match_label(n, label: str):
  return n[__LABEL__] == label

def match_name_value(n, name: str, value):
  return fnmatch.fnmatch(n[__NAME__], name) and np.array_equal(n[__VALUE__], value)

def match_name_label(n, name: str, label: str):
  return n[__LABEL__] == label and fnmatch.fnmatch(n[__NAME__], name)

def match_name_value_label(n, name: str, value, label: str):
  return n[__LABEL__] == label and fnmatch.fnmatch(n[__NAME__], name) and np.array_equal(n[__VALUE__], value)

def match_value_label(n, value, label: str):
  return n[__LABEL__] == label and np.array_equal(n[__VALUE__], value)

allfuncs = {
  'Name' : (match_name,  ('name',)),
  'Value': (match_value, ('value',)),
  'Label': (match_label, ('label',)),
  'NameAndValue' : (match_name_value,  ('name', 'value',)),
  'NameAndLabel' : (match_name_label,  ('name', 'label',)),
  'ValueAndLabel': (match_value_label, ('value', 'label',)),
  'NameValueAndLabel': (match_name_value_label, ('name', 'value', 'label',)),
}

MAXDEPTH = 10

# --------------------------------------------------------------------------
def create_functions(function, create_function, method, funcs, mesg):
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
    func = partial(function, method='dfs', depth=depth)
    func.__name__ = funcname
    func.__doc__  = doc
    setattr(module_object, funcname, func)
    # Generate get_xxx_from_predicate1, get_xxx_from_predicate2, ..., get_xxx_from_predicate{MAXDEPTH}
    funcname = f"{snake_name}{depth}"
    func = partial(function, method='dfs', depth=depth)
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
    setattr(module_object, funcname, partial(func, method=method))
    # Generate get_xxx_from_name, get_xxx_from_value, ..., get_xxx_from_name_value_and_label
    funcname = PYU.camel_to_snake(f"{prefix}{what}")
    # print(f"function.__name__ = {function.__name__}, funcname = {funcname}")
    func = create_function(predicate, nargs)
    func.__name__ = funcname
    func.__doc__  = """{0} from a {1}""".format(mesg, dwhat)
    setattr(module_object, funcname, partial(func, method=method))

    for depth in range(1,MAXDEPTH+1):
      # Generate getXXXFromName1, getXXXFromName2, ..., getXXXFromName{MAXDEPTH}
      # Generate getXXXFromValue1, getXXXFromValue2, ..., getXXXFromValue{MAXDEPTH}
      #   ...
      # Generate getXXXFromNameValueAndLabel1, getXXXFromNameValueAndLabel2, ..., getXXXFromNameValueAndLabel{MAXDEPTH}
      funcname = f"{prefix}{what}{depth}"
      func = create_function(predicate, nargs)
      func.__name__ = funcname
      func.__doc__  = """{0} from a {1} with depth={2}""".format(mesg, dwhat, depth)
      setattr(module_object, funcname, partial(func, method='dfs', depth=depth))
      # Generate get_xxx_from_name1, get_xxx_from_name2, ..., get_xxx_from_name{MAXDEPTH}
      # Generate get_xxx_from_value1, get_xxx_from_value2, ..., get_xxx_from_value{MAXDEPTH}
      #   ...
      # Generate get_xxx_from_name_value_and_label1, get_xxx_from_name_value_and_label2, ..., get_xxx_from_name_value_and_label{MAXDEPTH}
      funcname = "{0}{1}".format(PYU.camel_to_snake(f"{prefix}{what}"), depth)
      func = create_function(predicate, nargs)
      func.__name__ = funcname
      func.__doc__  = """{0} from a {1} with depth={2}""".format(mesg, dwhat, depth)
      setattr(module_object, funcname, partial(func, method='dfs', depth=depth))

# --------------------------------------------------------------------------
class NodeParser:

  DEFAULT="bfs"

  def __init__(self, sort=lambda n:n[__CHILDREN__]):
    self.sort = sort

  def bfs(self, parent, predicate):
    # print(f"NodeParser.bfs: parent = {I.getName(parent)}")
    temp = queue.Queue()
    temp.put(parent)
    while not temp.empty():
      node = temp.get()
      # print(f"NodeParser.bfs: node = {I.getName(node)}")
      if predicate(node):
        return node
      for child in self.sort(node):
        temp.put(child)
    return None

  def dfs(self, parent, predicate):
    # print(f"NodeParser.dfs: parent = {I.getName(parent)}")
    if predicate(parent):
      return parent
    return self._dfs(parent, predicate)

  def _dfs(self, parent, predicate):
    # print(f"NodeParser._dfs: parent = {I.getName(parent)}")
    for child in self.sort(parent):
      if predicate(child):
        return child
      # Explore next level
      result = self._dfs(child, predicate)
      if result is not None:
        return result
    return None

# --------------------------------------------------------------------------
class LevelNodeParser:

  MAXDEPTH=30

  def __init__(self, depth=MAXDEPTH, sort=lambda n:n[__CHILDREN__]):
    self.depth = depth
    self.sort = sort

  def bfs(self, parent, predicate, level=1):
    # print(f"LevelNodeParser.bfs: depth = {self.depth}: parent = {I.getName(parent)}")
    temp = queue.Queue()
    temp.put( (0, parent,) )
    while not temp.empty():
      level, node = temp.get()
      # print(f"LevelNodeParser.bfs: level:{level} < depth:{self.depth}: node = {I.getName(node)}")
      if predicate(node):
        return node
      if level < self.depth:
        for child in self.sort(node):
          temp.put( (level+1, child) )
    return None

  def dfs(self, parent, predicate):
    # print(f"LevelNodeParser.dfs: depth = {self.depth}: parent = {I.getName(parent)}")
    if predicate(parent):
      return parent
    return self._dfs(parent, predicate)

  def _dfs(self, parent, predicate, level=1):
    # print(f"LevelNodeParser.dfs: level = {level} < depth = {self.depth}: parent = {I.getName(parent)}")
    for child in self.sort(parent):
      if predicate(child):
        return child
      if level < self.depth:
        # Explore next level
        result = self._dfs(child, predicate, level=level+1)
        if result is not None:
          return result
    return None

# --------------------------------------------------------------------------
class NodeWalker:
  """ Return the first node found in the Python/CGNS tree """

  FORWARD  = lambda n:n[__CHILDREN__]
  BACKWARD = lambda n:reverse(n[__CHILDREN__])

  def __init__(self, parent: TreeNode, predicate: Callable[[TreeNode], bool],
                     method=NodeParser.DEFAULT, depth=0, sort=FORWARD):
    self.parent    = parent
    self.predicate = predicate
    # Register default value
    self.method = method
    self.depth  = depth
    self.sort  = sort

  @property
  def parent(self):
    return self._parent

  @parent.setter
  def parent(self, node: TreeNode):
    if is_valid_node(node):
      self._parent = node

  @property
  def predicate(self):
    return self._predicate

  @predicate.setter
  def predicate(self, predicate: Callable[[TreeNode], bool]):
    if callable(predicate):
      self._predicate = predicate
    else:
      raise TypeError("predicate must be a callable function.")

  @property
  def method(self):
    return self._method

  @method.setter
  def method(self, value: str):
    if value in ['bfs', 'dfs']:
      self._method = value
    else:
      raise ValueError("method must 'bfs' or 'dfs'.")

  @property
  def depth(self):
    return self._depth

  @depth.setter
  def depth(self, value: str):
    if isinstance(value, int) and value >= 0:
      self._depth = value
    else:
      raise ValueError("depth must a integer >= 0.")

  @property
  def sort(self):
    return self._sort

  @sort.setter
  def sort(self, value):
    if callable(value):
      self._sort = value
    else:
      raise TypeError("sort must be a callable function.")

  @property
  def parser(self):
    return self._parser

  def __call__(self, parent=None, predicate=None, method=None, explore=None, depth=None, sort=None):
    if parent and parent != self.parent:
      self.parent = parent
    if predicate and predicate != self.predicate:
      self.predicate = predicate
    if method and method != self.method:
      self.method = method
    if depth and depth != self.depth:
      self.depth = depth
    if sort and sort != self.sort:
      self.sort = sort
    # Create parser
    self._parser = LevelNodeParser(depth=self.depth, sort=self.sort) if self.depth > 0 else NodeParser(sort=self.sort)
    func = getattr(self._parser, self.method)
    return func(self._parent, self._predicate)


# --------------------------------------------------------------------------
def requestNodeFromPredicate(parent, predicate, method=NodeParser.DEFAULT, depth=None):
  walker = NodeWalker(parent, predicate)
  return walker(method=method, depth=depth)
  # parser = LevelNodeParser(depth=depth) if isinstance(depth, int) else NodeParser()
  # func   = getattr(parser, method)
  # return func(parent, predicate)

def create_request_child(predicate, nargs):
  def _get_request_from(parent, *args, **kwargs):
    pkwargs = dict([(narg, arg,) for narg, arg in zip(nargs, args)])
    return requestNodeFromPredicate(parent, partial(predicate, **pkwargs), **kwargs)
  return _get_request_from

create_functions(requestNodeFromPredicate, create_request_child, "bfs", allfuncs,
  "Return a child CGNS node or None (if it is not found)")

# --------------------------------------------------------------------------
def getNodeFromPredicate(parent, predicate, default=None, method=NodeParser.DEFAULT, depth=None):
  """ Return the list of first level childs of node matching a given predicate (callable function)"""
  node = requestNodeFromPredicate(parent, predicate, method=method, depth=depth)
  if node is not None:
    return node
  if default and is_valid_node(default):
    return default
  raise CGNSNodeFromPredicateNotFoundError(parent, predicate)

def create_get_child(predicate, nargs):
  def _get_node_from(parent, *args, **kwargs):
    pkwargs = dict([(narg, arg,) for narg, arg in zip(nargs, args)])
    try:
      return getNodeFromPredicate(parent, partial(predicate, **pkwargs), **kwargs)
    except CGNSNodeFromPredicateNotFoundError as e:
      print(f"For predicate : pkwargs = {pkwargs}", file=sys.stderr)
      raise e
  return _get_node_from

create_functions(getNodeFromPredicate, create_get_child, "bfs", allfuncs,
  "Return a child CGNS node or raise a CGNSNodeFromPredicateNotFoundError (if it is not found)")

# --------------------------------------------------------------------------
class NodesParser:

  DEFAULT="bfs"

  def __init__(self, func, sort=lambda n:n[__CHILDREN__]):
    self.func = func
    self.sort = sort

  def bfs(self, parent, predicate):
    # print(f"NodesParser.bfs: parent = {I.getName(parent)}")
    temp = queue.Queue()
    temp.put(parent)
    while not temp.empty():
      node = temp.get()
      # print(f"NodesParser.bfs: node = {I.getName(node)}")
      if predicate(node):
        self.func(node)
      for child in self.sort(node):
        temp.put(child)

  def dfs(self, parent, predicate):
    # print(f"NodesParser.dfs: parent = {I.getName(parent)}")
    if predicate(parent):
      self.func(parent)
    return self._dfs(parent, predicate)

  def _dfs(self, parent, predicate):
    # print(f"NodesParser._dfs: parent = {I.getName(parent)}")
    for child in self.sort(parent):
      if predicate(child):
        self.func(child)
      # Explore next level
      self._dfs(child, predicate)

# --------------------------------------------------------------------------
class ShallowNodesParser:

  """ Stop exploration if something found at a level """

  def __init__(self, func, sort=lambda n:n[__CHILDREN__]):
    self.func = func
    self.sort = sort

  def bfs(self, parent, predicate):
    # print(f"ShallowNodesParser.bfs: parent = {I.getName(parent)}")
    temp = queue.Queue()
    temp.put(parent)
    while not temp.empty():
      node = temp.get()
      # print(f"ShallowNodesParser.bfs: node = {I.getName(node)}")
      if predicate(node):
        self.func(node)
      else:
        for child in self.sort(node):
          temp.put(child)

  def dfs(self, parent, predicate):
    # print(f"ShallowNodesParser.dfs: parent = {I.getName(parent)}")
    if predicate(parent):
      self.func(parent)
    return self._dfs(parent, predicate)

  def _dfs(self, parent, predicate):
    # print(f"ShallowNodesParser._dfs: parent = {I.getName(parent)}")
    results = []
    for child in self.sort(parent):
      if predicate(child):
        self.func(child)
      else:
        # Explore next level
        self._dfs(child, predicate)

# --------------------------------------------------------------------------
class LevelNodesParser:

  """ Stop exploration at level """

  MAXDEPTH=30

  def __init__(self, func, depth=MAXDEPTH, sort=lambda n:n[__CHILDREN__]):
    self.func  = func
    self.depth = depth
    self.sort  = sort

  def bfs(self, parent, predicate):
    # print(f"LevelNodesParser.bfs: depth = {self.depth}: parent = {I.getName(parent)}")
    temp = queue.Queue()
    temp.put( (0, parent,) )
    while not temp.empty():
      level, node = temp.get()
      # print(f"LevelNodesParser.bfs: level:{level} < depth:{self.depth}: node = {I.getName(node)}")
      if predicate(node):
        self.func(node)
      if level < self.depth:
        for child in self.sort(node):
          temp.put( (level+1, child) )

  def dfs(self, parent, predicate):
    # print(f"LevelNodesParser.dfs: depth = {self.depth}: parent = {I.getName(parent)}")
    if predicate(parent):
      self.func(parent)
    return self._dfs(parent, predicate)

  def _dfs(self, parent, predicate, level=1):
    # print(f"LevelNodesParser._dfs: level = {level} < depth = {self.depth}: parent = {I.getName(parent)}")
    for child in self.sort(parent):
      if predicate(child):
        self.func(child)
      if level < self.depth:
        # Explore next level
        self._dfs(child, predicate, level=level+1)

# --------------------------------------------------------------------------
class NodesIterator:

  DEFAULT="bfs"

  def __init__(self, sort=lambda n:n[__CHILDREN__]):
    self.sort = sort

  def bfs(self, parent, predicate):
    # print(f"NodesIterator.bfs: parent = {I.getName(parent)}")
    temp = queue.Queue()
    temp.put(parent)
    while not temp.empty():
      node = temp.get()
      # print(f"NodesIterator.bfs: node = {I.getName(node)}")
      if predicate(node):
        yield node
      for child in self.sort(node):
        temp.put(child)

  def dfs(self, parent, predicate):
    # print(f"NodesIterator.dfs: parent = {I.getName(parent)}")
    if predicate(parent):
      yield parent
    yield from self._dfs(parent, predicate)

  def _dfs(self, parent, predicate):
    # print(f"NodesIterator._dfs: parent = {I.getName(parent)}")
    for child in self.sort(parent):
      if predicate(child):
        yield child
      # Explore next level
      yield from self._dfs(child, predicate)

# --------------------------------------------------------------------------
class ShallowNodesIterator:

  """ Stop exploration if something found at a level """

  def __init__(self, sort=lambda n:n[__CHILDREN__]):
    self.sort = sort

  def bfs(self, parent, predicate):
    # print(f"ShallowNodesIterator.bfs: parent = {I.getName(parent)}")
    temp = queue.Queue()
    temp.put(parent)
    while not temp.empty():
      node = temp.get()
      # print(f"ShallowNodesIterator.bfs: node = {I.getName(node)}")
      if predicate(node):
        yield node
      else:
        for child in self.sort(node):
          temp.put(child)

  def dfs(self, parent, predicate):
    # print(f"ShallowNodesIterator.dfs: parent = {I.getName(parent)}")
    if predicate(parent):
      yield parent
    yield from self._dfs(parent, predicate)

  def _dfs(self, parent, predicate):
    # print(f"ShallowNodesIterator._dfs: parent = {I.getName(parent)}")
    for child in self.sort(parent):
      if predicate(child):
        yield child
      else:
        # Explore next level
        yield from self._dfs(child, predicate)

# --------------------------------------------------------------------------
class LevelNodesIterator:

  """ Stop exploration at level """

  MAXDEPTH=30

  def __init__(self, depth=MAXDEPTH, sort=lambda n:n[__CHILDREN__]):
    self.depth  = depth
    self.sort  = sort
    self.result = []

  def bfs(self, parent, predicate):
    # print(f"LevelNodesIterator.bfs: depth = {self.depth}: parent = {I.getName(parent)}")
    temp = queue.Queue()
    temp.put( (0, parent,) )
    while not temp.empty():
      level, node = temp.get()
      # print(f"LevelNodesIterator.bfs: level:{level} < depth:{self.depth}: node = {I.getName(node)}")
      if predicate(node):
        yield node
      if level < self.depth:
        for child in self.sort(node):
          temp.put( (level+1, child) )

  def dfs(self, parent, predicate):
    # print(f"LevelNodesIterator.dfs: parent = {I.getName(parent)}")
    if predicate(parent):
      yield parent
    yield from self._dfs(parent, predicate)

  def _dfs(self, parent, predicate, level=1):
    # print(f"LevelNodesIterator._dfs: level = {level} < depth = {self.depth}: parent = {I.getName(parent)}")
    for child in self.sort(parent):
      if predicate(child):
        yield child
      if level < self.depth:
        # Explore next level
        yield from self._dfs(child, predicate, level=level+1)

# --------------------------------------------------------------------------
class NodesWalker:
  """ Deep First Walker of pyTree """

  FORWARD  = lambda n:n[__CHILDREN__]
  BACKWARD = lambda n:reversed(n[__CHILDREN__])

  def __init__(self, parent: TreeNode, predicate: Callable[[TreeNode], bool],
                     method=NodesParser.DEFAULT, explore='deep', depth=0, sort=FORWARD,
                     caching: bool=False):
    self.parent    = parent
    self.predicate = predicate
    # Register default value
    self.method  = method
    self.explore = explore
    self.depth   = depth
    self.sort    = sort
    self.caching = caching
    # Internal
    self._parser = None
    self._cache  = []

  @property
  def parent(self):
    return self._parent

  @parent.setter
  def parent(self, node: TreeNode):
    if is_valid_node(node):
      self._parent = node
      self.clean()

  @property
  def predicate(self):
    return self._predicate

  @predicate.setter
  def predicate(self, predicate: Callable[[TreeNode], bool]):
    if callable(predicate):
      self._predicate = predicate
      self.clean()
    else:
      raise TypeError("predicate must be a callable function.")

  @property
  def method(self):
    return self._method

  @method.setter
  def method(self, value: str):
    if value in ['bfs', 'dfs']:
      self._method = value
      self.clean()
    else:
      raise ValueError("method must 'bfs' or 'dfs'.")

  @property
  def explore(self):
    return self._explore

  @explore.setter
  def explore(self, value: str):
    if value in ['deep', 'shallow']:
      self._explore = value
      self.clean()
    else:
      raise ValueError("method must 'deep' or 'shallow'.")

  @property
  def depth(self):
    return self._depth

  @depth.setter
  def depth(self, value: str):
    if isinstance(value, int) and value >= 0:
      self._depth = value
      self.clean()
    else:
      raise ValueError("depth must a integer >= 0.")

  @property
  def sort(self):
    return self._sort

  @sort.setter
  def sort(self, value):
    if callable(value):
      self._sort = value
      self.clean()
    else:
      raise TypeError("sort must be a callable function.")

  @property
  def caching(self):
    return self._caching

  @caching.setter
  def caching(self, value):
    if isinstance(value, bool):
      self._caching = value
    else:
      raise TypeError("caching must be a boolean.")

  @property
  def cache(self):
    return self._cache

  @property
  def parser(self):
    return self._parser

  # def __call__(self, parent=None, predicate=None, method=None, explore=None, depth=None, sort=None):
  #   """ Generator of nodes with predicate """
  #   if parent and parent != self.parent:
  #     self.parent = parent
  #   if predicate and predicate != self.predicate:
  #     self.predicate = predicate
  #   if method and method != self.method:
  #     self.method = method
  #   if explore and explore != self.explore:
  #     self.explore = explore
  #   if depth and depth != self.depth:
  #     self.depth = depth
  #   if sort and sort != self.sort:
  #     self.sort = sort

  def __call__(self):
    # if parser is not None:
    #   return parser(self._parent, self._predicate)
    if self.caching:
      if not bool(self._cache):
        # Generate list
        f = lambda n: self._cache.append(n)
        if self.explore == "shallow":
          self._parser = ShallowNodesParser(f, sort=self.sort)
        elif self.depth > 0:
          self._parser = LevelNodesParser(f, depth=self.depth, sort=self.sort)
        else:
          self._parser = NodesParser(f, sort=self.sort)
        parser = getattr(self._parser, self.method)
        parser(self._parent, self._predicate)
      return self._cache
    else:
      # Generate iterator
      if self.explore == "shallow":
        self._parser = ShallowNodesIterator(sort=self.sort)
      elif self.depth > 0:
        self._parser = LevelNodesIterator(depth=self.depth, sort=self.sort)
      else:
        self._parser = NodesIterator(sort=self.sort)
      parser = getattr(self._parser, self.method)
      return parser(self._parent, self._predicate)

  def clean(self):
    """ Reset the cache """
    self._cache = []

  def __del__(self):
    self.clean()

# --------------------------------------------------------------------------
# def getNodesFromPredicate(parent, predicate, method=NodeParser.DEFAULT, explore='deep', depth=0):
#   walker = NodesWalker(parent, predicate, method=method, explore=explore, depth=depth, caching=True)
#   return walker()
def getNodesFromPredicate(*args, **kwargs):
  kwargs['caching'] = True
  walker = NodesWalker(*args, **kwargs)
  return walker()

getShallowNodesFromPredicate = partial(getNodesFromPredicate, explore='shallow')

def create_get_children(predicate, nargs):
  def _get_nodes_from(parent, *args, **kwargs):
    pkwargs = dict([(narg, arg,) for narg, arg in zip(nargs, args)])
    return getNodesFromPredicate(parent, partial(predicate, **pkwargs), **kwargs)
  return _get_nodes_from

mesg = "Return a list of all child CGNS nodes"
create_functions(getNodesFromPredicate, create_get_children, "dfs",
  dict((k,v) for k,v in allfuncs.items() if k not in ['NameValueAndLabel']),
  mesg)

prefix = getNodesFromPredicate.__name__.replace('Predicate', '')
for what, item in dict((k,v) for k,v in allfuncs.items() if k not in ['NameValueAndLabel']).items():
  dwhat = ' '.join(PYU.camel_to_snake(what).split('_'))
  predicate, nargs = item

  # Generate getNodesFromNameP, getNodesFromValueP, ...
  funcname = f"getShallowNodesFrom{what}"
  func = create_get_children(predicate, nargs)
  func.__name__ = funcname
  func.__doc__  = """get {0} from a {1}""".format(mesg, dwhat)
  setattr(module_object, funcname, partial(func, method='dfs', explore='shallow'))
  # Generate get_nodes_from_name, get_nodes_from_value, ...
  funcname = PYU.camel_to_snake(funcname)
  # print(f"function.__name__ = {function.__name__}, funcname = {funcname}")
  func = create_get_children(predicate, nargs)
  func.__name__ = funcname
  func.__doc__  = """get {0} from a {1}""".format(mesg, dwhat)
  setattr(module_object, funcname, partial(func, method='dfs', explore='shallow'))

# --------------------------------------------------------------------------
# def iterNodesFromPredicate(parent, predicate, method=NodeParser.DEFAULT, explore='deep', depth=0):
#   walker = NodesWalker(parent, predicate, method=method, explore=explore, depth=depth, caching=False)
#   return walker()
def iterNodesFromPredicate(*args, **kwargs):
  kwargs['caching'] = False
  walker = NodesWalker(*args, **kwargs)
  return walker()

iterShallowNodesFromPredicate = partial(iterNodesFromPredicate, explore='shallow')

def create_iter_children(predicate, nargs):
  def _iter_children_from(parent, *args, **kwargs):
    pkwargs = dict([(narg, arg,) for narg, arg in zip(nargs, args)])
    return getNodesFromPredicate(parent, partial(predicate, **pkwargs), **kwargs)
  return _iter_children_from

mesg = "Return an iterator on all child CGNS nodes"
create_functions(iterNodesFromPredicate, create_iter_children, "dfs",
  dict((k,v) for k,v in allfuncs.items() if k not in ['NameValueAndLabel']),
  mesg)

for what, item in dict((k,v) for k,v in allfuncs.items() if k not in ['NameValueAndLabel']).items():
  dwhat = ' '.join(PYU.camel_to_snake(what).split('_'))
  predicate, nargs = item

  # Generate iterNodesFromNameP, iterNodesFromValueP, ...
  funcname = f"iterShallowNodesFrom{what}"
  func = create_iter_children(predicate, nargs)
  func.__name__ = funcname
  func.__doc__  = """iter {0} from a {1}""".format(mesg, dwhat)
  setattr(module_object, funcname, partial(func, method='dfs', explore='shallow'))
  # Generate get_nodes_from_name, get_nodes_from_value, ...
  funcname = PYU.camel_to_snake(funcname)
  # print(f"function.__name__ = {function.__name__}, funcname = {funcname}")
  func = create_iter_children(predicate, nargs)
  func.__name__ = funcname
  func.__doc__  = """iter {0} from a {1}""".format(mesg, dwhat)
  setattr(module_object, funcname, partial(func, method='dfs', explore='shallow'))

# --------------------------------------------------------------------------
def create_get_child(predicate, nargs, args):
  def _get_node_from(parent, **kwargs):
    pkwargs = dict([(narg, arg,) for narg, arg in zip(nargs, args)])
    try:
      return getNodeFromPredicate(parent, partial(predicate, **pkwargs), **kwargs)
    except CGNSNodeFromPredicateNotFoundError as e:
      print(f"For predicate : pkwargs = {pkwargs}", file=sys.stderr)
      raise e
  return _get_node_from

def create_get_all_children(predicate, nargs, args):
  def _get_all_children_from(parent, **kwargs):
    pkwargs = dict([(narg, arg,) for narg, arg in zip(nargs, args)])
    return getNodesFromPredicate(parent, partial(predicate, **pkwargs), **kwargs)
  return _get_all_children_from

label_with_specific_depth = ['CGNSBase_t',
  'BaseIterativeData_t', 'Zone_t', 'Family_t',
  'Elements_t',
  'FlowSolution_t',
  'GridCoordinates_t',
  'ZoneBC_t', 'BC_t',
  'ZoneGridConnectivity_t', 'GridConnectivity_t', 'GridConnectivity1to1_t', 'OversetHoles_t',
  'ZoneIterativeData_t',
  'ZoneSubRegion_t',
]
for label in filter(lambda i : i not in ['CGNSTree_t'], CGL.__members__):
  suffix = label[:-2]
  suffix = suffix.replace('CGNS', '')
  snake_name = PYU.camel_to_snake(suffix)

  # Generate getBase, getZone, ..., getInvalid
  func = create_get_child(match_label, ('label',), (label,))
  funcname = f"get{suffix}"
  func.__name__ = funcname
  func.__doc__  = """get the first CGNS node from CGNS label {0}.""".format(label)
  setattr(module_object, funcname, partial(func, method='bfs'))
  # Generate get_base, get_zone, ..., get_invalid
  func = create_get_child(match_label, ('label',), (label,))
  funcname = f"get_{snake_name}"
  func.__name__ = funcname
  func.__doc__  = """get the first CGNS node from CGNS label {0}.""".format(label)
  setattr(module_object, funcname, partial(func, method='bfs'))

  # Generate getAllBase, getAllZone, ..., getAllInvalid
  pargs = {'method':'bfs', 'explore':'shallow'} if label in label_with_specific_depth else {'method':'dfs'}
  func = create_get_all_children(match_label, ('label',), (label,))
  funcname = f"getAll{suffix}"
  func.__name__ = funcname
  func.__doc__  = """get all CGNS nodes from CGNS label {0}.""".format(label)
  setattr(module_object, funcname, partial(func, **pargs))
  # Generate get_bases, get_zones, ..., get_invalid
  func = create_get_all_children(match_label, ('label',), (label,))
  funcname = f"get_all_{snake_name}"
  func.__name__ = funcname
  func.__doc__  = """get all CGNS nodes from CGNS label {0}.""".format(label)
  setattr(module_object, funcname, partial(func, **pargs))

  for depth in range(1,MAXDEPTH+1):
    suffix = f"{suffix}_" if suffix[-1] in [str(i) for i in range(1,MAXDEPTH+1)] else suffix
    snake_name = PYU.camel_to_snake(suffix)

    # Generate getBase1, getZone1, ..., getInvalid1
    func = create_get_child(match_label, ('label',), (label,))
    funcname = f"get{suffix}{depth}"
    func.__name__ = funcname
    func.__doc__  = """get the first CGNS node from CGNS label {0} with depth={1}.""".format(label, depth)
    setattr(module_object, funcname, partial(func, method='bfs', depth=depth))
    # Generate get_base1, get_zone1, ..., get_invalid1
    func = create_get_child(match_label, ('label',), (label,))
    funcname = f"get_{snake_name}{depth}"
    func.__name__ = funcname
    func.__doc__  = """get the first CGNS node from CGNS label {0} with depth={1}.""".format(label, depth)
    setattr(module_object, funcname, partial(func, method='bfs', depth=depth))

    # Generate getAllBase1, getAllBase2, ..., getAllBase{MAXDEPTH}
    # Generate getAllZone1, getAllZone2, ..., getAllZone{MAXDEPTH}
    #   ...
    # Generate getAllInvalid1, getAllInvalid2, ..., getAllInvalid{MAXDEPTH}
    func = create_get_all_children(match_label, ('label',), (label,))
    funcname = f"getAll{suffix}{depth}"
    func.__name__ = funcname
    func.__doc__  = """get all CGNS nodes from CGNS label {0} with depth={1}""".format(label, depth)
    setattr(module_object, funcname, partial(func, method='dfs', depth=depth))
    # Generate get_all_base1, get_all_base2, ..., get_all_base{MAXDEPTH}
    # Generate get_all_zone1, get_all_zone2, ..., get_all_zone{MAXDEPTH}
    #   ...
    # Generate get_all_invalid1, get_all_invalid2, ..., get_all_invalid{MAXDEPTH}
    func = create_get_all_children(match_label, ('label',), (label,))
    funcname = f"get_all_{snake_name}{depth}"
    func.__name__ = funcname
    func.__doc__  = """get all CGNS nodes from CGNS label {0} with depth={1}""".format(label, depth)
    setattr(module_object, funcname, partial(func, method='dfs', depth=depth))

# --------------------------------------------------------------------------
def create_functions_name(create_function, name):
  snake_name = PYU.camel_to_snake(name)

  # Generate getAcoustic, ..., getCoordinateX, ..., getZoneSubRegionPointers
  funcname = f"get{name}"
  func = create_function(match_name, ('name',), (name,))
  func.__name__ = funcname
  func.__doc__  = """get the CGNS node with name {0}.""".format(name)
  setattr(module_object, funcname, partial(func, method='dfs'))
  # Generate get_acoustic, ..., get_coordinate_x, ..., get_zone_sub_region_pointers
  funcname = f"get_{snake_name}"
  func = create_function(match_name, ('name',), (name,))
  func.__name__ = funcname
  func.__doc__  = """get the CGNS node with name {0}.""".format(name)
  setattr(module_object, funcname, partial(func, method='dfs'))

  for depth in range(1,MAXDEPTH+1):
    # Generate getAcoustic1, ..., getCoordinateX1, ..., getZoneSubRegionPointers1
    funcname = f"get{name}{depth}"
    func = create_function(match_name, ('name',), (name,))
    func.__name__ = funcname
    func.__doc__  = """get the CGNS node with name {0} with depth={1}""".format(name, depth)
    setattr(module_object, funcname, partial(func, method='dfs', depth=depth))
    # Generate get_acoustic1, ..., get_coordinateX1, ..., get_zone_sub_region_pointers1
    funcname = f"get_{snake_name}{depth}"
    func = create_function(match_name, ('name',), (name,))
    func.__name__ = funcname
    func.__doc__  = """get the CGNS node with name {0} with depth={1}""".format(name, depth)
    setattr(module_object, funcname, partial(func, method='dfs', depth=depth))

# --------------------------------------------------------------------------
def create_get_child_name(predicate, nargs, args):
  def _get_child_name(parent, **kwargs):
    pkwargs = dict([(narg, arg,) for narg, arg in zip(nargs, args)])
    return getNodeFromPredicate(parent, partial(predicate, **pkwargs), **kwargs)
  return _get_child_name

# for cgns_type in filter(lambda i : i not in ['Null', 'UserDefined'] and not i.startswith('max'), CGK.PointSetType.__members__):
#   create_functions_name(create_get_child_name, cgns_type)

for name in filter(lambda i : not i.startswith('__') and not i.endswith('__'), dir(CGK.Name)):
  create_functions_name(create_get_child_name, name)

# --------------------------------------------------------------------------
def getFamily(parent, family_name):
  if isinstance(family_name, str):
    family_name = [family_name]
  for node in getNodesFromLabel(parent, 'Family_t'):
    if I.getName(family_name_node) in family_name:
      return node
  raise ValueError("Unable to find Family_t with name : {family_name}")

def getAdditionalFamily(parent, family_name):
  if isinstance(family_name, str):
    family_name = [family_name]
  for node in getNodesFromLabel(parent, 'Family_t'):
    if I.getName(family_name_node) in family_name:
      return node
  raise ValueError("Unable to find Family_t with name : {family_name}")

# --------------------------------------------------------------------------
def create_get_from_family(label, family_label):
  def _get_from_family(parent, family_name):
    if isinstance(family_name, str):
      family_name = [family_name]
    for node in getNodesFromLabel(parent, label):
      family_name_node = requestNodeFromLabel(node, family_label)
      if family_name_node and I.getValue(family_name_node) in family_name:
        return node
    raise ValueError(f"Unable to find {label} from family name : {family_name}")
  return _get_from_family

def create_get_all_from_family(label, family_label):
  def _get_all_from_family(parent, family_name):
    if isinstance(family_name, str):
      family_name = [family_name]
    nodes = []
    for node in getNodesFromLabel(parent, label):
      family_name_node = requestNodeFromLabel(node, family_label)
      if family_name_node and I.getValue(family_name_node) in family_name:
        nodes.append(node)
    return nodes
  return _get_all_from_family

def create_iter_all_from_family(label, family_label):
  def _iter_all_from_family(parent, family_name):
    if isinstance(family_name, str):
      family_name = [family_name]
    nodes = []
    for node in getNodesFromLabel(parent, label):
      family_name_node = requestNodeFromLabel(node, family_label)
      if family_name_node and I.getValue(family_name_node) in family_name:
        yield node
    return nodes
  return _iter_all_from_family

for family_label in ['Family_t', 'AdditionalFamily_t']:
  for label in ['Zone_t', 'BC_t', 'ZoneSubRegion_t', 'GridConnectivity_t', 'GridConnectivity1to1_t', 'OversetHoles_t']:

    funcname = f"get{label[:-2]}From{family_label[:-2]}"
    func = create_get_from_family(label, family_label)
    func.__name__ = funcname
    func.__doc__  = """Return a CGNS node from {0} with name {1}.""".format(family_label, name)
    setattr(module_object, funcname, func)
    setattr(module_object, PYU.camel_to_snake(funcname), func)

    funcname = f"getAll{label[:-2]}From{family_label[:-2]}"
    func = create_get_all_from_family(label, family_label)
    func.__name__ = funcname
    func.__doc__  = """Return a list of all CGNS node from {0} with name {1}.""".format(family_label, name)
    setattr(module_object, funcname, func)
    setattr(module_object, PYU.camel_to_snake(funcname), func)

    funcname = f"getAll{label[:-2]}From{family_label[:-2]}"
    func = create_iter_all_from_family(label, family_label)
    func.__name__ = funcname
    func.__doc__  = """Iterates on CGNS node from {0} with name {1}.""".format(family_label, name)
    setattr(module_object, funcname, func)
    setattr(module_object, PYU.camel_to_snake(funcname), func)

# --------------------------------------------------------------------------
def getGridConnectivitiesFromFamily(parent, family_name):
  if isinstance(family_name, str):
    family_name = [family_name]
  for node in iterNodesFromPredicate(parent, lambda n: n[__LABEL__] in ["GridConnectivity_t", "GridConnectivity1to1_t"]):
    family_name_node = requestNodeFromLabel(node, "FamilyName_t")
    if family_name_node and I.getValue(family_name_node) in family_name:
      return node
  raise ValueError("Unable to find GridConnectivity_t or GridConnectivity1to1_t from family name : {family_name}")

def getAllGridConnectivitiesFromFamily(parent, family_name):
  if isinstance(family_name, str):
    family_name = [family_name]
  nodes = []
  for node in iterNodesFromPredicate(parent, lambda n: n[__LABEL__] in ["GridConnectivity_t", "GridConnectivity1to1_t"]):
    family_name_node = requestNodeFromLabel(node, "FamilyName_t")
    if family_name_node and I.getValue(family_name_node) in family_name:
      nodes.append(node)
  return nodes

def iterAllGridConnectivitiesFromFamily(parent, family_name):
  if isinstance(family_name, str):
    family_name = [family_name]
  nodes = []
  for node in iterNodesFromPredicate(parent, lambda n: n[__LABEL__] in ["GridConnectivity_t", "GridConnectivity1to1_t"]):
    family_name_node = requestNodeFromLabel(node, "FamilyName_t")
    if family_name_node and I.getValue(family_name_node) in family_name:
      yield node
  return nodes

# --------------------------------------------------------------------------
def getGridConnectivitiesFromAdditionalFamily(parent, family_name):
  if isinstance(family_name, str):
    family_name = [family_name]
  for node in iterNodesFromPredicate(parent, lambda n: n[__LABEL__] in ["GridConnectivity_t", "GridConnectivity1to1_t"]):
    family_name_node = requestNodeFromLabel(node, "AdditionalFamilyName_t")
    if family_name_node and I.getValue(family_name_node) in family_name:
      return node
  raise ValueError("Unable to find GridConnectivity_t or GridConnectivity1to1_t from family name : {family_name}")

def getAllGridConnectivitiesFromAdditionalFamily(parent, family_name):
  if isinstance(family_name, str):
    family_name = [family_name]
  nodes = []
  for node in iterNodesFromPredicate(parent, lambda n: n[__LABEL__] in ["GridConnectivity_t", "GridConnectivity1to1_t"]):
    family_name_node = requestNodeFromLabel(node, "AdditionalFamilyName_t")
    if family_name_node and I.getValue(family_name_node) in family_name:
      nodes.append(node)
  return nodes

def iterAllGridConnectivitiesFromAdditionalFamily(parent, family_name):
  if isinstance(family_name, str):
    family_name = [family_name]
  nodes = []
  for node in iterNodesFromPredicate(parent, lambda n: n[__LABEL__] in ["GridConnectivity_t", "GridConnectivity1to1_t"]):
    family_name_node = requestNodeFromLabel(node, "AdditionalFamilyName_t")
    if family_name_node and I.getValue(family_name_node) in family_name:
      yield node
  return nodes

# --------------------------------------------------------------------------
def getNodesDispatch1(node, predicate):
  """ Interface to adapted getNodesFromXXX1 function depending of predicate type"""
  if isinstance(predicate, str):
    return getNodesFromLabel1(node, predicate) if is_valid_label(predicate) else getNodesFromName1(node, predicate)
  elif isinstance(predicate, CGK.Label):
    return getNodesFromLabel1(node, predicate.name)
  elif isinstance(predicate, np.ndarray):
    return getNodesFromValue1(node, predicate)
  elif callable(predicate):
    return getNodesFromPredicate1(node, predicate)
  else:
    raise TypeError("predicate must be a string for name, a numpy for value, a CGNS Label or a callable python function.")

# --------------------------------------------------------------------------
def iterNodesByMatching(root, predicates):
  """Generator following predicates, doing 1 level search using
  getNodesFromLabel1 or getNodesFromName1. Equivalent to
  (predicate = 'type1_t/name2/type3_t' or ['type1_t', 'name2', lambda n: I.getType(n) == CGL.type3_t.name] )
  for level1 in I.getNodesFromType1(root, type1_t):
    for level2 in I.getNodesFromName1(level1, name2):
      for level3 in I.getNodesFromType1(level2, type3_t):
        ...
  """
  predicate_list = []
  if isinstance(predicates, str):
    for predicate in predicates.split('/'):
      predicate_list.append(eval(predicate) if predicate.startswith('lambda') else predicate)
  elif isinstance(predicates, (list, tuple)):
    predicate_list = predicates
  else:
    raise TypeError("predicates must be a sequence or a path as with strings separated by '/'.")

  yield from iterNodesByMatching__(root, predicate_list)

def iterNodesByMatching__(root, predicate_list):
  if len(predicate_list) > 1:
    next_roots = getNodesDispatch1(root, predicate_list[0])
    for node in next_roots:
      yield from iterNodesByMatching__(node, predicate_list[1:])
  elif len(predicate_list) == 1:
    yield from getNodesDispatch1(root, predicate_list[0])

# --------------------------------------------------------------------------
def getNodesByMatching(root, predicates):
  """Generator following predicates, doing 1 level search using
  getNodesFromLabel1 or getNodesFromName1. Equivalent to
  (predicate = 'type1_t/name2/type3_t' or ['type1_t', 'name2', lambda n: I.getType(n) == CGL.type3_t.name] )
  for level1 in I.getNodesFromType1(root, type1_t):
    for level2 in I.getNodesFromName1(level1, name2):
      for level3 in I.getNodesFromType1(level2, type3_t):
        ...
  """
  predicate_list = []
  if isinstance(predicates, str):
    for predicate in predicates.split('/'):
      predicate_list.append(eval(predicate) if predicate.startswith('lambda') else predicate)
  elif isinstance(predicates, (list, tuple)):
    predicate_list = predicates
  else:
    raise TypeError("predicates must be a sequence or a path as with strings separated by '/'.")

  results = []
  return getNodesByMatching__(root, predicate_list, results)

def getNodesByMatching__(root, predicate_list, results):
  if len(predicate_list) > 1:
    next_roots = getNodesDispatch1(root, predicate_list[0])
    for node in next_roots:
      getNodesByMatching__(node, predicate_list[1:], results)
  elif len(predicate_list) == 1:
    results.append( getNodesDispatch1(root, predicate_list[0]) )

iter_children_by_matching = iterNodesByMatching
get_children_by_matching  = getNodesByMatching

# --------------------------------------------------------------------------
def iterNodesWithParentsByMatching(root, predicates):
  """Same than iterNodesByMatching, but return
  a tuple of size len(predicates) containing the node and its parents
  """
  predicate_list = []
  if isinstance(predicates, str):
    for predicate in predicates.split('/'):
      predicate_list.append(eval(predicate) if predicate.startswith('lambda') else predicate)
  elif isinstance(predicates, (list, tuple)):
    predicate_list = predicates
  else:
    raise TypeError("predicates must be a sequence or a path with strings separated by '/'.")

  yield from iterNodesWithParentsByMatching__(root, predicate_list)

def iterNodesWithParentsByMatching__(root, predicate_list):
  if len(predicate_list) > 1:
    next_roots = getNodesDispatch1(root, predicate_list[0])
    for node in next_roots:
      for subnode in iterNodesWithParentsByMatching__(node, predicate_list[1:]):
        yield (node, *subnode)
  elif len(predicate_list) == 1:
    nodes =  getNodesDispatch1(root, predicate_list[0])
    for node in nodes:
      yield (node,)

# --------------------------------------------------------------------------
def getNodesWithParentsByMatching(root, predicates):
  """Same than iterNodesByMatching, but return
  a tuple of size len(predicates) containing the node and its parents
  """
  predicate_list = []
  if isinstance(predicates, str):
    for predicate in predicates.split('/'):
      predicate_list.append(eval(predicate) if predicate.startswith('lambda') else predicate)
  elif isinstance(predicates, (list, tuple)):
    predicate_list = predicates
  else:
    raise TypeError("predicates must be a sequence or a path with strings separated by '/'.")

  results = []
  return getNodesWithParentsByMatching__(root, predicate_list, results)

def getNodesWithParentsByMatching__(root, predicate_list, results):
  if len(predicate_list) > 1:
    next_roots = getNodesDispatch1(root, predicate_list[0])
    for node in next_roots:
      for subnode in getNodesWithParentsByMatching__(node, predicate_list[1:], results):
        results.append( (node, *subnode,) )
  elif len(predicate_list) == 1:
    nodes =  getNodesDispatch1(root, predicate_list[0])
    for node in nodes:
      results.append( (node,) )

iter_children_with_parents_by_matching = iterNodesWithParentsByMatching
get_children_with_parents_by_matching  = getNodesWithParentsByMatching

# # --------------------------------------------------------------------------
# def rmChildrenFromPredicate(parent: TreeNode, predicate: Callable[[TreeNode], bool]) -> NoReturn:
#   to_delete = []
#   for num, node in enumerate(parent[__CHILDREN__]):
#     if predicate(node):
#       to_delete.append(num)
#   for num in reversed(to_delete):
#     del parent[__CHILDREN__][num]

# # --------------------------------------------------------------------------
# class NodesBackwardParser:

#   """ Stop exploration if something found at a level """

#   def __init__(self, func, sort=lambda n:n[__CHILDREN__]):
#     self.func = func
#     self.sort = sort


#   def dfs(self, parent, predicate):
#     # print(f"NodesBackwardFunctor.dfs: parent = {I.getName(parent)}")
#     if predicate(parent):
#       self.result.append(parent)
#     return self._dfs(parent, predicate)

#   def _dfs(self, parent, predicate):
#     # print(f"ShallowNodesBackwardFunctor.dfs: parent = {I.getName(parent)}")
#     results = []
#     for ichild, child in enumerate(parent[__CHILDREN__]):
#       if predicate(child):
#         results.append(ichild)
#       # Explore next level
#       self._dfs(child, predicate)
#     for ichild in self.sort(results):
#       self.func(parent[__CHILDREN__][ichild])

# # --------------------------------------------------------------------------
# class ShallowNodesBackwardFunctor:

#   """ Stop exploration if something found at a level """

#   def __init__(self, func, sort=lambda n:n[__CHILDREN__]):
#     self.func = func
#     self.sort = sort

#   def dfs(self, parent, predicate):
#     # print(f"ShallowNodesBackwardFunctor.dfs: parent = {I.getName(parent)}")
#     results = []
#     for ichild, child in enumerate(parent[__CHILDREN__]):
#       if predicate(child):
#         results.append(ichild)
#       else:
#         # Explore next level
#         self.dfs(child, predicate)
#     for ichild in self.sort(results):
#       self.func(parent[__CHILDREN__][ichild])

# # --------------------------------------------------------------------------
# class LevelNodesBackwardFunctor:

#   """ Stop exploration at level """

#   MAXDEPTH=30

#   def __init__(self, func, depth=MAXDEPTH, sort=lambda n:n[__CHILDREN__]):
#     self.func  = func
#     self.depth = depth
#     self.sort  = sort

#   def dfs(self, parent, predicate, level=1):
#     # print(f"LevelNodesBackwardFunctor._dfs: level = {level} < depth = {self.depth}: parent = {I.getName(parent)}")
#     results = []
#     for ichild, child in enumerate(parent[__CHILDREN__]):
#       if predicate(child):
#         results.append(ichild)
#       if level < self.depth:
#         # Explore next level
#         self.dfs(child, predicate, level=level+1)
#     for ichild in self.sort(results):
#       self.func(parent[__CHILDREN__][ichild])

# # --------------------------------------------------------------------------
# def rm_nodes_from_predicate(parent: TreeNode, predicate: Callable[[TreeNode], bool]) -> NoReturn:
#   to_delete = []
#   for ichild, child in enumerate(parent[__CHILDREN__]):
#     if predicate(child):
#       to_delete.append(ichild)
#     else:
#       rm_nodes_from_predicate(child, name)
#   for ichild in reversed(to_delete):
#     del parent[__CHILDREN__][ichild]

# --------------------------------------------------------------------------
@check_is_label('ZoneSubRegion_t', 0)
@check_is_label('Zone_t', 1)
def getSubregionExtent(sub_region_node, zone):
  """
  Return the path of the node (starting from zone node) related to sub_region_node
  node (BC, GC or itself)
  """
  if requestNodeFromName1(sub_region_node, "BCRegionName") is not None:
    for zbc, bc in iterNodesWithParentsByMatching(zone, "ZoneBC_t/BC_t"):
      if I.getName(bc) == I.getValue(requestNodeFromName1(sub_region_node, "BCRegionName")):
        return I.getName(zbc) + '/' + I.getName(bc)
  elif requestNodeFromName1(sub_region_node, "GridConnectivityRegionName") is not None:
    gc_pathes = ["ZoneGridConnectivity_t/GridConnectivity_t", "ZoneGridConnectivity_t/GridConnectivity1to1_t"]
    for gc_path in gc_pathes:
      for zgc, gc in iterNodesWithParentsByMatching(zone, gc_path):
        if I.getName(gc) == I.getValue(requestNodeFromName1(sub_region_node, "GridConnectivityRegionName")):
          return I.getName(zgc) + '/' + I.getName(gc)
  else:
    return I.getName(sub_region_node)

  raise ValueError("ZoneSubRegion {0} has no valid extent".format(I.getName(sub_region_node)))

def getDistribution(node, distri_name=None):
  """
  Starting from node, return the CGNS#Distribution node if distri_name is None
  or the value of the requested distribution if distri_name is not None
  """
  return I.getNodeFromPath(node, '/'.join([':CGNS#Distribution', distri_name])) if distri_name \
      else requestNodeFromName1(node, ':CGNS#Distribution')

def getGlobalNumbering(node, lngn_name=None):
  """
  Starting from node, return the CGNS#GlobalNumbering node if lngn_name is None
  or the value of the requested globalnumbering if lngn_name is not None
  """
  return I.getNodeFromPath(node, '/'.join([':CGNS#GlobalNumbering', lngn_name])) if lngn_name \
      else requestNodeFromName1(node, ':CGNS#GlobalNumbering')

# --------------------------------------------------------------------------
def newDistribution(distributions = dict(), parent=None):
  """
  Create and return a CGNSNode to be used to store distribution data
  Attach it to parent node if not None
  In addition, add distribution arrays specified in distributions dictionnary.
  distributions must be a dictionnary {DistriName : distri_array}
  """
  distri_node = I.newUserDefinedData(':CGNS#Distribution', None, parent)
  for name, value in distributions.items():
    I.newDataArray(name, value, parent=distri_node)
  return distri_node

def newGlobalNumbering(glob_numberings = dict(), parent=None):
  """
  Create and return a CGNSNode to be used to store distribution data
  Attach it to parent node if not None
  In addition, add global numbering arrays specified in glob_numberings dictionnary.
  glob_numberings must be a dictionnary {NumberingName : lngn_array}
  """
  lngn_node = I.newUserDefinedData(':CGNS#GlobalNumbering', None, parent)
  for name, value in glob_numberings.items():
    I.newDataArray(name, value, parent=lngn_node)
  return lngn_node

# --------------------------------------------------------------------------
request_node_from_predicate = requestNodeFromPredicate
get_node_from_predicate     = getNodeFromPredicate
get_nodes_from_predicate    = getNodesFromPredicate
get_family                  = getFamily

get_shallow_nodes_from_predicate  = getShallowNodesFromPredicate
iter_shallow_nodes_from_predicate = iterShallowNodesFromPredicate

get_grid_connectivities_from_family      = getGridConnectivitiesFromFamily
get_all_grid_connectivities_from_family  = getAllGridConnectivitiesFromFamily
iter_all_grid_connectivities_from_family = getAllGridConnectivitiesFromFamily

get_grid_connectivities_from_additional_family      = getGridConnectivitiesFromAdditionalFamily
get_all_grid_connectivities_from_additional_family  = getAllGridConnectivitiesFromAdditionalFamily
iter_all_grid_connectivities_from_additional_family = getAllGridConnectivitiesFromAdditionalFamily

get_node_dispatch1                    = getNodesDispatch1
iter_nodes_by_matching                = iterNodesByMatching
iter_nodes_with_parents_matching      = iterNodesWithParentsByMatching
get_subregion_extent                  = getSubregionExtent
get_distribution                      = getDistribution
get_global_numbering                  = getGlobalNumbering
new_distribution                      = newDistribution
new_global_numbering                  = newGlobalNumbering
