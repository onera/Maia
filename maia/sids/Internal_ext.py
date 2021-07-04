from typing import List, Optional, NoReturn, Union, Tuple, Callable, Any
from abc import abstractmethod
from functools import wraps
from functools import partial
import sys
import copy
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

def search_nodes_dispatch(node, predicate, **kwargs):
  """ Interface to adapted getNodesFromXXX1 function depending of predicate type"""
  if isinstance(predicate, str):
    if is_valid_label(predicate):
      walker = NodesWalker(node, partial(match_label, label=predicate), **kwargs)
      return walker()
    else:
      walker = NodesWalker(node, partial(match_name, name=predicate), **kwargs)
      return walker()
  elif isinstance(predicate, CGK.Label):
    walker = NodesWalker(node, partial(match_label, label=predicate.name), **kwargs)
    return walker()
  elif callable(predicate):
    walker = NodesWalker(node, predicate, **kwargs)
    return walker()
  elif isinstance(predicate, np.ndarray):
    walker = NodesWalker(node, partial(match_value, value=predicate), **kwargs)
    return walker()
  else:
    raise TypeError("predicate must be a string for name, a numpy for value, a CGNS Label or a callable python function.")

MAXDEPTH = 10

# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
class NodeParserBase:

  DEFAULT="bfs"
  MAXDEPTH=30

  def __init__(self, depth=MAXDEPTH, sort=lambda children:children):
    self.depth = depth
    self.sort  = sort

  @abstractmethod
  def bfs(self, parent, predicate):
    pass

  def dfs(self, parent, predicate):
    # print(f"NodeParserBase.dfs: parent = {I.getName(parent)}")
    if predicate(parent):
      return parent
    return self._dfs(parent, predicate)

  @abstractmethod
  def _dfs(self, parent, predicate, level=1):
    pass

# --------------------------------------------------------------------------
class NodeParser(NodeParserBase):

  def bfs(self, parent, predicate):
    # print(f"NodeParser.bfs: parent = {I.getName(parent)}")
    temp = queue.Queue()
    temp.put(parent)
    while not temp.empty():
      node = temp.get()
      # print(f"NodeParser.bfs: node = {I.getName(node)}")
      if predicate(node):
        return node
      for child in self.sort(node[__CHILDREN__]):
        temp.put(child)
    return None

  def _dfs(self, parent, predicate):
    # print(f"NodeParser._dfs: parent = {I.getName(parent)}")
    for child in self.sort(parent[__CHILDREN__]):
      if predicate(child):
        return child
      # Explore next level
      result = self._dfs(child, predicate)
      if result is not None:
        return result
    return None

class LevelNodeParser(NodeParserBase):

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
        for child in self.sort(node[__CHILDREN__]):
          temp.put( (level+1, child) )
    return None

  def _dfs(self, parent, predicate, level=1):
    # print(f"LevelNodeParser.dfs: level = {level} < depth = {self.depth}: parent = {I.getName(parent)}")
    for child in self.sort(parent[__CHILDREN__]):
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

  FORWARD  = lambda children:children
  BACKWARD = lambda children:reverse(children)

  def __init__(self, parent: TreeNode, predicate: Callable[[TreeNode], bool],
                     search=NodeParser.DEFAULT, depth=0, sort=FORWARD):
    self.parent    = parent
    self.predicate = predicate
    # Register default value
    self.search = search
    self.depth  = depth
    self.sort   = sort

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
  def search(self):
    return self._search

  @search.setter
  def search(self, value: str):
    if value in ['bfs', 'dfs']:
      self._search = value
    else:
      raise ValueError("search must 'bfs' or 'dfs'.")

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

  # def __call__(self, parent=None, predicate=None, search=None, explore=None, depth=None, sort=None):
  #   if parent and parent != self.parent:
  #     self.parent = parent
  #   if predicate and predicate != self.predicate:
  #     self.predicate = predicate
  #   if search and search != self.search:
  #     self.search = search
  #   if depth and depth != self.depth:
  #     self.depth = depth
  #   if sort and sort != self.sort:
  #     self.sort = sort

  def __call__(self):
    # Create parser
    self._parser = LevelNodeParser(depth=self.depth, sort=self.sort) if self.depth > 0 else NodeParser(sort=self.sort)
    func = getattr(self._parser, self.search)
    return func(self._parent, self._predicate)


# --------------------------------------------------------------------------
def requestNodeFromPredicate(*args, **kwargs):
  walker = NodeWalker(*args, **kwargs)
  return walker()

def create_request_child(predicate, nargs):
  def _get_request_from(parent, *args, **kwargs):
    pkwargs = dict([(narg, arg,) for narg, arg in zip(nargs, args)])
    return requestNodeFromPredicate(parent, partial(predicate, **pkwargs), **kwargs)
  return _get_request_from

generate_functions(requestNodeFromPredicate, create_request_child, "bfs", allfuncs,
  "Return a child CGNS node or None (if it is not found)")

# --------------------------------------------------------------------------
def getNodeFromPredicate(parent, predicate, *args, **kwargs):
  """ Return the list of first level childs of node matching a given predicate (callable function)"""
  node = requestNodeFromPredicate(parent, predicate, *args, **kwargs)
  if node is not None:
    return node
  default = kwargs.get('default', None)
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

generate_functions(getNodeFromPredicate, create_get_child, "bfs", allfuncs,
  "Return a child CGNS node or raise a CGNSNodeFromPredicateNotFoundError (if it is not found)")

# --------------------------------------------------------------------------
class NodesIteratorBase:

  MAXDEPTH=10
  DEFAULT='bfs'

  def __init__(self, depth=MAXDEPTH, sort=lambda children:children):
    self.depth = depth
    self.sort  = sort

  @abstractmethod
  def bfs(self, parent, predicate):
    pass

  def dfs(self, parent, predicate):
    # print(f"NodesIterator.dfs: parent = {I.getName(parent)}")
    if predicate(parent):
      yield parent
    yield from self._dfs(parent, predicate)

  @abstractmethod
  def _dfs(self, parent, predicate, level=1):
    pass

# --------------------------------------------------------------------------
class NodesIterator(NodesIteratorBase):

  """ Stop exploration if something found at a level """

  def bfs(self, parent, predicate):
    # print(f"ShallowNodesIterator.bfs: parent = {I.getName(parent)}")
    temp = queue.Queue()
    temp.put(parent)
    while not temp.empty():
      node = temp.get()
      # print(f"ShallowNodesIterator.bfs: node = {I.getName(node)}")
      if predicate(node):
        yield node
      for child in self.sort(node[__CHILDREN__]):
        temp.put(child)

  def _dfs(self, parent, predicate, level=1):
    # print(f"ShallowNodesIterator._dfs: parent = {I.getName(parent)}")
    for child in self.sort(parent[__CHILDREN__]):
      if predicate(child):
        yield child
      # Explore next level
      yield from self._dfs(child, predicate)

class ShallowNodesIterator(NodesIteratorBase):

  """ Stop exploration if something found at a level """

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
        for child in self.sort(node[__CHILDREN__]):
          temp.put(child)

  def _dfs(self, parent, predicate, level=1):
    # print(f"ShallowNodesIterator._dfs: parent = {I.getName(parent)}")
    for child in self.sort(parent[__CHILDREN__]):
      if predicate(child):
        yield child
      else:
        # Explore next level
        yield from self._dfs(child, predicate)

class LevelNodesIterator(NodesIteratorBase):

  """ Stop exploration until a limited level """

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
        for child in self.sort(node[__CHILDREN__]):
          temp.put( (level+1, child) )

  def _dfs(self, parent, predicate, level=1):
    # print(f"LevelNodesIterator._dfs: level = {level} < depth = {self.depth}: parent = {I.getName(parent)}")
    for child in self.sort(parent[__CHILDREN__]):
      if predicate(child):
        yield child
      if level < self.depth:
        # Explore next level
        yield from self._dfs(child, predicate, level=level+1)

class ShallowLevelNodesIterator(NodesIteratorBase):

  """ Stop exploration if something found at a level until a limited level """

  def bfs(self, parent, predicate):
    # print(f"ShallowLevelNodesIterator.bfs: parent = {I.getName(parent)}")
    temp = queue.Queue()
    temp.put( (0, parent,) )
    while not temp.empty():
      level, node = temp.get()
      # print(f"ShallowLevelNodesIterator.bfs: node = {I.getName(node)}")
      if predicate(node):
        yield node
      else:
        if level < self.depth:
          for child in self.sort(node[__CHILDREN__]):
            temp.put( (level+1, child) )

  def _dfs(self, parent, predicate, level=1):
    # print(f"ShallowLevelNodesIterator._dfs: parent = {I.getName(parent)}")
    for child in self.sort(parent[__CHILDREN__]):
      if predicate(child):
        yield child
      else:
        if level < self.depth:
          # Explore next level
          yield from self._dfs(child, predicate, level=level+1)

# --------------------------------------------------------------------------
class NodesWalker:
  """ Deep First Walker of pyTree """

  FORWARD  = lambda children:children
  BACKWARD = lambda children:reversed(children)

  def __init__(self, parent: TreeNode,
                     predicate: Callable[[TreeNode], bool],
                     search: str=NodesIterator.DEFAULT,
                     explore: str='deep',
                     depth: int=0,
                     sort=FORWARD,
                     caching: bool=False):
    """
    Hold all the manner to explore and parse the CGNS Tree

    Args:
        parent (TreeNode): CGNS node root searching
        predicate (Callable[[TreeNode], bool]): condition to select node
        search (str, optional): 'dfs' for Depth-First-Search or 'bfs' for Breath-First-Search
        explore (str, optional): 'deep' explore the whole tree or 'shallow' stop exploring node child when the node is found
        depth (int, optional): stop exploring after the limited depth
        sort (Callable[TreeNode], optional): parsing children sort
        caching (bool, optional): Results is store into a list. Avoid parsing next call(s).
    """
    self.parent    = parent
    self.predicate = predicate
    # Register default value
    self.search  = search
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
  def search(self):
    return self._search

  @search.setter
  def search(self, value: str):
    if value in ['bfs', 'dfs']:
      self._search = value
      self.clean()
    else:
      raise ValueError("search must 'bfs' or 'dfs'.")

  @property
  def explore(self):
    return self._explore

  @explore.setter
  def explore(self, value: str):
    if value in ['deep', 'shallow']:
      self._explore = value
      self.clean()
    else:
      raise ValueError("search must 'deep' or 'shallow'.")

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

  def _get_parser(self):
    if self.explore == "shallow":
      if self.depth > 0:
        parser = ShallowLevelNodesIterator(depth=self.depth, sort=self.sort)
      else:
        parser = ShallowNodesIterator(sort=self.sort)
    else:
      if self.depth > 0:
        parser = LevelNodesIterator(depth=self.depth, sort=self.sort)
      else:
        parser = NodesIterator(sort=self.sort)
    return parser

  def __call__(self):
    # Generate iterator
    self._parser = self._get_parser()
    walker = getattr(self._parser, self.search)
    iterator = walker(self._parent, self._predicate)
    if self.caching:
      if not bool(self._cache):
        self._cache = list(iterator)
      return self._cache
    else:
      return iterator

  def apply(self, f, *args, **kwargs):
    for n in self.__call__():
      f(n, *args, **kwargs)

  def clean(self):
    """ Reset the cache """
    self._cache = []

  def __del__(self):
    self.clean()

# --------------------------------------------------------------------------
#
#   get_nodes_from...
#
# --------------------------------------------------------------------------
def getNodesFromPredicate(*args, **kwargs) -> List[TreeNode]:
  """
  Alias to NodesWalker with caching=True. A list of found node(s) is created.

  Args:
      parent (TreeNode): CGNS node root searching
      predicate (Callable[[TreeNode], bool]): condition to select node
      search (str, optional): 'dfs' for Depth-First-Search or 'bfs' for Breath-First-Search
      explore (str, optional): 'deep' explore the whole tree or 'shallow' stop exploring node child when the node is found
      depth (int, optional): stop exploring after the limited depth
      sort (Callable[TreeNode], optional): parsing children sort
      caching (bool, optional): Force

  Returns:
      List[TreeNode]: Description

  """
  caching = kwargs.get('caching')
  if caching is not None and caching is False:
    print(f"Warning: getNodesFromPredicate forces caching to True.")
  kwargs['caching'] = True
  walker = NodesWalker(*args, **kwargs)
  return walker()

sgetNodesFromPredicate = partial(getNodesFromPredicate, explore='shallow')

def create_get_nodes(predicate, nargs):
  """
    Alias for getNodesFrom... generator. A list of found node(s) is created
  """
  def _get_nodes_from(parent, *args, **kwargs):
    pkwargs = dict([(narg, arg,) for narg, arg in zip(nargs, args)])
    return getNodesFromPredicate(parent, partial(predicate, **pkwargs), **kwargs)
  return _get_nodes_from

# Alias for getNodesFrom... generation
mesg = "Return a list of all child CGNS nodes"
generate_functions(getNodesFromPredicate, create_get_nodes, "dfs",
  dict((k,v) for k,v in allfuncs.items() if k not in ['NameValueAndLabel']),
  mesg)

# Alias for getNodesFrom... with shallow exploration and dfs traversing generation
prefix = getNodesFromPredicate.__name__.replace('Predicate', '')
for what, item in dict((k,v) for k,v in allfuncs.items() if k not in ['NameValueAndLabel']).items():
  dwhat = ' '.join(PYU.camel_to_snake(what).split('_'))
  predicate, nargs = item

  # Generate sgetNodesFrom{Name, Label, ...}
  funcname = f"sgetNodesFrom{what}"
  func = create_get_nodes(predicate, nargs)
  func.__name__ = funcname
  func.__doc__  = """get {0} from a {1}""".format(mesg, dwhat)
  setattr(module_object, funcname, partial(func, search='dfs', explore='shallow'))
  # Generate sget_nodes_from_{name, label, ...}
  funcname = PYU.camel_to_snake(funcname)
  func = create_get_nodes(predicate, nargs)
  func.__name__ = funcname
  func.__doc__  = """get {0} from a {1}""".format(mesg, dwhat)
  setattr(module_object, funcname, partial(func, search='dfs', explore='shallow'))

# --------------------------------------------------------------------------
#
#   iter_nodes_from...
#
# --------------------------------------------------------------------------
def iterNodesFromPredicate(*args, **kwargs):
  """
  Alias to NodesWalker with caching=False. Iterator is generated each time parsing is done.

  Args:
      parent (TreeNode): CGNS node root searching
      predicate (Callable[[TreeNode], bool]): condition to select node
      search (str, optional): 'dfs' for Depth-First-Search or 'bfs' for Breath-First-Search
      explore (str, optional): 'deep' explore the whole tree or 'shallow' stop exploring node child when the node is found
      depth (int, optional): stop exploring after the limited depth
      sort (Callable[TreeNode], optional): parsing children sort
      caching (bool, optional): Force

  Returns:
      TYPE: TreeNode generator/iterator

  """
  caching = kwargs.get('caching')
  if caching is not None and caching is True:
    print(f"Warning: iterNodesFromPredicate forces caching to False.")
  kwargs['caching'] = False
  walker = NodesWalker(*args, **kwargs)
  return walker()

siterNodesFromPredicate = partial(iterNodesFromPredicate, explore='shallow')

def create_iter_children(predicate, nargs):
  """
    Alias for iterNodesFrom... generator
  """
  def _iter_children_from(parent, *args, **kwargs):
    pkwargs = dict([(narg, arg,) for narg, arg in zip(nargs, args)])
    return iterNodesFromPredicate(parent, partial(predicate, **pkwargs), **kwargs)
  return _iter_children_from

# Alias for iterNodesFrom... generation
generate_functions(iterNodesFromPredicate, create_iter_children, "dfs",
  dict((k,v) for k,v in allfuncs.items() if k not in ['NameValueAndLabel']),
  "Return an iterator on all child CGNS nodes")

# Alias for iterNodesFrom... with shallow exploration and dfs traversing generation
for what, item in dict((k,v) for k,v in allfuncs.items() if k not in ['NameValueAndLabel']).items():
  dwhat = ' '.join(PYU.camel_to_snake(what).split('_'))
  predicate, nargs = item

  # Generate siterNodesFrom{Name, Label, ...}
  funcname = f"siterNodesFrom{what}"
  func = create_iter_children(predicate, nargs)
  func.__name__ = funcname
  func.__doc__  = """iter {0} from a {1}""".format(mesg, dwhat)
  setattr(module_object, funcname, partial(func, search='dfs', explore='shallow'))
  # Generate siter_nodes_from_{name, label, ...}
  funcname = PYU.camel_to_snake(funcname)
  func = create_iter_children(predicate, nargs)
  func.__name__ = funcname
  func.__doc__  = """iter {0} from a {1}""".format(mesg, dwhat)
  setattr(module_object, funcname, partial(func, search='dfs', explore='shallow'))

# --------------------------------------------------------------------------
#
#   get_{label}, iter{label}
#   get_all_{label}, iter_all_{label}
#   get_{label}{depth}, iter{label}{depth}
#   get_all_{label}{depth}, iter_all_{label}{depth}
#
# --------------------------------------------------------------------------
def create_get_child(predicate, nargs, args):
  """
    Alias for getNodesFrom... generator
  """
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

def create_iter_all_children(predicate, nargs, args):
  def _iter_all_children_from(parent, **kwargs):
    pkwargs = dict([(narg, arg,) for narg, arg in zip(nargs, args)])
    return iterNodesFromPredicate(parent, partial(predicate, **pkwargs), **kwargs)
  return _iter_all_children_from

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

  # Generate get{Base, Zone, ..}
  func = create_get_child(match_label, ('label',), (label,))
  funcname = f"get{suffix}"
  func.__name__ = funcname
  func.__doc__  = """get the first CGNS node from CGNS label {0}.""".format(label)
  setattr(module_object, funcname, partial(func, search='bfs'))
  # Generate get_{base, get_zone, ...}
  func = create_get_child(match_label, ('label',), (label,))
  funcname = f"get_{snake_name}"
  func.__name__ = funcname
  func.__doc__  = """get the first CGNS node from CGNS label {0}.""".format(label)
  setattr(module_object, funcname, partial(func, search='bfs'))

  pargs = {'search':'bfs', 'explore':'shallow'} if label in label_with_specific_depth else {'search':'dfs'}
  # Generate getAll{Base, Zone, ...}
  pargs['caching'] = True
  func = create_get_all_children(match_label, ('label',), (label,))
  funcname = f"getAll{suffix}"
  func.__name__ = funcname
  func.__doc__  = """get all CGNS nodes from CGNS label {0}.""".format(label)
  setattr(module_object, funcname, partial(func, **pargs))
  # Generate get_all_{base, zone, ...}
  func = create_get_all_children(match_label, ('label',), (label,))
  funcname = f"get_all_{snake_name}"
  func.__name__ = funcname
  func.__doc__  = """get all CGNS nodes from CGNS label {0}.""".format(label)
  setattr(module_object, funcname, partial(func, **pargs))

  # Generate iterAll{Base, Zone, ...}
  pargs['caching'] = False
  func = create_iter_all_children(match_label, ('label',), (label,))
  funcname = f"getAll{suffix}"
  func.__name__ = funcname
  func.__doc__  = """Get all CGNS nodes from CGNS label {0}.""".format(label)
  setattr(module_object, funcname, partial(func, **pargs))
  # Generate iter_all_{base, zone, ...}
  func = create_iter_all_children(match_label, ('label',), (label,))
  funcname = f"iter_all_{snake_name}"
  func.__name__ = funcname
  func.__doc__  = """Iterate on all CGNS nodes from CGNS label {0}.""".format(label)
  setattr(module_object, funcname, partial(func, **pargs))

  for depth in range(1,MAXDEPTH+1):
    suffix = f"{suffix}_" if suffix[-1] in [str(i) for i in range(1,MAXDEPTH+1)] else suffix
    snake_name = PYU.camel_to_snake(suffix)

    # Generate get{Base, Zone, ...}{depth}
    func = create_get_child(match_label, ('label',), (label,))
    funcname = f"get{suffix}{depth}"
    func.__name__ = funcname
    func.__doc__  = """Get the first CGNS node from CGNS label {0} with depth={1}.""".format(label, depth)
    setattr(module_object, funcname, partial(func, search='bfs', depth=depth))
    # Generate get_{base, zone, ...}{depth}
    func = create_get_child(match_label, ('label',), (label,))
    funcname = f"get_{snake_name}{depth}"
    func.__name__ = funcname
    func.__doc__  = """Get the first CGNS node from CGNS label {0} with depth={1}.""".format(label, depth)
    setattr(module_object, funcname, partial(func, search='bfs', depth=depth))

    # Generate getAll{Base, Zone, ...}{depth}
    func = create_get_all_children(match_label, ('label',), (label,))
    funcname = f"getAll{suffix}{depth}"
    func.__name__ = funcname
    func.__doc__  = """Get all CGNS nodes from CGNS label {0} with depth={1}""".format(label, depth)
    setattr(module_object, funcname, partial(func, search='dfs', depth=depth, caching=True))
    # Generate get_all_{base, zone, ...}{depth}
    func = create_get_all_children(match_label, ('label',), (label,))
    funcname = f"get_all_{snake_name}{depth}"
    func.__name__ = funcname
    func.__doc__  = """Get all CGNS nodes from CGNS label {0} with depth={1}""".format(label, depth)
    setattr(module_object, funcname, partial(func, search='dfs', depth=depth, caching=True))

    # Generate iterAll{Base, Zone, ...}{depth}
    func = create_iter_all_children(match_label, ('label',), (label,))
    funcname = f"iterAll{suffix}{depth}"
    func.__name__ = funcname
    func.__doc__  = """Iterate on all CGNS nodes from CGNS label {0} with depth={1}""".format(label, depth)
    setattr(module_object, funcname, partial(func, search='dfs', depth=depth, caching=False))
    # Generate get_all_{base, zone, ...}{depth}
    func = create_iter_all_children(match_label, ('label',), (label,))
    funcname = f"iter_all_{snake_name}{depth}"
    func.__name__ = funcname
    func.__doc__  = """Iterate on all CGNS nodes from CGNS label {0} with depth={1}""".format(label, depth)
    setattr(module_object, funcname, partial(func, search='dfs', depth=depth, caching=False))

# --------------------------------------------------------------------------
#
#   NodesWalkers
#
# --------------------------------------------------------------------------
class NodesWalkers:

  def __init__(self, parent, predicates, **kwargs):
    self.parent     = parent
    self.predicates = predicates
    self.kwargs     = kwargs
    self.ancestors  = kwargs.get('ancestors', False)
    if kwargs.get('ancestors'):
      kwargs.pop('ancestors')
    self._cache = []

  @property
  def parent(self):
    return self._parent

  @parent.setter
  def parent(self, node: TreeNode):
    if is_valid_node(node):
      self._parent = node
      self.clean()

  @property
  def predicates(self):
    return self._predicates

  @predicates.setter
  def predicates(self, value):
    self._predicates = []
    if isinstance(value, str):
      self._predicates = value.split('/')
      self.clean()
    elif isinstance(value, (list, tuple, dict)):
      self._predicates = value
      self.clean()
    else:
      raise TypeError("predicates must be a sequence of predicates or a path of name or label separated by '/'.")

  @property
  def ancestors(self):
    return self._ancestor

  @ancestors.setter
  def ancestors(self, value):
    if isinstance(value, bool):
      self._ancestor = value
    else:
      raise TypeError("ancestors must be a boolean.")

  @property
  def caching(self):
    return self.kwargs.get("caching", False)

  @caching.setter
  def caching(self, value):
    if isinstance(value, bool):
      self.kwargs['caching'] = value
    else:
      raise TypeError("caching must be a boolean.")

  @property
  def cache(self):
    return self._cache

  @property
  def parser(self):
    return self._parser

  def _deconv_kwargs(self):
    predicates = []; for_each = []
    for kwargs in self.predicates:
      lkwargs = {}
      for k,v in kwargs.items():
        if k == 'predicate':
          predicates.append(v)
        else:
          lkwargs[k] = v
      for_each.append(lkwargs)
    if len(predicates) != len(self.predicates):
      raise ValueError(f"Missing predicate.")
    return predicates, for_each

  def __call__(self):
    if self.ancestors:
      return self._parse_with_parents()
    else:
      return self._parse()

  def _parse_with_parents(self):
    if any([isinstance(kwargs, dict) for kwargs in self.predicates]):
      predicates, for_each = self._deconv_kwargs()
      for index, kwargs in enumerate(for_each):
        if kwargs.get('caching'):
          print(f"Warning: unable to activate caching for predicate at index {index}.")
          kwargs['caching'] = False
      if self.caching:
        if not bool(self._cache):
          self._cache = list(iter_nodes_from_predicates_with_parents_for_each__(self.parent, predicates, for_each))
        return self._cache
      else:
        return iter_nodes_from_predicates_with_parents_for_each__(self.parent, predicates, for_each)
    else:
      if self.caching:
        if not bool(self._cache):
          kwargs = copy.deepcopy(self.kwargs)
          kwargs['caching'] = False
          self._cache = list(iter_nodes_from_predicates_with_parents__(self.parent, self.predicates, **kwargs))
        return self._cache
      else:
        return iter_nodes_from_predicates_with_parents__(self.parent, self.predicates, **self.kwargs)

  def _parse(self):
    if any([isinstance(kwargs, dict) for kwargs in self.predicates]):
      predicates, for_each = self._deconv_kwargs()
      for index, kwargs in enumerate(for_each):
        if kwargs.get('caching'):
          print(f"Warning: unable to activate caching for predicate at index {index}.")
          kwargs['caching'] = False
      if self.caching:
        if not bool(self._cache):
          self._cache = list(iter_nodes_from_predicates_for_each__(self.parent, predicates, for_each))
        return self._cache
      else:
        return iter_nodes_from_predicates_for_each__(self.parent, predicates, for_each)
    else:
      if self.caching:
        if not bool(self._cache):
          kwargs = copy.deepcopy(self.kwargs)
          kwargs['caching'] = False
          self._cache = list(iter_nodes_from_predicates__(self.parent, self.predicates, **kwargs))
        return self._cache
      else:
        return iter_nodes_from_predicates__(self.parent, self.predicates, **self.kwargs)

  def apply(self, f, *args, **kwargs):
    for n in self.__call__():
      f(n, *args, **kwargs)

  def clean(self):
    """ Reset the cache """
    self._cache = []

  def __del__(self):
    self.clean()

def iter_nodes_from_predicates_for_each__(parent, predicates, for_each):
  # print("iter_nodes_from_predicates_for_each__")
  if len(predicates) > 1:
    for node in search_nodes_dispatch(parent, predicates[0], **for_each[0]):
      yield from iter_nodes_from_predicates_for_each__(node, predicates[1:], for_each[1:])
  elif len(predicates) == 1:
    yield from search_nodes_dispatch(parent, predicates[0], **for_each[0])

def iter_nodes_from_predicates__(parent, predicates, **kwargs):
  # print("iter_nodes_from_predicates__")
  if len(predicates) > 1:
    for node in search_nodes_dispatch(parent, predicates[0], **kwargs):
      yield from iter_nodes_from_predicates__(node, predicates[1:], **kwargs)
  elif len(predicates) == 1:
    yield from search_nodes_dispatch(parent, predicates[0], **kwargs)

def iter_nodes_from_predicates_with_parents_for_each__(parent, predicates, for_each):
  # print("iter_nodes_from_predicates_with_parents_for_each__")
  if len(predicates) > 1:
    for node in search_nodes_dispatch(parent, predicates[0], **for_each[0]):
      for subnode in iter_nodes_from_predicates_with_parents_for_each__(node, predicates[1:], for_each[1:]):
        yield (node, *subnode)
  elif len(predicates) == 1:
    for node in search_nodes_dispatch(parent, predicates[0], **for_each[0]):
      yield (node,)

def iter_nodes_from_predicates_with_parents__(parent, predicates, **kwargs):
  # print("iter_nodes_from_predicates_with_parents__")
  if len(predicates) > 1:
    for node in search_nodes_dispatch(parent, predicates[0], **kwargs):
      for subnode in iter_nodes_from_predicates_with_parents__(node, predicates[1:], **kwargs):
        yield (node, *subnode)
  elif len(predicates) == 1:
    for node in search_nodes_dispatch(parent, predicates[0], **kwargs):
      yield (node,)

# --------------------------------------------------------------------------
#
#   iter_nodes_from...s, alias for NodesWalkers
#
# --------------------------------------------------------------------------
def iterNodesFromPredicates(parent, predicates, **kwargs):
  """
  Alias to NodesWalkers with caching=False. Iterator is generated each time parsing is done.

  Args:
      parent (TreeNode): CGNS node root searching
      predicate (Callable[[TreeNode], bool]): condition to select node
      search (str, optional): 'dfs' for Depth-First-Search or 'bfs' for Breath-First-Search
      explore (str, optional): 'deep' explore the whole tree or 'shallow' stop exploring node child when the node is found
      depth (int, optional): stop exploring after the limited depth
      sort (Callable[TreeNode], optional): parsing children sort
      caching (bool, optional): Force

  Returns:
      TYPE: TreeNode generator/iterator

  """
  _predicates = []
  if isinstance(predicates, str):
    # for predicate in predicates.split('/'):
    #   _predicates.append(eval(predicate) if predicate.startswith('lambda') else predicate)
    _predicates = predicates.split('/')
  elif isinstance(predicates, (list, tuple)):
    _predicates = predicates
  else:
    raise TypeError("predicates must be a sequence or a path as with strings separated by '/'.")

  return iterNodesFromPredicates__(parent, _predicates, **kwargs)

def iterNodesFromPredicates__(*args, **kwargs):
  caching = kwargs.get('caching')
  if caching is not None and caching is True:
    print(f"Warning: iterNodesFromPredicates forces caching to False.")
  kwargs['caching'] = False
  walker = NodesWalkers(*args, **kwargs)
  return walker()

siterNodesFromPredicates = partial(iterNodesFromPredicates, explore='shallow')

def create_iter_childrens(predicate, nargs):
  """
    Alias for iterNodesFrom...s generator
  """
  def _iter_children_froms(parent, *args, **kwargs):
    pkwargs = dict([(narg, arg,) for narg, arg in zip(nargs, args)])
    return iterNodesFromPredicates(parent, partial(predicate, **pkwargs), **kwargs)
  return _iter_children_froms

# Alias for iterNodesFrom...s generation
generate_functions(iterNodesFromPredicates, create_iter_childrens, "dfs",
  dict((k,v) for k,v in allfuncs.items() if k not in ['NameValueAndLabel']),
  "Return an iterator on all CGNS nodes stifies the predicate(s)")

# Alias for iterNodesFrom...s with shallow exploration and dfs traversing generation
for what, item in dict((k,v) for k,v in allfuncs.items() if k not in ['NameValueAndLabel']).items():
  dwhat = ' '.join(PYU.camel_to_snake(what).split('_'))
  predicate, nargs = item

  # Generate siterNodesFrom{Name, Label, ...}s
  funcname = f"siterNodesFrom{what}s"
  func = create_iter_childrens(predicate, nargs)
  func.__name__ = funcname
  func.__doc__  = """iter {0} from a {1}""".format(mesg, dwhat)
  setattr(module_object, funcname, partial(func, search='dfs', explore='shallow'))
  # Generate siter_nodes_from_{name, label, ...}s
  funcname = PYU.camel_to_snake(funcname)
  func = create_iter_childrens(predicate, nargs)
  func.__name__ = funcname
  func.__doc__  = """iter {0} from a {1}""".format(mesg, dwhat)
  setattr(module_object, funcname, partial(func, search='dfs', explore='shallow'))

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

def iterNodesByMatching(root, predicates):
  """Generator following predicates, doing 1 level search using
  getNodesFromLabel1 or getNodesFromName1. Equivalent to
  (predicate = 'type1_t/name2/type3_t' or ['type1_t', 'name2', lambda n: I.getType(n) == CGL.type3_t.name] )
  for level1 in I.getNodesFromType1(root, type1_t):
    for level2 in I.getNodesFromName1(level1, name2):
      for level3 in I.getNodesFromType1(level2, type3_t):
        ...
  """
  _predicates = []
  if isinstance(predicates, str):
    # for predicate in predicates.split('/'):
    #   _predicates.append(eval(predicate) if predicate.startswith('lambda') else predicate)
    _predicates = predicates.split('/')
  elif isinstance(predicates, (list, tuple)):
    _predicates = predicates
  else:
    raise TypeError("predicates must be a sequence or a path as with strings separated by '/'.")

  walker = NodesWalkers(root, _predicates, search='dfs', depth=1)
  return walker()
  # yield from iterNodesByMatching__(root, _predicates)

# def iterNodesByMatching__(root, predicates):
#   if len(predicates) > 1:
#     next_roots = getNodesDispatch1(root, predicates[0])
#     for node in next_roots:
#       yield from iterNodesByMatching__(node, predicates[1:])
#   elif len(predicates) == 1:
#     yield from getNodesDispatch1(root, predicates[0])

getNodesByMatching = iterNodesByMatching

iter_children_by_matching = iterNodesByMatching
get_children_by_matching  = getNodesByMatching

# --------------------------------------------------------------------------
def iterNodesWithParentsByMatching(root, predicates):
  """Same than iterNodesByMatching, but return
  a tuple of size len(predicates) containing the node and its parents
  """
  _predicates = []
  if isinstance(predicates, str):
    for predicate in predicates.split('/'):
      _predicates.append(eval(predicate) if predicate.startswith('lambda') else predicate)
  elif isinstance(predicates, (list, tuple)):
    _predicates = predicates
  else:
    raise TypeError("predicates must be a sequence or a path with strings separated by '/'.")

  walker = NodesWalkers(root, _predicates, search='dfs', depth=1, ancestors=True)
  return walker()
  # yield from iterNodesWithParentsByMatching__(root, _predicates)

# def iterNodesWithParentsByMatching__(root, predicates):
#   if len(predicates) > 1:
#     for node in getNodesDispatch1(root, predicates[0]):
#       # print(f"node (from getNodesDispatch1) {len(predicates)} : {I.getName(node)}")
#       for subnode in iterNodesWithParentsByMatching__(node, predicates[1:]):
#         # print(f"subnode (from iterNodesWithParentsByMatching__) {len(predicates)} :     {[I.getName(n) for n in subnode]}")
#         yield (node, *subnode)
#   elif len(predicates) == 1:
#     for node in getNodesDispatch1(root, predicates[0]):
#       # print(f"node (from getNodesDispatch1) {len(predicates)}==1 : -> {I.getName(node)}")
#       yield (node,)

getNodesWithParentsByMatching = iterNodesWithParentsByMatching

iter_children_with_parents_by_matching = iterNodesWithParentsByMatching
get_children_with_parents_by_matching  = getNodesWithParentsByMatching

# --------------------------------------------------------------------------
def rmChildrenFromPredicate(parent: TreeNode, predicate: Callable[[TreeNode], bool]) -> NoReturn:
  results = []
  for ichild, child in enumerate(parent[__CHILDREN__]):
    if predicate(child):
      results.append(ichild)
  for ichild in reversed(results):
    del parent[__CHILDREN__][ichild]

rm_children_from_predicate = rmChildrenFromPredicate

def create_rm_children(predicate, nargs):
  def _rm_children_from(parent, *args):
    pkwargs = dict([(narg, arg,) for narg, arg in zip(nargs, args)])
    return rmChildrenFromPredicate(parent, partial(predicate, **pkwargs))
  return _rm_children_from

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

generate_rmkeep_functions(rmChildrenFromPredicate, create_rm_children, allfuncs,
  "Remove all direct child CGNS nodes")

# --------------------------------------------------------------------------
def keepChildrenFromPredicate(parent: TreeNode, predicate: Callable[[TreeNode], bool]) -> NoReturn:
  rmChildrenFromPredicate(parent, lambda n: not predicate(n))

keep_children_from_predicate = keepChildrenFromPredicate

def create_keep_children(predicate, nargs):
  def _keep_children_from(parent, *args):
    pkwargs = dict([(narg, arg,) for narg, arg in zip(nargs, args)])
    return keepChildrenFromPredicate(parent, partial(predicate, **pkwargs))
  return _keep_children_from

generate_rmkeep_functions(keepChildrenFromPredicate, create_keep_children, allfuncs,
  "Keep all direct child CGNS nodes")

# --------------------------------------------------------------------------
def rmNodesFromPredicate(parent, predicate, **kwargs):
  depth = kwargs.get('depth')
  if depth and not isinstance(depth, int):
    raise TypeError(f"depth must be an integer.")
  if depth and depth > 1:
    rmNodesFromPredicateWithLevel__(parent, predicate, depth)
  else:
    rmNodesFromPredicate__(parent, predicate)

def rmNodesFromPredicateWithLevel__(parent, predicate, depth, level=1):
  results = []
  for ichild, child in enumerate(parent[__CHILDREN__]):
    if predicate(child):
      results.append(ichild)
    else:
      if level < depth:
        rmNodesFromPredicateWithLevel__(child, predicate, depth, level=level+1)
  for ichild in reversed(results):
    del parent[__CHILDREN__][ichild]

def rmNodesFromPredicate__(parent, predicate):
  results = []
  for ichild, child in enumerate(parent[__CHILDREN__]):
    if predicate(child):
      results.append(ichild)
    else:
      rmNodesFromPredicate__(child, predicate)
  for ichild in reversed(results):
    del parent[__CHILDREN__][ichild]

rm_nodes_from_predicate = rmNodesFromPredicate

def create_rm_nodes(predicate, nargs):
  def _rm_nodes_from(parent, *args, **kwargs):
    pkwargs = dict([(narg, arg,) for narg, arg in zip(nargs, args)])
    return rmNodesFromPredicate(parent, partial(predicate, **pkwargs), **kwargs)
  return _rm_nodes_from

generate_functions(rmNodesFromPredicate, create_rm_nodes, "dfs",
  dict((k,v) for k,v in allfuncs.items() if k not in ['NameValueAndLabel']),
  "Remove all found child CGNS nodes")

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

iter_nodes_from_predicate  = iterNodesFromPredicate

sget_nodes_from_predicate  = sgetNodesFromPredicate
siter_nodes_from_predicate = siterNodesFromPredicate

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
