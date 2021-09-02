from functools import partial

from .node_walker   import NodeWalker
from .nodes_walker  import NodesWalker
from .nodes_walkers import NodesWalkers
from .compare       import CGNSNodeFromPredicateNotFoundError
from .predicate     import auto_predicate

def _convert_to_callable(predicates):
  """
  Convert a list a "convenience" predicates to a list a true callable predicates
  The list can also be given as a '/' separated string
  """
  _predicates = []
  if isinstance(predicates, str):
    _predicates = [auto_predicate(p) for p in predicates.split('/')]
  elif isinstance(predicates, (list, tuple)):
    _predicates = []
    for p in predicates:
      if isinstance(p, dict):
        #Create a new dict with a callable predicate
        _predicates.append({**p, 'predicate' : auto_predicate(p['predicate'])})
      else:
        _predicates.append(auto_predicate(p))
  else:
    raise TypeError("predicates must be a sequence or a path as with strings separated by '/'.")
  return _predicates

# ---------------------------------------------------------------------------- #
# API for NodeWalker
# ---------------------------------------------------------------------------- #
def request_node_from_predicate(*args, **kwargs):
  walker = NodeWalker(*args, **kwargs)
  return walker()

def get_node_from_predicate(root, predicate, *args, **kwargs):
  """ Return the list of first level childs of node matching a given predicate (callable function)"""
  node = request_node_from_predicate(root, predicate, *args, **kwargs)
  if node is not None:
    return node
  default = kwargs.get('default', None)
  if default and is_valid_node(default):
    return default
  raise CGNSNodeFromPredicateNotFoundError(root, predicate)

# ---------------------------------------------------------------------------- #
# API for NodesWalker
# ---------------------------------------------------------------------------- #
def get_nodes_from_predicate(*args, **kwargs):
  """
  Alias to NodesWalker with caching=True. A list of found node(s) is created.

  Args:
      root (TreeNode): CGNS node root searching
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
    print(f"Warning: get_nodes_from_predicate forces caching to True.")
  kwargs['caching'] = True

  walker = NodesWalker(*args, **kwargs)
  return walker()

def iter_nodes_from_predicate(*args, **kwargs):
  """
  Alias to NodesWalker with caching=False. Iterator is generated each time parsing is done.

  Args:
      root (TreeNode): CGNS node root searching
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
    print(f"Warning: iter_nodes_from_predicate forces caching to False.")
  kwargs['caching'] = False

  walker = NodesWalker(*args, **kwargs)
  return walker()


# ---------------------------------------------------------------------------- #
# API for NodesWalkers
# ---------------------------------------------------------------------------- #
def iter_nodes_from_predicates(root, predicates, **kwargs):
  """
  Alias to NodesWalkers with caching=False. Iterator is generated each time parsing is done.

  Args:
      root (TreeNode): CGNS node root searching
      predicate (Callable[[TreeNode], bool]): condition to select node
      search (str, optional): 'dfs' for Depth-First-Search or 'bfs' for Breath-First-Search
      explore (str, optional): 'deep' explore the whole tree or 'shallow' stop exploring node child when the node is found
      depth (int, optional): stop exploring after the limited depth
      sort (Callable[TreeNode], optional): parsing children sort
      caching (bool, optional): Force

  Returns:
      TYPE: TreeNode generator/iterator

  """
  _predicates = _convert_to_callable(predicates)

  caching = kwargs.get('caching')
  if caching is not None and caching is True:
    print(f"Warning: iter_nodes_from_predicates forces caching to False.")
  kwargs['caching'] = False

  walker = NodesWalkers(root, _predicates, **kwargs)
  return walker()

def get_nodes_from_predicates(root, predicates, **kwargs):
  """
  Alias to NodesWalkers with caching=True. A list of found node(s) is created.

  Args:
      root (TreeNode): CGNS node root searching
      predicate (Callable[[TreeNode], bool]): condition to select node
      search (str, optional): 'dfs' for Depth-First-Search or 'bfs' for Breath-First-Search
      explore (str, optional): 'deep' explore the whole tree or 'shallow' stop exploring node child when the node is found
      depth (int, optional): stop exploring after the limited depth
      sort (Callable[TreeNode], optional): parsing children sort
      caching (bool, optional): Force

  Returns:
      TYPE: TreeNode generator/iterator

  """
  _predicates = _convert_to_callable(predicates)

  caching = kwargs.get('caching')
  if caching is not None and caching is False:
    print(f"Warning: get_nodes_from_predicates forces caching to True.")
  kwargs['caching'] = True

  walker = NodesWalkers(root, _predicates, **kwargs)
  return walker()

# Aliases for legacy code -- using default argument deep instead of shallow for search

def getNodeFromPredicate(root, predicate, *args, **kwargs):
  """ Alias for get_node_from_predicate"""
  return get_node_from_predicate(root, predicate, *args, **kwargs)

def requestNodeFromPredicate(root, predicate, *args, **kwargs):
  """ Alias for request_node_from_predicate"""
  return request_node_from_predicate(root, predicate, *args, **kwargs)

def getNodesFromPredicate(root, predicate, *args, **kwargs):
  """ Alias for get_nodes_from_predicate (legacy), with default value 'deep' for search"""
  if 'explore' not in kwargs:
    kwargs['explore'] = 'deep'
  return get_nodes_from_predicate(root, predicate, *args, **kwargs)

def iterNodesFromPredicate(root, predicate, *args, **kwargs):
  """ Alias for iter_nodes_from_predicate (legacy), with default value 'deep' for search"""
  if 'explore' not in kwargs:
    kwargs['explore'] = 'deep'
  return iter_nodes_from_predicate(root, predicate, *args, **kwargs)

def getNodesFromPredicates(root, predicate, *args, **kwargs):
  """ Alias for get_nodes_from_predicates (legacy), with default value 'deep' for search"""
  if 'explore' not in kwargs:
    kwargs['explore'] = 'deep'
  return get_nodes_from_predicates(root, predicate, *args, **kwargs)

def iterNodesFromPredicates(root, predicate, *args, **kwargs):
  """ Alias for iter_nodes_from_predicates (legacy), with default value 'deep' for search"""
  if 'explore' not in kwargs:
    kwargs['explore'] = 'deep'
  return iter_nodes_from_predicates(root, predicate, *args, **kwargs)


