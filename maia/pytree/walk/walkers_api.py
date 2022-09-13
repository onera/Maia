from .node_walker   import NodeWalker
from .nodes_walker  import NodesWalker
from .nodes_walkers import NodesWalkers
from .predicate     import auto_predicate, auto_predicates

from maia.pytree.compare import CGNSNodeFromPredicateNotFoundError

# ---------------------------------------------------------------------------- #
# API for NodeWalker
# ---------------------------------------------------------------------------- #
def get_node_from_predicate(root, predicate, **kwargs):
  _predicate = auto_predicate(predicate)
  walker = NodeWalker(root, _predicate, **kwargs)
  return walker()

def request_node_from_predicate(root, predicate, *args, **kwargs):
  """ Return the list of first level childs of node matching a given predicate (callable function)"""
  default = kwargs.pop('default', None)
  node = get_node_from_predicate(root, predicate, *args, **kwargs)
  if node is not None:
    return node
  if default:
    return default
  raise CGNSNodeFromPredicateNotFoundError(root, predicate)

# ---------------------------------------------------------------------------- #
# API for NodesWalker
# ---------------------------------------------------------------------------- #
def get_nodes_from_predicate(root, predicate, **kwargs):
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
  _predicate = auto_predicate(predicate)
  caching = kwargs.get('caching')
  if caching is not None and caching is False:
    print(f"Warning: get_nodes_from_predicate forces caching to True.")
  kwargs['caching'] = True

  walker = NodesWalker(root, _predicate, **kwargs)
  return walker()

def iter_nodes_from_predicate(root, predicate, **kwargs):
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
  _predicate = auto_predicate(predicate)
  caching = kwargs.get('caching')
  if caching is not None and caching is True:
    print(f"Warning: iter_nodes_from_predicate forces caching to False.")
  kwargs['caching'] = False

  walker = NodesWalker(root, _predicate, **kwargs)
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
  _predicates = auto_predicates(predicates)

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
  _predicates = auto_predicates(predicates)

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


