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
def requestNodeFromPredicate(*args, **kwargs):
  walker = NodeWalker(*args, **kwargs)
  return walker()

def getNodeFromPredicate(root, predicate, *args, **kwargs):
  """ Return the list of first level childs of node matching a given predicate (callable function)"""
  node = requestNodeFromPredicate(root, predicate, *args, **kwargs)
  if node is not None:
    return node
  default = kwargs.get('default', None)
  if default and is_valid_node(default):
    return default
  raise CGNSNodeFromPredicateNotFoundError(root, predicate)

# ---------------------------------------------------------------------------- #
# API for NodesWalker
# ---------------------------------------------------------------------------- #
def getNodesFromPredicate(*args, **kwargs):
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
    print(f"Warning: getNodesFromPredicate forces caching to True.")
  kwargs['caching'] = True

  walker = NodesWalker(*args, **kwargs)
  return walker()

sgetNodesFromPredicate = partial(getNodesFromPredicate, explore='shallow')

def iterNodesFromPredicate(*args, **kwargs):
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
    print(f"Warning: iterNodesFromPredicate forces caching to False.")
  kwargs['caching'] = False

  walker = NodesWalker(*args, **kwargs)
  return walker()
siterNodesFromPredicate = partial(iterNodesFromPredicate, explore='shallow')

# ---------------------------------------------------------------------------- #
# API for NodesWalkers
# ---------------------------------------------------------------------------- #
def iterNodesFromPredicates(root, predicates, **kwargs):
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
    print(f"Warning: iterNodesFromPredicates forces caching to False.")
  kwargs['caching'] = False

  walker = NodesWalkers(root, _predicates, **kwargs)
  return walker()

siterNodesFromPredicates = partial(iterNodesFromPredicates, explore='shallow')

def getNodesFromPredicates(root, predicates, **kwargs):
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
    print(f"Warning: getNodesFromPredicates forces caching to True.")
  kwargs['caching'] = True

  walker = NodesWalkers(root, _predicates, **kwargs)
  return walker()

sgetNodesFromPredicates = partial(getNodesFromPredicates, explore='shallow')

# ---------------------------------------------------------------------------- #
# Shortcuts
# ---------------------------------------------------------------------------- #
get_node_from_predicate     = getNodeFromPredicate
request_node_from_predicate = requestNodeFromPredicate

get_nodes_from_predicate   = getNodesFromPredicate
iter_nodes_from_predicate  = iterNodesFromPredicate
sget_nodes_from_predicate  = sgetNodesFromPredicate
siter_nodes_from_predicate = siterNodesFromPredicate

get_nodes_from_predicates   = getNodesFromPredicates
iter_nodes_from_predicates  = iterNodesFromPredicates
sget_nodes_from_predicates  = sgetNodesFromPredicates
siter_nodes_from_predicates = siterNodesFromPredicates
