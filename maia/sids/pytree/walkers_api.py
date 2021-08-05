from functools import partial

from .node_walker   import NodeWalker
from .nodes_walker  import NodesWalker
from .nodes_walkers import NodesWalkers
from .compare       import CGNSNodeFromPredicateNotFoundError

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
  _predicates = []
  if isinstance(predicates, str):
    # for predicate in predicates.split('/'):
    #   _predicates.append(eval(predicate) if predicate.startswith('lambda') else predicate)
    _predicates = predicates.split('/')
  elif isinstance(predicates, (list, tuple)):
    _predicates = predicates
  else:
    raise TypeError("predicates must be a sequence or a path as with strings separated by '/'.")

  return iterNodesFromPredicates__(root, _predicates, **kwargs)

def iterNodesFromPredicates__(*args, **kwargs): #Duplicated
  caching = kwargs.get('caching')
  if caching is not None and caching is True:
    print(f"Warning: iterNodesFromPredicates forces caching to False.")
  kwargs['caching'] = False
  walker = NodesWalkers(*args, **kwargs)
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
  _predicates = []
  if isinstance(predicates, str):
    # for predicate in predicates.split('/'):
    #   _predicates.append(eval(predicate) if predicate.startswith('lambda') else predicate)
    _predicates = predicates.split('/')
  elif isinstance(predicates, (list, tuple)):
    _predicates = predicates
  else:
    raise TypeError("predicates must be a sequence or a path as with strings separated by '/'.")

  return getNodesFromPredicates__(root, _predicates, **kwargs)

def getNodesFromPredicates__(*args, **kwargs): #Duplicated
  caching = kwargs.get('caching')
  if caching is not None and caching is False:
    print(f"Warning: getNodesFromPredicates forces caching to True.")
  kwargs['caching'] = True
  walker = NodesWalkers(*args, **kwargs)
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
