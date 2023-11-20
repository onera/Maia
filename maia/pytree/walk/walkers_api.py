from maia.pytree.typing import *

from .node_walker   import NodeWalker
from .nodes_walker  import NodesWalker
from .node_walkers  import NodeWalkers
from .nodes_walkers import NodesWalkers
from .predicate     import auto_predicate, auto_predicates

from maia.pytree.compare import CGNSNodeFromPredicateNotFoundError

# ---------------------------------------------------------------------------- #
# API for NodeWalker
# ---------------------------------------------------------------------------- #
def get_node_from_predicate(root:CGNSTree, predicate, **kwargs) -> Optional[CGNSTree]:
  """ Return the first node in input tree matching the given predicate, or None

  The search can be fine-tuned with the following kwargs:

  - ``depth`` (int or pair of int): limit the search between the depths *minD* and 
    *maxD*, 0 beeing the input node itself and None meaning unlimited.
    If a single int is provided, it is assigned to *maxD*.
    Defaults to ``(0,None)``.
  - ``search`` (str): use a Depth-First-Search (``'dfs'``) or
    Breath-First-Search (``'bfs'``) algorithm. Defaults to ``'dfs'``.
  
  Args:
    root (CGNSTree): Tree is which the search is performed
    predicate (callable): condition to select node, which must
      have the following signature: ``f(n:CGNSTree) -> bool``
    **kwargs: Additional options (see above)
  Returns:
    CGNSTree or None: Node found

  Note:
    This function admits the following shorcuts: 

    - :func:`get_node_from_name|label|value|name_and_label` (embedded predicate)
    - :func:`get_child_from_name|label|value|name_and_label` (embedded predicate + depth=[1,1])
  """
  _predicate = auto_predicate(predicate)
  walker = NodeWalker(root, _predicate, **kwargs)
  return walker()

def request_node_from_predicate(root:CGNSTree, predicate, *args, **kwargs) -> CGNSTree:
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
def get_nodes_from_predicate(root:CGNSTree, predicate, **kwargs) -> List[CGNSTree]:
  """ Return the list of all nodes in input tree matching the given predicate

  The search can be fine-tuned with the following kwargs:

  - ``depth`` (int or pair of int): see :func:`get_node_from_predicate`
  - ``search`` (str): see :func:`get_node_from_predicate`
  - ``explore`` (str): Explore the whole tree (``'deep'``) or stop exploring the current branch
    once predicate is satisfied (``'shallow'``). Defaults to ``'shallow'``.

  Args:
      root (CGNSTree): Tree is which the search is performed
      predicate (callable): condition to select node, which must
        have the following signature: ``f(n:CGNSTree) -> bool``
      **kwargs: Additional options (see above)
  Returns:
    list of CGNSTree: Nodes found

  Note:
    This function admits the following shorcuts: 

    - :func:`get_nodes_from_name|label|value|name_and_label` (embedded predicate)
    - :func:`get_children_from_name|label|value|name_and_label` (embedded predicate + depth=[1,1])
  """
  _predicate = auto_predicate(predicate)
  caching = kwargs.get('caching')
  if caching is not None and caching is False:
    print(f"Warning: get_nodes_from_predicate forces caching to True.")
  kwargs['caching'] = True

  walker = NodesWalker(root, _predicate, **kwargs)
  return walker()

def iter_nodes_from_predicate(root:CGNSTree, predicate, **kwargs) -> Iterator[CGNSTree]:
  """ Iterator version of :func:`get_nodes_from_predicate`

  Note:
    This function admits the following shorcuts: 

    - :func:`iter_nodes_from_name|label|value|name_and_label` (embedded predicate)
    - :func:`iter_children_from_name|label|value|name_and_label` (embedded predicate + depth=[1,1])
  """
  _predicate = auto_predicate(predicate)
  caching = kwargs.get('caching')
  if caching is not None and caching is True:
    print(f"Warning: iter_nodes_from_predicate forces caching to False.")
  kwargs['caching'] = False

  walker = NodesWalker(root, _predicate, **kwargs)
  return walker()

# ---------------------------------------------------------------------------- #
# API for NodeWalkers
# ---------------------------------------------------------------------------- #

def get_node_from_predicates(root:CGNSTree, predicates, **kwargs) -> Optional[CGNSTree]:
  """ Return the first node in input tree matching the chain of predicates, or None

  The search can be fine-tuned with the following kwargs:

  - ``depth`` (int or pair of int): see :func:`get_node_from_predicate`
  - ``search`` (str): see :func:`get_node_from_predicate`

  Args:
      root (CGNSTree): Tree is which the search is performed
      predicates (list of callable): conditions to select next node, each one
        having the following signature: ``f(n:CGNSTree) -> bool``
      **kwargs: Additional options (see above)
  Returns:
    CGNSTree or None: Node found

  Note:
    This function admits the following shorcuts: 

    - :func:`get_node_from_names|labels|values|name_and_labels` (embedded predicate)
    - :func:`get_child_from_names|labels|values|name_and_labels` (embedded predicate + depth=[1,1])
  """
  _predicates = auto_predicates(predicates)
  walker = NodeWalkers(root, _predicates, **kwargs)
  return walker()


# ---------------------------------------------------------------------------- #
# API for NodesWalkers
# ---------------------------------------------------------------------------- #
def iter_nodes_from_predicates(root:CGNSTree, predicates, **kwargs) -> Iterator[CGNSTree]:
  """ Iterator version of :func:`get_nodes_from_predicates`

  Note:
    This function admits the following shorcuts: 

    - :func:`iter_nodes_from_names|labels|values|name_and_labels` (embedded predicate)
    - :func:`iter_children_from_names|labels|values|name_and_labels` (embedded predicate + depth=[1,1])
  """
  _predicates = auto_predicates(predicates)

  caching = kwargs.get('caching')
  if caching is not None and caching is True:
    print(f"Warning: iter_nodes_from_predicates forces caching to False.")
  kwargs['caching'] = False

  walker = NodesWalkers(root, _predicates, **kwargs)
  return walker()

def get_nodes_from_predicates(root:CGNSTree, predicates, **kwargs) -> List[CGNSTree]:
  """ Return the list of all nodes in input tree matching the chain of predicates

  The search can be fine-tuned with the following kwargs:

  - ``depth`` (int or pair of int): see :func:`get_node_from_predicate`
  - ``search`` (str): see :func:`get_node_from_predicate`
  - ``explore`` (str): see :func:`get_nodes_from_predicate`
  - ``ancestors`` (bool): If ``False`` (default), keep only the terminal nodes.
    If ``True``, keep the intermediate nodes and return
    a list of tuples of nodes instead of a list of nodes.

  Args:
      root (CGNSTree): Tree is which the search is performed
      predicates (list of callable): conditions to select next node, each one
        having the following signature: ``f(n:CGNSTree) -> bool``
      **kwargs: Additional options (see above)
  Returns:
    list of CGNSTree: Nodes found

  Note:
    This function admits the following shorcuts: 

    - :func:`get_nodes_from_names|labels|values|name_and_labels` (embedded predicate)
    - :func:`get_children_from_names|labels|values|name_and_labels` (embedded predicate + depth=[1,1])
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

def getNodeFromPredicates(root, predicate, *args, **kwargs):
  """ Alias for get_node_from_predicates (legacy), with default value 'deep' for search"""
  if 'explore' not in kwargs:
    kwargs['explore'] = 'deep'
  return get_node_from_predicates(root, predicate, *args, **kwargs)

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


