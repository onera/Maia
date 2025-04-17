from maia.pytree.typing import *
from maia.pytree.meta   import CGNSNodeNotFoundError

from maia.pytree.predicate     import auto_predicate, auto_predicates, \
                                      match_name, match_label, match_value, match_name_label


from .node_walker   import NodeWalker
from .nodes_walker  import NodesWalker
from .node_walkers  import NodeWalkers
from .nodes_walkers import NodesWalkers

# ---------------------------------------------------------------------------- #
# API for NodeWalker
# ------------------
# > Generic version
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

# > Specialized versions
def get_child_from_predicate(root:CGNSTree, predicate, **kwargs):
  """Specialization of get_node_predicate with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_node_from_predicate(root, predicate, **kwargs)

def get_node_from_name(root:CGNSTree, name:str, **kwargs):
  """Specialization of get_node_from_predicate with embedded predicate match_name"""
  return get_node_from_predicate(root, lambda n : match_name(n, name), **kwargs)
def get_child_from_name(root:CGNSTree, name:str, **kwargs):
  """Specialization of get_node_from_name with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_node_from_name(root, name, **kwargs)

def get_node_from_label(root:CGNSTree, label:str, **kwargs):
  """Specialization of get_node_from_predicate with embedded predicate match_label"""
  return get_node_from_predicate(root, lambda n : match_label(n, label), **kwargs)
def get_child_from_label(root:CGNSTree, label:str, **kwargs):
  """Specialization of get_node_from_label with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_node_from_label(root, label, **kwargs)

def get_node_from_value(root:CGNSTree, value, **kwargs):
  """Specialization of get_node_from_predicate with embedded predicate match_value"""
  return get_node_from_predicate(root, lambda n : match_value(n, value), **kwargs)
def get_child_from_value(root:CGNSTree, value, **kwargs):
  """Specialization of get_node_from_value with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_node_from_value(root, value, **kwargs)

def get_node_from_name_and_label(root:CGNSTree, name:str, label:str, **kwargs):
  """Specialization of get_node_from_predicate with embedded predicate match_name_label"""
  return get_node_from_predicate(root, lambda n : match_name_label(n, name, label), **kwargs)
def get_child_from_name_and_label(root:CGNSTree, name:str, label:str, **kwargs):
  """Specialization of get_node_from_name_and_label with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_node_from_name_and_label(root, name, label, **kwargs)

# > Generic version
def request_node_from_predicate(root:CGNSTree, predicate, *args, **kwargs) -> CGNSTree:
  """ Return the list of first level childs of node matching a given predicate (callable function)"""
  if (node := get_node_from_predicate(root, predicate, *args, **kwargs)) is not None:
    return node
  raise CGNSNodeNotFoundError(root, predicate)

# > Specialized versions
def request_child_from_predicate(root:CGNSTree, predicate, **kwargs):
  """Specialization of request_node_predicate with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return request_node_from_predicate(root, predicate, **kwargs)

def request_node_from_name(root:CGNSTree, name:str, **kwargs):
  """Specialization of request_node_from_predicate with embedded predicate match_name"""
  return request_node_from_predicate(root, lambda n : match_name(n, name), **kwargs)
def request_child_from_name(root:CGNSTree, name:str, **kwargs):
  """Specialization of request_node_from_name with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return request_node_from_name(root, name, **kwargs)

def request_node_from_label(root:CGNSTree, label:str, **kwargs):
  """Specialization of request_node_from_predicate with embedded predicate match_label"""
  return request_node_from_predicate(root, lambda n : match_label(n, label), **kwargs)
def request_child_from_label(root:CGNSTree, label:str, **kwargs):
  """Specialization of request_node_from_label with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return request_node_from_label(root, label, **kwargs)

def request_node_from_value(root:CGNSTree, value, **kwargs):
  """Specialization of request_node_from_predicate with embedded predicate match_value"""
  return request_node_from_predicate(root, lambda n : match_value(n, value), **kwargs)
def request_child_from_value(root:CGNSTree, value, **kwargs):
  """Specialization of request_node_from_value with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return request_node_from_value(root, value, **kwargs)

def request_node_from_name_and_label(root:CGNSTree, name:str, label:str, **kwargs):
  """Specialization of request_node_from_predicate with embedded predicate match_name_label"""
  return request_node_from_predicate(root, lambda n : match_name_label(n, name, label), **kwargs)
def request_child_from_name_and_label(root:CGNSTree, name:str, label:str, **kwargs):
  """Specialization of request_node_from_name_and_label with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return request_node_from_name_and_label(root, name, label, **kwargs)

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# API for NodesWalker
# -------------------
# > Generic version
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

# > Specialized versions
def get_children_from_predicate(root:CGNSTree, predicate, **kwargs):
  """Specialization of get_nodes_predicate with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_nodes_from_predicate(root, predicate, **kwargs)

def get_nodes_from_name(root:CGNSTree, name:str, **kwargs):
  """Specialization of get_nodes_from_predicate with embedded predicate match_name"""
  return get_nodes_from_predicate(root, lambda n : match_name(n, name), **kwargs)
def get_children_from_name(root:CGNSTree, name:str, **kwargs):
  """Specialization of get_nodes_from_name with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_nodes_from_name(root, name, **kwargs)

def get_nodes_from_label(root:CGNSTree, label:str, **kwargs):
  """Specialization of get_nodes_from_predicate with embedded predicate match_label"""
  return get_nodes_from_predicate(root, lambda n : match_label(n, label), **kwargs)
def get_children_from_label(root:CGNSTree, label:str, **kwargs):
  """Specialization of get_nodes_from_label with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_nodes_from_label(root, label, **kwargs)

def get_nodes_from_value(root:CGNSTree, value, **kwargs):
  """Specialization of get_nodes_from_predicate with embedded predicate match_value"""
  return get_nodes_from_predicate(root, lambda n : match_value(n, value), **kwargs)
def get_children_from_value(root:CGNSTree, value, **kwargs):
  """Specialization of get_nodes_from_value with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_nodes_from_value(root, value, **kwargs)

def get_nodes_from_name_and_label(root:CGNSTree, name:str, label:str, **kwargs):
  """Specialization of get_nodes_from_predicate with embedded predicate match_name_label"""
  return get_nodes_from_predicate(root, lambda n : match_name_label(n, name, label), **kwargs)
def get_children_from_name_and_label(root:CGNSTree, name:str, label:str, **kwargs):
  """Specialization of get_nodes_from_name_and_label with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_nodes_from_name_and_label(root, name, label, **kwargs)

# > Generic version
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

# > Specialized versions
def iter_children_from_predicate(root:CGNSTree, predicate, **kwargs):
  """Specialization of iter_nodes_predicate with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return iter_nodes_from_predicate(root, predicate, **kwargs)

def iter_nodes_from_name(root:CGNSTree, name:str, **kwargs):
  """Specialization of iter_nodes_from_predicate with embedded predicate match_name"""
  return iter_nodes_from_predicate(root, lambda n : match_name(n, name), **kwargs)
def iter_children_from_name(root:CGNSTree, name:str, **kwargs):
  """Specialization of iter_nodes_from_name with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return iter_nodes_from_name(root, name, **kwargs)

def iter_nodes_from_label(root:CGNSTree, label:str, **kwargs):
  """Specialization of iter_nodes_from_predicate with embedded predicate match_label"""
  return iter_nodes_from_predicate(root, lambda n : match_label(n, label), **kwargs)
def iter_children_from_label(root:CGNSTree, label:str, **kwargs):
  """Specialization of iter_nodes_from_label with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return iter_nodes_from_label(root, label, **kwargs)

def iter_nodes_from_value(root:CGNSTree, value, **kwargs):
  """Specialization of iter_nodes_from_predicate with embedded predicate match_value"""
  return iter_nodes_from_predicate(root, lambda n : match_value(n, value), **kwargs)
def iter_children_from_value(root:CGNSTree, value, **kwargs):
  """Specialization of iter_nodes_from_value with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return iter_nodes_from_value(root, value, **kwargs)

def iter_nodes_from_name_and_label(root:CGNSTree, name:str, label:str, **kwargs):
  """Specialization of iter_nodes_from_predicate with embedded predicate match_name_label"""
  return iter_nodes_from_predicate(root, lambda n : match_name_label(n, name, label), **kwargs)
def iter_children_from_name_and_label(root:CGNSTree, name:str, label:str, **kwargs):
  """Specialization of iter_nodes_from_name_and_label with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return iter_nodes_from_name_and_label(root, name, label, **kwargs)

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# API for NodeWalkers
# -------------------
# > Generic version
def get_node_from_predicates(root:CGNSTree, predicates, **kwargs) -> Optional[CGNSTree]:
  """ Return the first node in input tree matching the chain of predicates, or None

  The search can be fine-tuned with the following kwargs:

  - ``depth`` (int or pair of int): see :func:`get_node_from_predicate`
  - ``search`` (str): see :func:`get_node_from_predicate`
  - ``ancestors`` (bool): If ``False`` (default), keep only the terminal node.
    If ``True``, keep the intermediate nodes and return a tuple of nodes instead of a single node.

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

# > Specialized versions
def get_child_from_predicates(root:CGNSTree, predicates, **kwargs):
  """Specialization of get_node_predicates with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_node_from_predicates(root, predicates, **kwargs)

def get_node_from_names(root:CGNSTree, names:List[str], **kwargs):
  """Specialization of get_node_from_predicates with embedded predicates match_name"""
  predicates = [lambda n,name=name : match_name(n, name) for name in names]
  return get_node_from_predicates(root, predicates, **kwargs)
def get_child_from_names(root:CGNSTree, names:List[str], **kwargs):
  """Specialization of get_node_from_names with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_node_from_names(root, names, **kwargs)

def get_node_from_labels(root:CGNSTree, labels:List[str], **kwargs):
  """Specialization of get_node_from_predicates with embedded predicates match_label"""
  predicates = [lambda n,label=label : match_label(n, label) for label in labels]
  return get_node_from_predicates(root, predicates, **kwargs)
def get_child_from_labels(root:CGNSTree, labels:List[str], **kwargs):
  """Specialization of get_node_from_labels with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_node_from_labels(root, labels, **kwargs)

def get_node_from_values(root:CGNSTree, values, **kwargs):
  """Specialization of get_node_from_predicates with embedded predicates match_value"""
  predicates = [lambda n,value=value : match_value(n, value) for value in values]
  return get_node_from_predicates(root, predicates, **kwargs)
def get_child_from_values(root:CGNSTree, values, **kwargs):
  """Specialization of get_node_from_values with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_node_from_values(root, values, **kwargs)

def get_node_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], **kwargs):
  """Specialization of get_node_from_predicates with embedded predicates match_name_label"""
  assert len(names) == len(labels)
  predicates = [lambda n,name=name,label=label : match_name_label(n, name, label) for name,label in zip(names, labels)]
  return get_node_from_predicates(root, predicates, **kwargs)
def get_child_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], **kwargs):
  """Specialization of get_node_from_name_and_labels with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_node_from_name_and_labels(root, names, labels, **kwargs)
# ---------------------------------------------------------------------------- #



# ---------------------------------------------------------------------------- #
# API for NodesWalkers
# --------------------
# > Generic version
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

# > Specialized versions
def iter_children_from_predicates(root:CGNSTree, predicates, **kwargs):
  """Specialization of iter_nodes_predicates with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return iter_nodes_from_predicates(root, predicates, **kwargs)

def iter_nodes_from_names(root:CGNSTree, names:List[str], **kwargs):
  """Specialization of iter_nodes_from_predicates with embedded predicates match_name"""
  predicates = [lambda n,name=name : match_name(n, name) for name in names]
  return iter_nodes_from_predicates(root, predicates, **kwargs)
def iter_children_from_names(root:CGNSTree, names:List[str], **kwargs):
  """Specialization of iter_nodes_from_names with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return iter_nodes_from_names(root, names, **kwargs)

def iter_nodes_from_labels(root:CGNSTree, labels:List[str], **kwargs):
  """Specialization of iter_nodes_from_predicates with embedded predicates match_label"""
  predicates = [lambda n,label=label : match_label(n, label) for label in labels]
  return iter_nodes_from_predicates(root, predicates, **kwargs)
def iter_children_from_labels(root:CGNSTree, labels:List[str], **kwargs):
  """Specialization of iter_nodes_from_labels with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return iter_nodes_from_labels(root, labels, **kwargs)

def iter_nodes_from_values(root:CGNSTree, values, **kwargs):
  """Specialization of iter_nodes_from_predicates with embedded predicates match_value"""
  predicates = [lambda n,value=value : match_value(n, value) for value in values]
  return iter_nodes_from_predicates(root, predicates, **kwargs)
def iter_children_from_values(root:CGNSTree, values, **kwargs):
  """Specialization of iter_nodes_from_values with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return iter_nodes_from_values(root, values, **kwargs)

def iter_nodes_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], **kwargs):
  """Specialization of iter_nodes_from_predicates with embedded predicates match_name_label"""
  assert len(names) == len(labels)
  predicates = [lambda n,name=name,label=label : match_name_label(n, name, label) for name,label in zip(names, labels)]
  return iter_nodes_from_predicates(root, predicates, **kwargs)
def iter_children_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], **kwargs):
  """Specialization of iter_nodes_from_name_and_labels with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return iter_nodes_from_name_and_labels(root, names, labels, **kwargs)

# > Generic version
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

# > Specialized versions
def get_children_from_predicates(root:CGNSTree, predicates, **kwargs):
  """Specialization of get_nodes_predicates with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_nodes_from_predicates(root, predicates, **kwargs)

def get_nodes_from_names(root:CGNSTree, names:List[str], **kwargs):
  """Specialization of get_nodes_from_predicates with embedded predicates match_name"""
  predicates = [lambda n,name=name : match_name(n, name) for name in names]
  return get_nodes_from_predicates(root, predicates, **kwargs)
def get_children_from_names(root:CGNSTree, names:List[str], **kwargs):
  """Specialization of get_nodes_from_names with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_nodes_from_names(root, names, **kwargs)

def get_nodes_from_labels(root:CGNSTree, labels:List[str], **kwargs):
  """Specialization of get_nodes_from_predicates with embedded predicates match_label"""
  predicates = [lambda n,label=label : match_label(n, label) for label in labels]
  return get_nodes_from_predicates(root, predicates, **kwargs)
def get_children_from_labels(root:CGNSTree, labels:List[str], **kwargs):
  """Specialization of get_nodes_from_labels with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_nodes_from_labels(root, labels, **kwargs)

def get_nodes_from_values(root:CGNSTree, values, **kwargs):
  """Specialization of get_nodes_from_predicates with embedded predicates match_value"""
  predicates = [lambda n,value=value : match_value(n, value) for value in values]
  return get_nodes_from_predicates(root, predicates, **kwargs)
def get_children_from_values(root:CGNSTree, values, **kwargs):
  """Specialization of get_nodes_from_values with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_nodes_from_values(root, values, **kwargs)

def get_nodes_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], **kwargs):
  """Specialization of get_nodes_from_predicates with embedded predicates match_name_label"""
  assert len(names) == len(labels)
  predicates = [lambda n,name=name,label=label : match_name_label(n, name, label) for name,label in zip(names, labels)]
  return get_nodes_from_predicates(root, predicates, **kwargs)
def get_children_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], **kwargs):
  """Specialization of get_nodes_from_name_and_labels with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_nodes_from_name_and_labels(root, names, labels, **kwargs)

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Miscallaneous searches
def get_node_from_path(root:CGNSTree, path:str) -> Optional[CGNSTree]:
  """ Return the node in input tree matching given path, or None

  A path is a str containing a full list of names, separated by ``'/'``, leading
  to the node to select. Root name should not be included in path.
  Wildcards are not accepted in path.

  Args:
      root (CGNSTree): Tree is which the search is performed
      path (str): path of the node to select
  Returns:
    CGNSTree or None: Node found
  Example:
    >>> zone = PT.yaml.to_node('''
    ... Zone Zone_t:
    ...   ZoneBC ZoneBC_t:
    ...     BC BC_t "Null":
    ...       GridLocation GridLocation_t "Vertex":
    ... ''')
    >>> PT.get_node_from_path(zone, 'ZoneBC/BC/GridLocation')
    # Return node GridLocation
    >>> PT.get_node_from_path(zone, 'ZoneBC/BC/PointRange')
    # Return None
  """
  if path == '':
    return root
  names = path.split('/')
  node = root
  for name in names:
    try:
      node = next((c for c in node[2] if c[0] == name))
    except StopIteration:
      return
  return node

def request_node_from_path(root:CGNSTree, path:str) -> CGNSTree:
  if (node := get_node_from_path(root, path)) is not None:
    return node
  raise CGNSNodeNotFoundError(root, path)

def get_all_Zone_t(root:CGNSTree) -> List[CGNSTree]:
  """ Return the list of all the Zone_t nodes found in input tree

  This function is SIDS aware, and will only search nodes in relevant places.

  Args:
      root (CGNSTree): Tree is which the search is performed
  Returns:
    list of CGNSTree: Nodes found
  See also:
    This function has the iterator counterpart :func:`iter_all_Zone_t`

  Example:
    >>> tree = PT.yaml.to_cgns_tree('''
    ... BaseA CGNSBase_t:
    ...   Zone1 Zone_t:
    ...   Zone2 Zone_t:
    ... BaseB CGNSBase_t:
    ...   Zone3 Zone_t:
    ... ''')
    >>> [PT.get_name(n) for n in PT.iter_all_Zone_t(tree)]
    ['Zone1', 'Zone2', 'Zone3']
  """
  return list(iter_all_Zone_t(root))

def iter_all_Zone_t(root:CGNSTree) -> Iterator[CGNSTree]:
  root_label = root[3]
  if root_label == 'Zone_t':
    yield root
  elif root_label == 'CGNSBase_t':
    yield from iter_children_from_label(root, 'Zone_t')
  elif root_label == 'CGNSTree_t':
    yield from iter_children_from_labels(root, ['CGNSBase_t', 'Zone_t'])
  

def get_all_CGNSBase_t(root:CGNSTree) -> List[CGNSTree]:
  """ Return the list of all the CGNSBase_t nodes found in input tree

  This function is SIDS aware, and will only search nodes in relevant places.

  Args:
      root (CGNSTree): Tree is which the search is performed
  Returns:
    list of CGNSTree: Nodes found
  See also:
    This function has the iterator counterpart :func:`iter_all_CGNSBase_t`
  Example:
    >>> tree = PT.yaml.to_cgns_tree('''
    ... BaseA CGNSBase_t:
    ...   Zone1 Zone_t:
    ...   Zone2 Zone_t:
    ... BaseB CGNSBase_t:
    ...   Zone3 Zone_t:
    ... ''')
    >>> [PT.get_name(n) for n in PT.get_all_CGNSBase_t(tree)]
    ['BaseA', 'BaseB']
  """
  return list(iter_all_CGNSBase_t(root))

def iter_all_CGNSBase_t(root:CGNSTree) -> Iterator[CGNSTree]:
  if root[3] == 'CGNSBase_t':
    yield root
  elif root[3] == 'CGNSTree_t':
    yield from iter_children_from_label(root, 'CGNSBase_t')

def get_all_subsets(root:CGNSTree, filter_loc:Optional[List[str]]=None) -> List[CGNSTree]:
  """
  Search and collect all the subsets nodes found under root and the root
  itself if it is a subset
  If filter_loc list is not None, select only the subsets nodes of given
  GridLocation.
  """
  return list(iter_all_subsets(root,filter_loc))

def iter_all_subsets(root:CGNSTree, filter_loc:Optional[List[str]]=None) -> Iterator[CGNSTree]:
  """
  Search and iter on all the subsets nodes found under root
  If filter_loc list is not None, select only the subsets nodes of given
  GridLocation.
  """
  import maia.pytree as PT

  eligible_subset_paths = ['CGNSBase_t/Zone_t/ZoneBC_t/BC_t',
                           'CGNSBase_t/Zone_t/ZoneBC_t/BC_t/BCDataSet_t',
                           'CGNSBase_t/Zone_t/ZoneSubRegion_t',
                           'CGNSBase_t/Zone_t/DiscreteData_t',
                           'CGNSBase_t/Zone_t/FlowSolution_t',
                           'CGNSBase_t/Zone_t/ZoneGridConnectivity_t/GridConnectivity1to1_t',
                           'CGNSBase_t/Zone_t/ZoneGridConnectivity_t/GridConnectivity_t']

  root_label = PT.get_label(root)
  if root_label != 'CGNSTree_t':
    eligible_subset_paths = [path for path in eligible_subset_paths if root_label in path]

  subset_paths = []
  for path in eligible_subset_paths:
    path_split = path.split(root_label+'/')
    if len(path_split)>1:
      subset_paths.append(path_split[1])
    else:
      if filter_loc is None or PT.Subset.GridLocation(root) in filter_loc:
        pl_n = get_child_from_name(root, 'PointList')
        pr_n = get_child_from_name(root, 'PointRange')
        if (pl_n is not None) or (pr_n is not None):
          yield root

  get_location = lambda node, ancst: PT.Subset.GridLocation(node) if PT.get_label(node) != 'BCDataSet_t' \
                                                                  else PT.BCDataSet.GridLocation(node, ancst[-1])
  for path in subset_paths:
    for subset_n in iter_children_from_predicates(root, path, ancestors=True):
      ancestors, child = subset_n[:-1], subset_n[-1]
      if filter_loc is None or get_location(child, ancestors) in filter_loc:
        pl_n = get_child_from_name(child, 'PointList')
        pr_n = get_child_from_name(child, 'PointRange')
        if (pl_n is not None) or (pr_n is not None):
          yield child

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Searches returning pathes

def predicates_to_paths(root:CGNSTree, predicates) -> List[str]:
  """
  An utility function searching descendants matching predicates,
  and returning the path of these nodes (instead of the nodes themselves)
  """
  paths = []
  for nodes in iter_nodes_from_predicates(root, predicates, depth=[1,1], ancestors=True):
    paths.append('/'.join([n[0] for n in nodes]))
  return paths

def predicates_to_path(root:CGNSTree, predicates) -> Optional[str]:
  """
  An utility function searching descendants matching predicates,
  and returning the path of the first matching nodes (instead of the node itself)
  """
  nodes = get_node_from_predicates(root, predicates, depth=[1,1], ancestors=True)
  if None in nodes:
    return None
  else:
    return '/'.join([n[0] for n in nodes])
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
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
  """ Alias for get_node_from_predicates"""
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
# ---------------------------------------------------------------------------- #

