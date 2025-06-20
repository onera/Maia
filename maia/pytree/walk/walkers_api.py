from typing import overload, TypeVar
from maia.pytree.typing import *
from maia.pytree.meta   import CGNSNodeNotFoundError

from maia.pytree.pred import name_matches, label_matches, value_is
from .auto_pred import auto_predicate, auto_predicates


from .node_walker   import NodeWalker
from .nodes_walker  import NodesWalker
from .node_walkers  import NodeWalkers
from .nodes_walkers import NodesWalkers


# ---------------------------------------------------------------------------- #
# API for NodeWalker
# ------------------
# > Generic version
def get_node_from_predicate(root:CGNSTree, predicate:Predicate, **kwargs) -> Optional[CGNSTree]:
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
def get_child_from_predicate(root:CGNSTree, predicate:Predicate, **kwargs) -> Optional[CGNSTree]:
  """Specialization of get_node_predicate with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_node_from_predicate(root, predicate, **kwargs)

def get_node_from_name(root:CGNSTree, name:str, **kwargs) -> Optional[CGNSTree]:
  """Specialization of get_node_from_predicate with embedded predicate name_matches"""
  return get_node_from_predicate(root, name_matches(name), **kwargs)
def get_child_from_name(root:CGNSTree, name:str, **kwargs) -> Optional[CGNSTree]:
  """Specialization of get_node_from_name with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_node_from_name(root, name, **kwargs)

def get_node_from_label(root:CGNSTree, label:str, **kwargs) -> Optional[CGNSTree]:
  """Specialization of get_node_from_predicate with embedded predicate label_matches"""
  return get_node_from_predicate(root, label_matches(label), **kwargs)
def get_child_from_label(root:CGNSTree, label:str, **kwargs) -> Optional[CGNSTree]:
  """Specialization of get_node_from_label with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_node_from_label(root, label, **kwargs)

def get_node_from_value(root:CGNSTree, value, **kwargs) -> Optional[CGNSTree]:
  """Specialization of get_node_from_predicate with embedded predicate value_is"""
  return get_node_from_predicate(root, value_is(value), **kwargs)
def get_child_from_value(root:CGNSTree, value, **kwargs) -> Optional[CGNSTree]:
  """Specialization of get_node_from_value with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_node_from_value(root, value, **kwargs)

def get_node_from_name_and_label(root:CGNSTree, name:str, label:str, **kwargs) -> Optional[CGNSTree]:
  """Specialization of get_node_from_predicate with embedded predicate match_name_label"""
  return get_node_from_predicate(root, name_matches(name) & label_matches(label), **kwargs)
def get_child_from_name_and_label(root:CGNSTree, name:str, label:str, **kwargs) -> Optional[CGNSTree]:
  """Specialization of get_node_from_name_and_label with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_node_from_name_and_label(root, name, label, **kwargs)

# > Generic version
def find_node_from_predicate(root:CGNSTree, predicate:Predicate, *args, **kwargs) -> CGNSTree:
  """ Return the list of first level childs of node matching a given predicate (callable function)"""
  if (node := get_node_from_predicate(root, predicate, *args, **kwargs)) is not None:
    return node
  raise CGNSNodeNotFoundError(root, predicate)

# > Specialized versions
def find_child_from_predicate(root:CGNSTree, predicate:Predicate, **kwargs) -> CGNSTree:
  """Specialization of find_node_predicate with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return find_node_from_predicate(root, predicate, **kwargs)

def find_node_from_name(root:CGNSTree, name:str, **kwargs) -> CGNSTree:
  """Specialization of find_node_from_predicate with embedded predicate name_matches"""
  return find_node_from_predicate(root, name_matches(name), **kwargs)
def find_child_from_name(root:CGNSTree, name:str, **kwargs) -> CGNSTree:
  """Specialization of find_node_from_name with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return find_node_from_name(root, name, **kwargs)

def find_node_from_label(root:CGNSTree, label:str, **kwargs) -> CGNSTree:
  """Specialization of find_node_from_predicate with embedded predicate label_matches"""
  return find_node_from_predicate(root, label_matches(label), **kwargs)
def find_child_from_label(root:CGNSTree, label:str, **kwargs) -> CGNSTree:
  """Specialization of find_node_from_label with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return find_node_from_label(root, label, **kwargs)

def find_node_from_value(root:CGNSTree, value, **kwargs) -> CGNSTree:
  """Specialization of find_node_from_predicate with embedded predicate value_is"""
  return find_node_from_predicate(root, value_is(value), **kwargs)
def find_child_from_value(root:CGNSTree, value, **kwargs) -> CGNSTree:
  """Specialization of find_node_from_value with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return find_node_from_value(root, value, **kwargs)

def find_node_from_name_and_label(root:CGNSTree, name:str, label:str, **kwargs) -> CGNSTree:
  """Specialization of find_node_from_predicate with embedded predicate match_name_label"""
  return find_node_from_predicate(root, name_matches(name) & label_matches(label), **kwargs)
def find_child_from_name_and_label(root:CGNSTree, name:str, label:str, **kwargs) -> CGNSTree:
  """Specialization of find_node_from_name_and_label with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return find_node_from_name_and_label(root, name, label, **kwargs)

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# API for NodesWalker
# -------------------
# > Generic version
def get_nodes_from_predicate(root:CGNSTree, predicate:Predicate, **kwargs) -> List[CGNSTree]:
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
def get_children_from_predicate(root:CGNSTree, predicate:Predicate, **kwargs) -> List[CGNSTree]:
  """Specialization of get_nodes_predicate with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_nodes_from_predicate(root, predicate, **kwargs)

def get_nodes_from_name(root:CGNSTree, name:str, **kwargs) -> List[CGNSTree]:
  """Specialization of get_nodes_from_predicate with embedded predicate name_matches"""
  return get_nodes_from_predicate(root, name_matches(name), **kwargs)
def get_children_from_name(root:CGNSTree, name:str, **kwargs) -> List[CGNSTree]:
  """Specialization of get_nodes_from_name with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_nodes_from_name(root, name, **kwargs)

def get_nodes_from_label(root:CGNSTree, label:str, **kwargs) -> List[CGNSTree]:
  """Specialization of get_nodes_from_predicate with embedded predicate label_matches"""
  return get_nodes_from_predicate(root, label_matches(label), **kwargs)
def get_children_from_label(root:CGNSTree, label:str, **kwargs) -> List[CGNSTree]:
  """Specialization of get_nodes_from_label with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_nodes_from_label(root, label, **kwargs)

def get_nodes_from_value(root:CGNSTree, value, **kwargs) -> List[CGNSTree]:
  """Specialization of get_nodes_from_predicate with embedded predicate value_is"""
  return get_nodes_from_predicate(root, value_is(value), **kwargs)
def get_children_from_value(root:CGNSTree, value, **kwargs) -> List[CGNSTree]:
  """Specialization of get_nodes_from_value with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_nodes_from_value(root, value, **kwargs)

def get_nodes_from_name_and_label(root:CGNSTree, name:str, label:str, **kwargs) -> List[CGNSTree]:
  """Specialization of get_nodes_from_predicate with embedded predicate match_name_label"""
  return get_nodes_from_predicate(root, name_matches(name) & label_matches(label), **kwargs)
def get_children_from_name_and_label(root:CGNSTree, name:str, label:str, **kwargs) -> List[CGNSTree]:
  """Specialization of get_nodes_from_name_and_label with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_nodes_from_name_and_label(root, name, label, **kwargs)

# > Generic version
def iter_nodes_from_predicate(root:CGNSTree, predicate:Predicate, **kwargs) -> Iterator[CGNSTree]:
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
def iter_children_from_predicate(root:CGNSTree, predicate:Predicate, **kwargs) -> Iterator[CGNSTree]:
  """Specialization of iter_nodes_predicate with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return iter_nodes_from_predicate(root, predicate, **kwargs)

def iter_nodes_from_name(root:CGNSTree, name:str, **kwargs) -> Iterator[CGNSTree]:
  """Specialization of iter_nodes_from_predicate with embedded predicate name_matches"""
  return iter_nodes_from_predicate(root, name_matches(name), **kwargs)
def iter_children_from_name(root:CGNSTree, name:str, **kwargs) -> Iterator[CGNSTree]:
  """Specialization of iter_nodes_from_name with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return iter_nodes_from_name(root, name, **kwargs)

def iter_nodes_from_label(root:CGNSTree, label:str, **kwargs) -> Iterator[CGNSTree]:
  """Specialization of iter_nodes_from_predicate with embedded predicate label_matches"""
  return iter_nodes_from_predicate(root, label_matches(label), **kwargs)
def iter_children_from_label(root:CGNSTree, label:str, **kwargs) -> Iterator[CGNSTree]:
  """Specialization of iter_nodes_from_label with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return iter_nodes_from_label(root, label, **kwargs)

def iter_nodes_from_value(root:CGNSTree, value, **kwargs) -> Iterator[CGNSTree]:
  """Specialization of iter_nodes_from_predicate with embedded predicate value_is"""
  return iter_nodes_from_predicate(root, value_is(value), **kwargs)
def iter_children_from_value(root:CGNSTree, value, **kwargs) -> Iterator[CGNSTree]:
  """Specialization of iter_nodes_from_value with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return iter_nodes_from_value(root, value, **kwargs)

def iter_nodes_from_name_and_label(root:CGNSTree, name:str, label:str, **kwargs) -> Iterator[CGNSTree]:
  """Specialization of iter_nodes_from_predicate with embedded predicate match_name_label"""
  return iter_nodes_from_predicate(root, name_matches(name) & label_matches(label), **kwargs)
def iter_children_from_name_and_label(root:CGNSTree, name:str, label:str, **kwargs) -> Iterator[CGNSTree]:
  """Specialization of iter_nodes_from_name_and_label with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return iter_nodes_from_name_and_label(root, name, label, **kwargs)

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# API for NodeWalkers
# -------------------
# > Generic version

# For typing : overload to indicate if we return one or several nodes, depending of ancestors flag
@overload
def get_node_from_predicates(root:CGNSTree, predicates:Predicates, ancestors:Literal[True], **kwargs) -> Optional[Tuple[CGNSTree, ...]]: ...
@overload
def get_node_from_predicates(root:CGNSTree, predicates:Predicates, ancestors:Literal[False], **kwargs) -> Optional[CGNSTree]: ...
@overload
def get_node_from_predicates(root:CGNSTree, predicates:Predicates, **kwargs) -> Optional[CGNSTree]: ...

def get_node_from_predicates(root:CGNSTree, predicates:Predicates, ancestors:bool=False, **kwargs) -> Union[None, CGNSTree, Tuple[CGNSTree, ...]]:
  """ Return the first node in input tree matching the chain of predicates, or None

  The search can be fine-tuned with the following kwargs:

  - ``depth`` (int or pair of int): see :func:`get_node_from_predicate`
  - ``search`` (str): see :func:`get_node_from_predicate`

  Args:
      root (CGNSTree): Tree is which the search is performed
      predicates (list of callable): conditions to select next node, each one
        having the following signature: ``f(n:CGNSTree) -> bool``
      ancestors (bool): If ``False`` (default), keep only the terminal node.
        If ``True``, keep the intermediate nodes and return a tuple of nodes instead of a single node.
      **kwargs: Additional options (see above)
  Returns:
    CGNSTree or None: Node found

  Note:
    This function admits the following shorcuts: 

    - :func:`get_node_from_names|labels|values|name_and_labels` (embedded predicate)
    - :func:`get_child_from_names|labels|values|name_and_labels` (embedded predicate + depth=[1,1])
  """
  _predicates = auto_predicates(predicates)
  kwargs['ancestors'] = ancestors
  walker = NodeWalkers(root, _predicates, **kwargs)
  return walker()

# > Specialized versions
@overload
def get_child_from_predicates(root:CGNSTree, predicates:Predicates, ancestors:Literal[True], **kwargs) -> Optional[Tuple[CGNSTree, ...]]: ...
@overload
def get_child_from_predicates(root:CGNSTree, predicates:Predicates, ancestors:Literal[False], **kwargs) -> Optional[CGNSTree]: ...
@overload
def get_child_from_predicates(root:CGNSTree, predicates:Predicates, **kwargs) -> Optional[CGNSTree]: ...

def get_child_from_predicates(root:CGNSTree, predicates:Predicates, ancestors=False, **kwargs):
  """Specialization of get_node_predicates with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_node_from_predicates(root, predicates, ancestors, **kwargs)

@overload
def get_node_from_names(root:CGNSTree, names:List[str], ancestors:Literal[True], **kwargs) -> Optional[Tuple[CGNSTree, ...]]: ...
@overload
def get_node_from_names(root:CGNSTree, names:List[str], ancestors:Literal[False], **kwargs) -> Optional[CGNSTree]: ...
@overload
def get_node_from_names(root:CGNSTree, names:List[str], **kwargs) -> Optional[CGNSTree]: ...

def get_node_from_names(root:CGNSTree, names:List[str], ancestors=False, **kwargs):
  """Specialization of get_node_from_predicates with embedded predicates name_matches"""
  predicates:Predicates = [name_matches(name) for name in names]
  return get_node_from_predicates(root, predicates, ancestors, **kwargs)

@overload
def get_child_from_names(root:CGNSTree, names:List[str], ancestors:Literal[True], **kwargs) -> Optional[Tuple[CGNSTree, ...]]: ...
@overload
def get_child_from_names(root:CGNSTree, names:List[str], ancestors:Literal[False], **kwargs) -> Optional[CGNSTree]: ...
@overload
def get_child_from_names(root:CGNSTree, names:List[str], **kwargs) -> Optional[CGNSTree]: ...

def get_child_from_names(root:CGNSTree, names:List[str], ancestors=False, **kwargs):
  """Specialization of get_node_from_names with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_node_from_names(root, names, ancestors, **kwargs)

@overload
def get_node_from_labels(root:CGNSTree, labels:List[str], ancestors:Literal[True], **kwargs) -> Optional[Tuple[CGNSTree, ...]]: ...
@overload
def get_node_from_labels(root:CGNSTree, labels:List[str], ancestors:Literal[False], **kwargs) -> Optional[CGNSTree]: ...
@overload
def get_node_from_labels(root:CGNSTree, labels:List[str], **kwargs) -> Optional[CGNSTree]: ...

def get_node_from_labels(root:CGNSTree, labels:List[str], ancestors=False, **kwargs):
  """Specialization of get_node_from_predicates with embedded predicates label_matches"""
  predicates:Predicates = [label_matches(label) for label in labels]
  return get_node_from_predicates(root, predicates, ancestors, **kwargs)

@overload
def get_child_from_labels(root:CGNSTree, labels:List[str], ancestors:Literal[True], **kwargs) -> Optional[Tuple[CGNSTree, ...]]: ...
@overload
def get_child_from_labels(root:CGNSTree, labels:List[str], ancestors:Literal[False], **kwargs) -> Optional[CGNSTree]: ...
@overload
def get_child_from_labels(root:CGNSTree, labels:List[str], **kwargs) -> Optional[CGNSTree]: ...

def get_child_from_labels(root:CGNSTree, labels:List[str], ancestors=False, **kwargs):
  """Specialization of get_node_from_labels with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_node_from_labels(root, labels, ancestors, **kwargs)

@overload
def get_node_from_values(root:CGNSTree, values, ancestors:Literal[True], **kwargs) -> Optional[Tuple[CGNSTree, ...]]: ...
@overload
def get_node_from_values(root:CGNSTree, values, ancestors:Literal[False], **kwargs) -> Optional[CGNSTree]: ...
@overload
def get_node_from_values(root:CGNSTree, values, **kwargs) -> Optional[CGNSTree]: ...

def get_node_from_values(root:CGNSTree, values, ancestors=False, **kwargs):
  """Specialization of get_node_from_predicates with embedded predicates value_is"""
  predicates:Predicates = [value_is(value) for value in values]
  return get_node_from_predicates(root, predicates, ancestors, **kwargs)

@overload
def get_child_from_values(root:CGNSTree, values, ancestors:Literal[True], **kwargs) -> Optional[Tuple[CGNSTree, ...]]: ...
@overload
def get_child_from_values(root:CGNSTree, values, ancestors:Literal[False], **kwargs) -> Optional[CGNSTree]: ...
@overload
def get_child_from_values(root:CGNSTree, values, **kwargs) -> Optional[CGNSTree]: ...

def get_child_from_values(root:CGNSTree, values, ancestors=False, **kwargs):
  """Specialization of get_node_from_values with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_node_from_values(root, values, ancestors, **kwargs)

@overload
def get_node_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], ancestors:Literal[True], **kwargs) -> Optional[Tuple[CGNSTree, ...]]: ...
@overload
def get_node_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], ancestors:Literal[False], **kwargs) -> Optional[CGNSTree]: ...
@overload
def get_node_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], **kwargs) -> Optional[CGNSTree]: ...

def get_node_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], ancestors=False, **kwargs):
  """Specialization of get_node_from_predicates with embedded predicates match_name_label"""
  assert len(names) == len(labels)
  predicates:Predicates = [name_matches(name) & label_matches(label) for name,label in zip(names, labels)]
  return get_node_from_predicates(root, predicates, ancestors, **kwargs)

@overload
def get_child_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], ancestors:Literal[True], **kwargs) -> Optional[Tuple[CGNSTree, ...]]: ...
@overload
def get_child_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], ancestors:Literal[False], **kwargs) -> Optional[CGNSTree]: ...
@overload
def get_child_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], **kwargs) -> Optional[CGNSTree]: ...

def get_child_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], ancestors=False, **kwargs):
  #"""Specialization of get_node_from_name_and_labels with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_node_from_name_and_labels(root, names, labels, ancestors, **kwargs)
# ---------------------------------------------------------------------------- #



# ---------------------------------------------------------------------------- #
# API for NodesWalkers
# --------------------
# > Generic version
@overload
def iter_nodes_from_predicates(root:CGNSTree, predicates:Predicates, ancestors:Literal[True], **kwargs) -> Iterator[Tuple[CGNSTree, ...]]: ...
@overload
def iter_nodes_from_predicates(root:CGNSTree, predicates:Predicates, ancestors:Literal[False], **kwargs) -> Iterator[CGNSTree]: ...
@overload
def iter_nodes_from_predicates(root:CGNSTree, predicates:Predicates, **kwargs) -> Iterator[CGNSTree]: ...

def iter_nodes_from_predicates(root:CGNSTree, predicates:Predicates, ancestors=False, **kwargs) -> Union[Iterator[CGNSTree], Iterator[Tuple[CGNSTree, ...]]]:
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
  kwargs['ancestors'] = ancestors

  walker = NodesWalkers(root, _predicates, **kwargs)
  return walker()

# > Specialized versions
@overload
def iter_children_from_predicates(root:CGNSTree, predicates:Predicates, ancestors:Literal[True], **kwargs) -> Iterator[Tuple[CGNSTree, ...]]: ...
@overload
def iter_children_from_predicates(root:CGNSTree, predicates:Predicates, ancestors:Literal[False], **kwargs) -> Iterator[CGNSTree]: ...
@overload
def iter_children_from_predicates(root:CGNSTree, predicates:Predicates, **kwargs) -> Iterator[CGNSTree]: ...

def iter_children_from_predicates(root:CGNSTree, predicates:Predicates, ancestors=False, **kwargs):
  """Specialization of iter_nodes_predicates with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return iter_nodes_from_predicates(root, predicates, ancestors, **kwargs)

@overload
def iter_nodes_from_names(root:CGNSTree, names:List[str], ancestors:Literal[True], **kwargs) -> Iterator[Tuple[CGNSTree, ...]]: ...
@overload
def iter_nodes_from_names(root:CGNSTree, names:List[str], ancestors:Literal[False], **kwargs) -> Iterator[CGNSTree]: ...
@overload
def iter_nodes_from_names(root:CGNSTree, names:List[str], **kwargs) -> Iterator[CGNSTree]: ...

def iter_nodes_from_names(root:CGNSTree, names:List[str], ancestors=False, **kwargs):
  """Specialization of iter_nodes_from_predicates with embedded predicates name_matches"""
  predicates:Predicates = [name_matches(name) for name in names]
  return iter_nodes_from_predicates(root, predicates, ancestors, **kwargs)

@overload
def iter_children_from_names(root:CGNSTree, names:List[str], ancestors:Literal[True], **kwargs) -> Iterator[Tuple[CGNSTree, ...]]: ...
@overload
def iter_children_from_names(root:CGNSTree, names:List[str], ancestors:Literal[False], **kwargs) -> Iterator[CGNSTree]: ...
@overload
def iter_children_from_names(root:CGNSTree, names:List[str], **kwargs) -> Iterator[CGNSTree]: ...

def iter_children_from_names(root:CGNSTree, names:List[str], ancestors=False, **kwargs):
  """Specialization of iter_nodes_from_names with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return iter_nodes_from_names(root, names, ancestors, **kwargs)

@overload
def iter_nodes_from_labels(root:CGNSTree, labels:List[str], ancestors:Literal[True], **kwargs) -> Iterator[Tuple[CGNSTree, ...]]: ...
@overload
def iter_nodes_from_labels(root:CGNSTree, labels:List[str], ancestors:Literal[False], **kwargs) -> Iterator[CGNSTree]: ...
@overload
def iter_nodes_from_labels(root:CGNSTree, labels:List[str], **kwargs) -> Iterator[CGNSTree]: ...

def iter_nodes_from_labels(root:CGNSTree, labels:List[str], ancestors=False, **kwargs):
  """Specialization of iter_nodes_from_predicates with embedded predicates label_matches"""
  predicates:Predicates = [label_matches(label) for label in labels]
  return iter_nodes_from_predicates(root, predicates, ancestors, **kwargs)

@overload
def iter_children_from_labels(root:CGNSTree, labels:List[str], ancestors:Literal[True], **kwargs) -> Iterator[Tuple[CGNSTree, ...]]: ...
@overload
def iter_children_from_labels(root:CGNSTree, labels:List[str], ancestors:Literal[False], **kwargs) -> Iterator[CGNSTree]: ...
@overload
def iter_children_from_labels(root:CGNSTree, labels:List[str], **kwargs) -> Iterator[CGNSTree]: ...

def iter_children_from_labels(root:CGNSTree, labels:List[str], ancestors=False, **kwargs):
  """Specialization of iter_nodes_from_labels with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return iter_nodes_from_labels(root, labels, ancestors, **kwargs)

@overload
def iter_nodes_from_values(root:CGNSTree, values, ancestors:Literal[True], **kwargs) -> Iterator[Tuple[CGNSTree, ...]]: ...
@overload
def iter_nodes_from_values(root:CGNSTree, values, ancestors:Literal[False], **kwargs) -> Iterator[CGNSTree]: ...
@overload
def iter_nodes_from_values(root:CGNSTree, values, **kwargs) -> Iterator[CGNSTree]: ...

def iter_nodes_from_values(root:CGNSTree, values, ancestors=False, **kwargs):
  """Specialization of iter_nodes_from_predicates with embedded predicates value_is"""
  predicates:Predicates = [value_is(value) for value in values]
  return iter_nodes_from_predicates(root, predicates, ancestors, **kwargs)

@overload
def iter_children_from_values(root:CGNSTree, values, ancestors:Literal[True], **kwargs) -> Iterator[Tuple[CGNSTree, ...]]: ...
@overload
def iter_children_from_values(root:CGNSTree, values, ancestors:Literal[False], **kwargs) -> Iterator[CGNSTree]: ...
@overload
def iter_children_from_values(root:CGNSTree, values, **kwargs) -> Iterator[CGNSTree]: ...

def iter_children_from_values(root:CGNSTree, values, ancestors=False, **kwargs):
  """Specialization of iter_nodes_from_values with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return iter_nodes_from_values(root, values, ancestors, **kwargs)

@overload
def iter_nodes_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], ancestors:Literal[True], **kwargs) -> Iterator[Tuple[CGNSTree, ...]]: ...
@overload
def iter_nodes_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], ancestors:Literal[False], **kwargs) -> Iterator[CGNSTree]: ...
@overload
def iter_nodes_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], **kwargs) -> Iterator[CGNSTree]: ...

def iter_nodes_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], ancestors=False, **kwargs):
  """Specialization of iter_nodes_from_predicates with embedded predicates match_name_label"""
  assert len(names) == len(labels)
  predicates:Predicates = [name_matches(name) & label_matches(label) for name,label in zip(names, labels)]
  return iter_nodes_from_predicates(root, predicates, ancestors, **kwargs)

@overload
def iter_children_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], ancestors:Literal[True], **kwargs) -> Iterator[Tuple[CGNSTree, ...]]: ...
@overload
def iter_children_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], ancestors:Literal[False], **kwargs) -> Iterator[CGNSTree]: ...
@overload
def iter_children_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], **kwargs) -> Iterator[CGNSTree]: ...

def iter_children_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], ancestors=False, **kwargs):
  """Specialization of iter_nodes_from_name_and_labels with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return iter_nodes_from_name_and_labels(root, names, labels, ancestors, **kwargs)

# > Generic version
@overload
def get_nodes_from_predicates(root:CGNSTree, predicates:Predicates, ancestors:Literal[True], **kwargs) -> List[Tuple[CGNSTree, ...]]: ...
@overload
def get_nodes_from_predicates(root:CGNSTree, predicates:Predicates, ancestors:Literal[False], **kwargs) -> List[CGNSTree]: ...
@overload
def get_nodes_from_predicates(root:CGNSTree, predicates:Predicates, **kwargs) -> List[CGNSTree]: ...

def get_nodes_from_predicates(root:CGNSTree, predicates:Predicates, ancestors:bool=False, **kwargs) -> Union[List[CGNSTree],List[Tuple[CGNSTree, ...]]]:
  """ Return the list of all nodes in input tree matching the chain of predicates

  The search can be fine-tuned with the following kwargs:

  - ``depth`` (int or pair of int): see :func:`get_node_from_predicate`
  - ``search`` (str): see :func:`get_node_from_predicate`
  - ``explore`` (str): see :func:`get_nodes_from_predicate`

  Args:
      root (CGNSTree): Tree is which the search is performed
      predicates (list of callable): conditions to select next node, each one
        having the following signature: ``f(n:CGNSTree) -> bool``
      ancestors (bool): If ``False`` (default), keep only the terminal nodes.
        If ``True``, keep the intermediate nodes and return a list of tuples of nodes instead of a list of nodes.
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
  kwargs['ancestors'] = ancestors

  walker = NodesWalkers(root, _predicates, **kwargs)
  return walker()

# > Specialized versions
@overload
def get_children_from_predicates(root:CGNSTree, predicates:Predicates, ancestors:Literal[True], **kwargs) -> List[Tuple[CGNSTree, ...]]: ...
@overload
def get_children_from_predicates(root:CGNSTree, predicates:Predicates, ancestors:Literal[False], **kwargs) -> List[CGNSTree]: ...
@overload
def get_children_from_predicates(root:CGNSTree, predicates:Predicates, **kwargs) -> List[CGNSTree]: ...

def get_children_from_predicates(root:CGNSTree, predicates:Predicates, ancestors=False, **kwargs):
  """Specialization of get_nodes_predicates with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_nodes_from_predicates(root, predicates, ancestors, **kwargs)

@overload
def get_nodes_from_names(root:CGNSTree, names:List[str], ancestors:Literal[True], **kwargs) -> List[Tuple[CGNSTree, ...]]: ...
@overload
def get_nodes_from_names(root:CGNSTree, names:List[str], ancestors:Literal[False], **kwargs) -> List[CGNSTree]: ...
@overload
def get_nodes_from_names(root:CGNSTree, names:List[str], **kwargs) -> List[CGNSTree]: ...

def get_nodes_from_names(root:CGNSTree, names:List[str], ancestors=False, **kwargs):
  """Specialization of get_nodes_from_predicates with embedded predicates name_matches"""
  predicates:Predicates = [name_matches(name) for name in names]
  return get_nodes_from_predicates(root, predicates, ancestors, **kwargs)

@overload
def get_children_from_names(root:CGNSTree, names:List[str], ancestors:Literal[True], **kwargs) -> List[Tuple[CGNSTree, ...]]: ...
@overload
def get_children_from_names(root:CGNSTree, names:List[str], ancestors:Literal[False], **kwargs) -> List[CGNSTree]: ...
@overload
def get_children_from_names(root:CGNSTree, names:List[str], **kwargs) -> List[CGNSTree]: ...

def get_children_from_names(root:CGNSTree, names:List[str], ancestors=False, **kwargs):
  """Specialization of get_nodes_from_names with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_nodes_from_names(root, names, ancestors, **kwargs)

@overload
def get_nodes_from_labels(root:CGNSTree, labels:List[str], ancestors:Literal[True], **kwargs) -> List[Tuple[CGNSTree, ...]]: ...
@overload
def get_nodes_from_labels(root:CGNSTree, labels:List[str], ancestors:Literal[False], **kwargs) -> List[CGNSTree]: ...
@overload
def get_nodes_from_labels(root:CGNSTree, labels:List[str], **kwargs) -> List[CGNSTree]: ...

def get_nodes_from_labels(root:CGNSTree, labels:List[str], ancestors=False, **kwargs):
  """Specialization of get_nodes_from_predicates with embedded predicates label_matches"""
  predicates:Predicates = [label_matches(label) for label in labels]
  return get_nodes_from_predicates(root, predicates, ancestors, **kwargs)

@overload
def get_children_from_labels(root:CGNSTree, labels:List[str], ancestors:Literal[True], **kwargs) -> List[Tuple[CGNSTree, ...]]: ...
@overload
def get_children_from_labels(root:CGNSTree, labels:List[str], ancestors:Literal[False], **kwargs) -> List[CGNSTree]: ...
@overload
def get_children_from_labels(root:CGNSTree, labels:List[str], **kwargs) -> List[CGNSTree]: ...

def get_children_from_labels(root:CGNSTree, labels:List[str], ancestors=False, **kwargs):
  """Specialization of get_nodes_from_labels with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_nodes_from_labels(root, labels, ancestors, **kwargs)

@overload
def get_nodes_from_values(root:CGNSTree, values, ancestors:Literal[True], **kwargs) -> List[Tuple[CGNSTree, ...]]: ...
@overload
def get_nodes_from_values(root:CGNSTree, values, ancestors:Literal[False], **kwargs) -> List[CGNSTree]: ...
@overload
def get_nodes_from_values(root:CGNSTree, values, **kwargs) -> List[CGNSTree]: ...

def get_nodes_from_values(root:CGNSTree, values, ancestors=False, **kwargs):
  """Specialization of get_nodes_from_predicates with embedded predicates value_is"""
  predicates:Predicates = [value_is(value) for value in values]
  return get_nodes_from_predicates(root, predicates, ancestors, **kwargs)

@overload
def get_children_from_values(root:CGNSTree, values, ancestors:Literal[True], **kwargs) -> List[Tuple[CGNSTree, ...]]: ...
@overload
def get_children_from_values(root:CGNSTree, values, ancestors:Literal[False], **kwargs) -> List[CGNSTree]: ...
@overload
def get_children_from_values(root:CGNSTree, values, **kwargs) -> List[CGNSTree]: ...

def get_children_from_values(root:CGNSTree, values, ancestors=False, **kwargs):
  """Specialization of get_nodes_from_values with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_nodes_from_values(root, values, ancestors, **kwargs)

@overload
def get_nodes_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], ancestors:Literal[True], **kwargs) -> List[Tuple[CGNSTree, ...]]: ...
@overload
def get_nodes_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], ancestors:Literal[False], **kwargs) -> List[CGNSTree]: ...
@overload
def get_nodes_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], **kwargs) -> List[CGNSTree]: ...

def get_nodes_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], ancestors=False, **kwargs):
  """Specialization of get_nodes_from_predicates with embedded predicates match_name_label"""
  assert len(names) == len(labels)
  predicates:Predicates = [name_matches(name) & label_matches(label) for name,label in zip(names, labels)]
  return get_nodes_from_predicates(root, predicates, ancestors, **kwargs)

@overload
def get_children_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], ancestors:Literal[True], **kwargs) -> List[Tuple[CGNSTree, ...]]: ...
@overload
def get_children_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], ancestors:Literal[False], **kwargs) -> List[CGNSTree]: ...
@overload
def get_children_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], **kwargs) -> List[CGNSTree]: ...

def get_children_from_name_and_labels(root:CGNSTree, names:List[str], labels:List[str], ancestors=False, **kwargs):
  """Specialization of get_nodes_from_name_and_labels with depth=[1,1]"""
  kwargs['depth'] = [1,1]
  return get_nodes_from_name_and_labels(root, names, labels, ancestors, **kwargs)

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Miscallaneous searches
def get_node_from_path(root:CGNSTree, path:CGNSPath) -> Optional[CGNSTree]:
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
    >>> node = PT.get_node_from_path(zone, 'ZoneBC/BC/GridLocation')
    >>> PT.get_name(node) # Node is returned
    'GridLocation'
    >>> PT.get_node_from_path(zone, 'ZoneBC/BC/PointRange') # Returns None
  """
  if path == '':
    return root
  names = path.split('/')
  node = root
  for name in names:
    try:
      node = next((c for c in node[2] if c[0] == name))
    except StopIteration:
      return None
  return node

def find_node_from_path(root:CGNSTree, path:CGNSPath) -> CGNSTree:
  if (node := get_node_from_path(root, path)) is not None:
    return node
  raise CGNSNodeNotFoundError(root, path)


# For get|iter_all_Zone|Base_t, we allow generic type to preserve input
# tree kind

Tree = TypeVar('Tree', bound=CGNSTree)
def get_all_Zone_t(root:Tree) -> List[Tree]:
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

def iter_all_Zone_t(root:Tree) -> Iterator[Tree]:
  root_label = root[3]
  if root_label == 'Zone_t':
    yield root
  else:
    for base in iter_all_CGNSBase_t(root):
      yield from iter_children_from_label(base, 'Zone_t') #type:ignore[misc] #(iter_children is not generic)
  

def get_all_CGNSBase_t(root:Tree) -> List[Tree]:
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

def iter_all_CGNSBase_t(root:Tree) -> Iterator[Tree]:
  if root[3] == 'CGNSBase_t':
    yield root
  elif root[3] == 'CGNSTree_t':
    yield from iter_children_from_label(root, 'CGNSBase_t') #type:ignore[misc] #(iter_children is not generic)

def get_all_subsets(root:CGNSTree, filter_loc:Optional[List[str]]=None) -> List[CGNSTree]:
  """
  Search and collect all the subsets nodes found under root and the root
  itself if it is a subset
  If filter_loc list is not None, select only the subsets nodes of given
  GridLocation.
  """
  return list(iter_all_subsets(root,filter_loc))

def iter_all_subsets(root:CGNSTree, filter_loc:Optional[Sequence[str]]=None) -> Iterator[CGNSTree]:
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

def predicates_to_paths(root:CGNSTree, predicates:Predicates) -> List[CGNSPath]:
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
  if nodes is None:
    return None
  else:
    return '/'.join([n[0] for n in nodes])
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Aliases for legacy code -- using default argument deep instead of shallow for search

def getNodeFromPredicate(root, predicate, *args, **kwargs):
  """ Alias for get_node_from_predicate"""
  return get_node_from_predicate(root, predicate, *args, **kwargs)

def findNodeFromPredicate(root, predicate, *args, **kwargs):
  """ Alias for find_node_from_predicate"""
  return find_node_from_predicate(root, predicate, *args, **kwargs)

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

