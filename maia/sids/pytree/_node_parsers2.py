from .nodes_walker import NodesWalker
from .predicate import auto_predicate


#Parsers for nodesWalkers


def search_nodes_dispatch(node, predicate_like, **kwargs):
  """ Interface to adapted getNodesFromXXX1 function depending of predicate type"""
  predicate = auto_predicate(predicate_like)
  walker = NodesWalker(node, predicate, **kwargs)
  return walker()

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

