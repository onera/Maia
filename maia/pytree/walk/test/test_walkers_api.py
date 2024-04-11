import pytest
import os

import maia.pytree           as PT
import maia.pytree.yaml      as PTy
import maia.pytree.predicate as PTp

from maia.pytree.meta import CGNSNodeFromPredicateNotFoundError

yt = """
Zone Zone_t:
  ZoneBC ZoneBC_t:
    bc1 BC_t:
      FamilyName FamilyName_t 'BC1':
      Index_i IndexArray_t:
    bc2 BC_t:
      FamilyName FamilyName_t 'BC2':
      Index_ii IndexArray_t:
  FamilyName FamilyName_t 'ROW1':
"""

dir_path = PT.__path__[0]

get_names = lambda nodes : [PT.get_name(node) for node in nodes]

def test_get_node_from_predicate():
  tree = PTy.to_node(yt)

  assert PT.get_node_from_predicate(tree, lambda n: PTp.match_name(n, 'bc2')) == PT.get_node_from_predicate(tree, 'bc2')
  assert PT.get_node_from_predicate(tree, 'bc8') is None
  assert PT.getNodeFromPredicate(tree, 'BC_t') == PT.get_node_from_predicate(tree, 'BC_t')
  assert PT.get_node_from_predicate(tree, 'BC_t', sort=lambda l:reversed(l))[0] == 'bc2'

def test_request_node_from_predicate():
  tree = PTy.to_node(yt)

  assert PT.request_node_from_predicate(tree, 'bc2') is not None
  assert PT.requestNodeFromPredicate(tree, 'bc2') == PT.request_node_from_predicate(tree, 'bc2')
  with pytest.raises(CGNSNodeFromPredicateNotFoundError):
    PT.request_node_from_predicate(tree, 'bc8')
  assert PT.request_node_from_predicate(tree, 'bc8', default=tree)[0] == "Zone"


def test_get_nodes_from_predicate():
  tree = PTy.to_node(yt)

  assert isinstance(PT.get_nodes_from_predicate(tree, 'bc*'), list)

  #Auto predicate
  assert PT.get_nodes_from_predicate(tree, lambda n: PTp.match_name(n, 'bc*')) == PT.get_nodes_from_predicate(tree, 'bc*')

  # snake_case => shallow search, CamelCase => Deep seach
  bc_or_family = lambda n: PT.get_label(n) in ['BC_t', 'FamilyName_t']
  assert get_names(PT.get_nodes_from_predicate(tree, bc_or_family)) == ['bc1', 'bc2', 'FamilyName']
  assert get_names(PT.getNodesFromPredicate(tree, bc_or_family)) == ['bc1', 'FamilyName', 'bc2', 'FamilyName', 'FamilyName']

def test_iter_nodes_from_predicate():
  tree = PTy.to_node(yt)

  assert not isinstance(PT.iter_nodes_from_predicate(tree, 'bc*'), list) #Generator

  #Auto predicate
  assert list(PT.iter_nodes_from_predicate(tree, lambda n: PTp.match_name(n, 'bc*'))) == list(PT.iter_nodes_from_predicate(tree, 'bc*'))

  # snake_case => shallow search, CamelCase => Deep seach
  bc_or_family = lambda n: PT.get_label(n) in ['BC_t', 'FamilyName_t']
  assert get_names(PT.iter_nodes_from_predicate(tree, bc_or_family)) == ['bc1', 'bc2', 'FamilyName']
  assert get_names(PT.iterNodesFromPredicate(tree, bc_or_family)) == ['bc1', 'FamilyName', 'bc2', 'FamilyName', 'FamilyName']
 
def test_get_node_from_predicates(): 
  tree = PTy.to_node(yt)

  # Single predicate fallback to from_predicate
  assert PT.get_node_from_predicates(tree, "FamilyName_t") == PT.get_node_from_predicate(tree, "FamilyName_t")

  # Auto predicate
  assert PT.get_node_from_predicates(tree, ["BC_t", "FamilyName_t"]) == \
      PT.get_node_from_predicates(tree, [lambda n: PTp.match_label(n, 'BC_t'), lambda n: PTp.match_label(n, 'FamilyName_t')])
  assert PT.get_node_from_predicates(tree, "BC_t/FamilyName_t") == \
      PT.get_node_from_predicates(tree, [lambda n: PTp.match_label(n, 'BC_t'), lambda n: PTp.match_label(n, 'FamilyName_t')])

  assert PT.get_value(PT.get_node_from_predicates(tree, ["BC_t", "FamilyName_t"])) == "BC1" # Only one is returned

  # Common kwargs vs specific options for each predicate
  assert PT.get_node_from_predicates(tree, "BC_t/IndexArray_t", depth=1) is None
  predicates = [{'predicate':'BC_t', 'depth':2}, {'predicate':'IndexArray_t', 'depth':1}]
  assert PT.get_node_from_predicates(tree, predicates) is not None

def test_get_nodes_from_predicates():
  tree = PTy.to_node(yt)

  # Single predicate fallback to from_predicate
  assert PT.get_nodes_from_predicates(tree, "FamilyName_t") == PT.get_nodes_from_predicate(tree, "FamilyName_t")
  assert PT.getNodesFromPredicates(tree, "FamilyName_t") == PT.getNodesFromPredicate(tree, "FamilyName_t")

  # Auto predicate
  assert PT.get_nodes_from_predicates(tree, ["BC_t", "FamilyName_t"]) == \
      PT.get_nodes_from_predicates(tree, [lambda n: PTp.match_label(n, 'BC_t'), lambda n: PTp.match_label(n, 'FamilyName_t')])
  assert PT.get_nodes_from_predicates(tree, "BC_t/FamilyName_t") == \
      PT.get_nodes_from_predicates(tree, [lambda n: PTp.match_label(n, 'BC_t'), lambda n: PTp.match_label(n, 'FamilyName_t')])
  assert PT.getNodesFromPredicates(tree, "BC_t/FamilyName_t") == \
      PT.getNodesFromPredicates(tree, [lambda n: PTp.match_label(n, 'BC_t'), lambda n: PTp.match_label(n, 'FamilyName_t')])

  # With ancestors
  results = PT.get_nodes_from_predicates(tree, "BC_t/FamilyName_t", ancestors=True)
  assert PT.get_name(results[0][0]) == "bc1" and PT.get_value(results[0][1]) == "BC1"
  assert PT.get_name(results[1][0]) == "bc2" and PT.get_value(results[1][1]) == "BC2"

  results = PT.getNodesFromPredicates(tree, "Zone_t/ZoneBC_t/BC_t/FamilyName_t", ancestors=True)
  for result in results:
    assert len(result) == 4

  # Share option between predicates 
  assert PT.get_nodes_from_predicates(tree, "BC_t/IndexArray_t", depth=1) == []

  # Specific options for each predicate
  predicates = [{'predicate':'BC_t', 'depth':2}, {'predicate':'IndexArray_t', 'depth':1}]
  assert len(PT.get_nodes_from_predicates(tree, predicates)) == 2

def test_iter_nodes_from_predicates():
  tree = PTy.to_node(yt)

  # Just ckeck that we have same result than get
  results_iter = PT.iter_nodes_from_predicates(tree, "BC_t/FamilyName_t", ancestors=True)
  results_get  = PT.get_nodes_from_predicates(tree, "BC_t/FamilyName_t", ancestors=True)
  assert     isinstance(results_get,  list)
  assert not isinstance(results_iter, list)
  for result_iter, result_get in zip(results_iter, results_get):
    assert result_iter == result_get

  results_iter = PT.iterNodesFromPredicates(tree, "BC_t/FamilyName_t", ancestors=True)
  results_get  = PT.getNodesFromPredicates(tree, "BC_t/FamilyName_t", ancestors=True)
  assert     isinstance(results_get,  list)
  assert not isinstance(results_iter, list)
  for result_iter, result_get in zip(results_iter, results_get):
    assert result_iter == result_get


def test_predicates_to_path():
  with open(os.path.join(dir_path, "test", "minimal_tree.yaml"), 'r') as yt:
    tree = PTy.to_cgns_tree(yt)

  path = PT.predicates_to_path(tree, ["Base", "Zone_t", "ZGC*", lambda n: int(n[0][-1]) >= 2 and int(n[0][-1]) <= 4])
  assert path == 'Base/ZoneI/ZGCA/gc2'
  assert PT.predicates_to_path(tree, 'Nope/*') is None

def test_predicates_to_paths():
  with open(os.path.join(dir_path, "test", "minimal_tree.yaml"), 'r') as yt:
    tree = PTy.to_cgns_tree(yt)

  paths = PT.predicates_to_paths(tree, ["Base", "Zone_t", "ZGC*", lambda n: int(n[0][-1]) >= 2 and int(n[0][-1]) <= 4])
  assert paths == ['Base/ZoneI/ZGCA/gc2', 'Base/ZoneI/ZGCB/gc3', 'Base/ZoneI/ZGCB/gc4']
  assert PT.predicates_to_paths(tree, 'Nope/*') == []
