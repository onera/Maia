import pytest
import os

import maia.pytree           as PT
import maia.pytree.yaml      as PTy
import maia.pytree.pred      as PTp

from maia.pytree.meta import CGNSNodeNotFoundError

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

basic_tree = PTy.to_node(yt)

def test_get_node_from_predicate():

  assert PT.get_node_from_predicate(basic_tree, PTp.name_matches('bc2')) == PT.get_node_from_predicate(basic_tree, 'bc2')
  assert PT.get_node_from_predicate(basic_tree, 'bc8') is None
  assert PT.getNodeFromPredicate(basic_tree, 'BC_t') == PT.get_node_from_predicate(basic_tree, 'BC_t')
  assert PT.get_node_from_predicate(basic_tree, 'BC_t', sort=lambda l:reversed(l))[0] == 'bc2'

def test_request_node_from_predicate():
  assert PT.find_node_from_predicate(basic_tree, 'bc2') is not None
  assert PT.findNodeFromPredicate(basic_tree, 'bc2') == PT.find_node_from_predicate(basic_tree, 'bc2')
  with pytest.raises(CGNSNodeNotFoundError):
    PT.find_node_from_predicate(basic_tree, 'bc8')

def test_get_nodes_from_predicate():

  assert isinstance(PT.get_nodes_from_predicate(basic_tree, 'bc*'), list)

  # Auto predicate
  assert PT.get_nodes_from_predicate(basic_tree, PTp.name_matches('bc*')) == PT.get_nodes_from_predicate(basic_tree, 'bc*')

  # snake_case => shallow search, CamelCase => Deep search
  bc_or_family = lambda n: PT.get_label(n) in ['BC_t', 'FamilyName_t']
  assert get_names(PT.get_nodes_from_predicate(basic_tree, bc_or_family)) == ['bc1', 'bc2', 'FamilyName']
  assert get_names(PT.getNodesFromPredicate(basic_tree, bc_or_family)) == ['bc1', 'FamilyName', 'bc2', 'FamilyName', 'FamilyName']

def test_iter_nodes_from_predicate():
  assert not isinstance(PT.iter_nodes_from_predicate(basic_tree, 'bc*'), list) # Generator

  # Auto predicate
  assert list(PT.iter_nodes_from_predicate(basic_tree, PTp.name_matches('bc*'))) == list(PT.iter_nodes_from_predicate(basic_tree, 'bc*'))

  # snake_case => shallow search, CamelCase => Deep search
  bc_or_family = lambda n: PT.get_label(n) in ['BC_t', 'FamilyName_t']
  assert get_names(PT.iter_nodes_from_predicate(basic_tree, bc_or_family)) == ['bc1', 'bc2', 'FamilyName']
  assert get_names(PT.iterNodesFromPredicate(basic_tree, bc_or_family)) == ['bc1', 'FamilyName', 'bc2', 'FamilyName', 'FamilyName']

def test_get_node_from_predicates():

  # Single predicate fallback to from_predicate
  assert PT.get_node_from_predicates(basic_tree, "FamilyName_t") == PT.get_node_from_predicate(basic_tree, "FamilyName_t")

  # Auto predicate
  assert PT.get_node_from_predicates(basic_tree, ["BC_t", "FamilyName_t"]) == \
      PT.get_node_from_predicates(basic_tree, [PTp.label_matches('BC_t'), PTp.label_matches('FamilyName_t')])
  assert PT.get_node_from_predicates(basic_tree, "BC_t/FamilyName_t") == \
      PT.get_node_from_predicates(basic_tree, [PTp.label_matches('BC_t'), PTp.label_matches('FamilyName_t')])

  assert PT.get_value(PT.get_node_from_predicates(basic_tree, ["BC_t", "FamilyName_t"])) == "BC1" # Only one is returned

  # Common kwargs vs specific options for each predicate
  assert PT.get_node_from_predicates(basic_tree, "BC_t/IndexArray_t", depth=1) is None
  predicates = [{'predicate':'BC_t', 'depth':2}, {'predicate':'IndexArray_t', 'depth':1}]
  assert PT.get_node_from_predicates(basic_tree, predicates) is not None

  # Note : this one failed before #224, because NodeWalkers is badly implemented 
  assert PT.get_child_from_predicates(basic_tree, ['ZoneBC_t', 'BC_t', 'Index_ii']) is not None
  assert PT.get_child_from_predicates(basic_tree, []) is None
  assert PT.get_child_from_predicates(basic_tree, [], ancestors=True) == ()

def test_get_nodes_from_predicates():

  # Single predicate fallback to from_predicate
  assert PT.get_nodes_from_predicates(basic_tree, "FamilyName_t") == PT.get_nodes_from_predicate(basic_tree, "FamilyName_t")
  assert PT.getNodesFromPredicates(basic_tree, "FamilyName_t") == PT.getNodesFromPredicate(basic_tree, "FamilyName_t")

  # Auto predicate
  assert PT.get_nodes_from_predicates(basic_tree, ["BC_t", "FamilyName_t"]) == \
      PT.get_nodes_from_predicates(basic_tree, [PTp.label_matches('BC_t'), PTp.label_matches('FamilyName_t')])
  assert PT.get_nodes_from_predicates(basic_tree, "BC_t/FamilyName_t") == \
      PT.get_nodes_from_predicates(basic_tree, [PTp.label_matches('BC_t'), PTp.label_matches('FamilyName_t')])
  assert PT.getNodesFromPredicates(basic_tree, "BC_t/FamilyName_t") == \
      PT.getNodesFromPredicates(basic_tree, [PTp.label_matches('BC_t'), PTp.label_matches('FamilyName_t')])

  # With ancestors
  results = PT.get_nodes_from_predicates(basic_tree, "BC_t/FamilyName_t", ancestors=True)
  assert PT.get_name(results[0][0]) == "bc1" and PT.get_value(results[0][1]) == "BC1"
  assert PT.get_name(results[1][0]) == "bc2" and PT.get_value(results[1][1]) == "BC2"

  results = PT.getNodesFromPredicates(basic_tree, "Zone_t/ZoneBC_t/BC_t/FamilyName_t", ancestors=True)
  for result in results:
    assert len(result) == 4

  # Share option between predicates
  assert PT.get_nodes_from_predicates(basic_tree, "BC_t/IndexArray_t", depth=1) == []

  # Specific options for each predicate
  predicates = [{'predicate':'BC_t', 'depth':2}, {'predicate':'IndexArray_t', 'depth':1}]
  assert len(PT.get_nodes_from_predicates(basic_tree, predicates)) == 2

def test_iter_nodes_from_predicates():

  # Just check that we have same result than get
  results_iter = PT.iter_nodes_from_predicates(basic_tree, "BC_t/FamilyName_t", ancestors=True)
  results_get  = PT.get_nodes_from_predicates(basic_tree, "BC_t/FamilyName_t", ancestors=True)
  assert     isinstance(results_get,  list)
  assert not isinstance(results_iter, list)
  for result_iter, result_get in zip(results_iter, results_get):
    assert result_iter == result_get

  results_iter = PT.iterNodesFromPredicates(basic_tree, "BC_t/FamilyName_t", ancestors=True)
  results_get  = PT.getNodesFromPredicates(basic_tree, "BC_t/FamilyName_t", ancestors=True)
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


# ---------------------------------------------------------------------------
# Tests sur les fonctions simples avec enfants

def test_get_child_from_value():
  bc1 = PT.get_node_from_name(basic_tree, 'bc1')
  child = PT.get_child_from_value(bc1, 'BC1')
  assert child is not None
  assert PT.get_value(child) == 'BC1'
  no_child = PT.get_child_from_value(bc1, 'NonExisting')
  assert no_child is None

def test_request_child_from_predicate():
  bc1 = PT.get_node_from_name(basic_tree, 'bc1')
  node = PT.find_child_from_predicate(bc1, PTp.name_matches('FamilyName'))
  assert node is not None
  assert PT.get_name(node) == 'FamilyName'
  
  with pytest.raises(CGNSNodeNotFoundError):
      PT.find_child_from_predicate(bc1, 'NonExist')

def test_request_child_from_name():
  bc2 = PT.get_node_from_name(basic_tree, 'bc2')
  node = PT.find_child_from_name(bc2, 'FamilyName')
  assert node is not None
  assert PT.get_name(node) == 'FamilyName'
  with pytest.raises(CGNSNodeNotFoundError):
      PT.find_child_from_name(bc2, 'NonExist')

def test_request_child_from_label():
  bc2 = PT.get_node_from_name(basic_tree, 'bc2')
  node = PT.find_child_from_label(bc2, 'FamilyName_t')
  assert node is not None
  assert PT.get_label(node) == 'FamilyName_t'
  with pytest.raises(CGNSNodeNotFoundError):
    PT.find_child_from_label(bc2, 'NonExistLabel')

def test_request_child_from_value():
  bc1 = PT.get_node_from_name(basic_tree, 'bc1')
  node = PT.find_child_from_value(bc1, 'BC1')
  assert node is not None
  assert PT.get_value(node) == 'BC1'
  with pytest.raises(CGNSNodeNotFoundError):
    PT.find_child_from_value(bc1, 'NonExist')

def test_request_child_from_name_and_label():
  bc1 = PT.get_node_from_name(basic_tree, 'bc1')
  node = PT.find_child_from_name_and_label(bc1, 'FamilyName', 'FamilyName_t')
  assert node is not None
  assert PT.get_name(node) == 'FamilyName'
  with pytest.raises(CGNSNodeNotFoundError):
    PT.find_child_from_name_and_label(bc1, 'NonExist', 'FamilyName_t')

# ---------------------------------------------------------------------------
# Tests des fonctions de recherche multiples avec warning sur le paramètre caching

def test_get_nodes_from_predicate_warning(capsys):
  nodes = PT.get_nodes_from_predicate(basic_tree, 'bc*', caching=False)
  captured = capsys.readouterr().out
  assert "Warning: get_nodes_from_predicate forces caching to True." in captured
  assert isinstance(nodes, list)

def test_iter_nodes_from_predicate_warning(capsys):
  _ = list(PT.iter_nodes_from_predicate(basic_tree, 'bc*', caching=True))
  captured = capsys.readouterr().out
  assert "Warning: iter_nodes_from_predicate forces caching to False." in captured

# ---------------------------------------------------------------------------
# Tests itératifs sur la valeur

def test_iter_nodes_from_value():
  bc1 = PT.get_node_from_name(basic_tree, 'bc1')
  nodes = list(PT.iter_nodes_from_value(bc1, 'BC1'))
  assert len(nodes) == 1
  assert PT.get_value(nodes[0]) == 'BC1'

def test_iter_children_from_value():
  bc1 = PT.get_node_from_name(basic_tree, 'bc1')
  nodes = list(PT.iter_children_from_value(bc1, 'BC1'))
  assert len(nodes) == 1
  assert PT.get_value(nodes[0]) == 'BC1'

# ---------------------------------------------------------------------------
# Tests des fonctions sur les prédicats combinés par nom et label

def test_iter_nodes_from_name_and_label():
  nodes = list(PT.iter_nodes_from_name_and_label(basic_tree, 'FamilyName', 'FamilyName_t'))
  values = [PT.get_value(n) for n in nodes]
  assert values == ['BC1', 'BC2', 'ROW1']

def test_iter_children_from_name_and_label():
  bc2 = PT.get_node_from_name(basic_tree, 'bc2')
  nodes = list(PT.iter_children_from_name_and_label(bc2, 'FamilyName', 'FamilyName_t'))
  assert len(nodes) == 1
  assert PT.get_name(nodes[0]) == 'FamilyName'

# ---------------------------------------------------------------------------
# Tests sur les fonctions de recherche par chaînes de noms/valeurs

def test_get_node_from_names():
  node = PT.get_node_from_names(basic_tree, ['Zone', 'ZoneBC', 'bc1'])
  assert node is not None
  assert PT.get_name(node) == 'bc1'

def test_get_child_from_names():
  node = PT.get_child_from_names(basic_tree, ['ZoneBC'])
  assert node is not None
  assert PT.get_name(node) == 'ZoneBC'

def test_get_node_from_values():
  bc1 = PT.get_node_from_name(basic_tree, 'bc1')
  node = PT.get_node_from_values(bc1, ['BC1'])
  assert node is not None
  assert PT.get_value(node) == 'BC1'

def test_get_child_from_values():
  bc1 = PT.get_node_from_name(basic_tree, 'bc1')
  node = PT.get_child_from_values(bc1, ['BC1'])
  assert node is not None
  assert PT.get_value(node) == 'BC1'

def test_get_node_from_name_and_labels():
  bc1 = PT.get_node_from_name(basic_tree, 'bc1')
  node = PT.get_node_from_name_and_labels(bc1, ['FamilyName'], ['FamilyName_t'])
  assert node is not None
  assert PT.get_name(node) == 'FamilyName'

def test_get_child_from_name_and_labels():
  bc2 = PT.get_node_from_name(basic_tree, 'bc2')
  node = PT.get_child_from_name_and_labels(bc2, ['FamilyName'], ['FamilyName_t'])
  assert node is not None
  assert PT.get_name(node) == 'FamilyName'

# ---------------------------------------------------------------------------
# Tests itératifs sur les recherches par prédicats multiples

def test_iter_nodes_from_predicates_warning(capsys):
  _ = list(PT.iter_nodes_from_predicates(basic_tree, 'bc*', caching=True))
  captured = capsys.readouterr().out
  assert "Warning: iter_nodes_from_predicates forces caching to False." in captured

def test_iter_nodes_from_names():
  nodes = list(PT.iter_nodes_from_names(basic_tree, ['Zone', 'ZoneBC', 'bc1']))
  assert len(nodes) == 1
  assert PT.get_name(nodes[0]) == 'bc1'

def test_iter_children_from_names():
  nodes = list(PT.iter_children_from_names(basic_tree, ['ZoneBC']))
  assert len(nodes) == 1
  assert PT.get_name(nodes[0]) == 'ZoneBC'
  nodes = list(PT.iter_children_from_names(basic_tree, ['ZoneBC', 'FamilyName']))
  assert len(nodes) == 0

def test_iter_nodes_from_values():
  bc1 = PT.get_node_from_name(basic_tree, 'bc1')
  nodes = list(PT.iter_nodes_from_values(bc1, ['BC1']))
  assert len(nodes) == 1
  assert PT.get_value(nodes[0]) == 'BC1'

def test_iter_children_from_values():
  bc1 = PT.get_node_from_name(basic_tree, 'bc1')
  nodes = list(PT.iter_children_from_values(bc1, ['BC1']))
  assert len(nodes) == 1
  assert PT.get_value(nodes[0]) == 'BC1'

def test_iter_nodes_from_name_and_labels():
  nodes = list(PT.iter_nodes_from_name_and_labels(basic_tree, ['FamilyName'], ['FamilyName_t']))
  assert len(nodes) == 3
  for n in nodes:
      assert PT.get_name(n) == 'FamilyName'

  nodes = list(PT.iter_nodes_from_name_and_labels(basic_tree, ['bc*', 'FamilyName'], ['BC_t', 'FamilyName_t']))
  assert len(nodes) == 2

def test_iter_children_from_name_and_labels_duplicate():
  bc2 = PT.get_node_from_name(basic_tree, 'bc2')
  nodes = list(PT.iter_children_from_name_and_labels(bc2, ['FamilyName'], ['FamilyName_t']))
  assert len(nodes) == 1
  assert PT.get_name(nodes[0]) == 'FamilyName'

# ---------------------------------------------------------------------------
# Tests sur les fonctions de recherche multiples (nodes)

def test_get_nodes_from_predicates_warning(capsys):
  nodes = PT.get_nodes_from_predicates(basic_tree, 'bc*', caching=False)
  captured = capsys.readouterr().out
  assert "Warning: get_nodes_from_predicates forces caching to True." in captured
  assert isinstance(nodes, list)

def test_get_nodes_from_values():
  bc1 = PT.get_node_from_name(basic_tree, 'bc1')
  nodes = PT.get_nodes_from_values(bc1, ['BC1'])
  assert isinstance(nodes, list)
  assert len(nodes) == 1
  assert PT.get_value(nodes[0]) == 'BC1'

def test_get_children_from_values():
  bc1 = PT.get_node_from_name(basic_tree, 'bc1')
  nodes = PT.get_children_from_values(bc1, ['BC1'])
  assert isinstance(nodes, list)
  assert len(nodes) == 1
  assert PT.get_value(nodes[0]) == 'BC1'

def test_get_nodes_from_name_and_labels():
  bc1 = PT.get_node_from_name(basic_tree, 'bc1')
  nodes = PT.get_nodes_from_name_and_labels(bc1, ['FamilyName'], ['FamilyName_t'])
  assert isinstance(nodes, list)
  assert len(nodes) == 1
  assert PT.get_name(nodes[0]) == 'FamilyName'

def test_get_children_from_name_and_labels():
  bc2 = PT.get_node_from_name(basic_tree, 'bc2')
  nodes = PT.get_children_from_name_and_labels(bc2, ['FamilyName'], ['FamilyName_t'])
  assert isinstance(nodes, list)
  assert len(nodes) == 1
  assert PT.get_name(nodes[0]) == 'FamilyName'

# ---------------------------------------------------------------------------
# Test sur l'alias hérité getNodeFromPredicates
def test_getNodeFromPredicates():
  node1 = PT.get_node_from_predicates(basic_tree, 'FamilyName_t')
  node2 = PT.getNodeFromPredicates(basic_tree, 'FamilyName_t')
  assert node1 == node2

def test_get_all_Zone_t():
  # From Zone
  yaml_str = "MyZone Zone_t:"
  tree = PTy.to_node(yaml_str)
  zones = PT.get_all_Zone_t(tree)
  assert len(zones) == 1
  assert PT.get_name(zones[0]) == "MyZone"

  # From Base
  yaml_str = """
  Base CGNSBase_t:
    Zone1 Zone_t:
    Other Something_t:
    Zone2 Zone_t:
  """
  tree = PTy.to_node(yaml_str)
  zones = PT.get_all_Zone_t(tree)
  assert len(zones) == 2
  assert get_names(zones) == ["Zone1", "Zone2"]

  # From Tree
  yaml_str = """
  Tree CGNSTree_t:
    BaseA CGNSBase_t:
      Zone1 Zone_t:
      ZoneExtra Zone_t:
    BaseB CGNSBase_t:
      Zone2 Zone_t:
  """
  tree = PTy.to_node(yaml_str)
  zones = PT.get_all_Zone_t(tree)
  assert len(zones) == 3
  assert get_names(zones) == ["Zone1", "ZoneExtra", "Zone2"]

def test_get_all_cgnsbase_t_from_cgnsbase_t():
  # From Base
  yaml_str = "BaseA CGNSBase_t:"
  tree = PTy.to_node(yaml_str)
  bases = PT.get_all_CGNSBase_t(tree)
  assert len(bases) == 1
  assert PT.get_name(bases[0]) == "BaseA"

  # From Tree
  yaml_str = """
  Tree CGNSTree_t:
    BaseA CGNSBase_t:
    BaseB CGNSBase_t:
    NotBase SomethingElse_t:
  """
  tree = PTy.to_node(yaml_str)
  bases = PT.get_all_CGNSBase_t(tree)
  assert len(bases) == 2
  assert get_names(bases) == ["BaseA", "BaseB"]

def test_get_node_from_labels():
  node = PT.get_node_from_labels(basic_tree, ['FamilyName_t'])
  assert node is not None
  assert PT.get_value(node) == 'BC1'

def test_get_nodes_from_names():
  nodes = PT.get_nodes_from_names(basic_tree, ['bc1'])
  assert isinstance(nodes, list)
  assert len(nodes) == 1
  assert PT.get_name(nodes[0]) == "bc1"

def test_get_nodes_from_labels():
  nodes = PT.get_nodes_from_labels(basic_tree, ['BC_t'])
  assert isinstance(nodes, list)
  assert len(nodes) == 2
  names = sorted([PT.get_name(n) for n in nodes])
  assert names == ["bc1", "bc2"]
