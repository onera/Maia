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

def get_ZoneBC(root):
    return PT.get_child_from_name(root, 'ZoneBC')

def get_bc1(zonebc):
    return PT.get_child_from_name(zonebc, 'bc1')

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

  # Auto predicate
  assert PT.get_nodes_from_predicate(tree, lambda n: PTp.match_name(n, 'bc*')) == PT.get_nodes_from_predicate(tree, 'bc*')

  # snake_case => shallow search, CamelCase => Deep search
  bc_or_family = lambda n: PT.get_label(n) in ['BC_t', 'FamilyName_t']
  assert get_names(PT.get_nodes_from_predicate(tree, bc_or_family)) == ['bc1', 'bc2', 'FamilyName']
  assert get_names(PT.getNodesFromPredicate(tree, bc_or_family)) == ['bc1', 'FamilyName', 'bc2', 'FamilyName', 'FamilyName']

def test_iter_nodes_from_predicate():
  tree = PTy.to_node(yt)

  assert not isinstance(PT.iter_nodes_from_predicate(tree, 'bc*'), list) # Generator

  # Auto predicate
  assert list(PT.iter_nodes_from_predicate(tree, lambda n: PTp.match_name(n, 'bc*'))) == list(PT.iter_nodes_from_predicate(tree, 'bc*'))

  # snake_case => shallow search, CamelCase => Deep search
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

  # Just check that we have same result than get
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


# ---------------------------------------------------------------------------
# Tests sur les fonctions simples avec enfants

def test_get_child_from_value():
  tree = PTy.to_node(yt)
  bc1 = PT.get_node_from_name(tree, 'bc1')
  child = PT.get_child_from_value(bc1, 'BC1')
  assert child is not None
  assert PT.get_value(child) == 'BC1'
  no_child = PT.get_child_from_value(bc1, 'NonExisting')
  assert no_child is None

def test_request_child_from_predicate():
  tree = PTy.to_node(yt)
  bc1 = PT.get_node_from_name(tree, 'bc1')
  node = PT.request_child_from_predicate(bc1, lambda n: PTp.match_name(n, 'FamilyName'))
  assert node is not None
  assert PT.get_name(node) == 'FamilyName'
  node_default = PT.request_child_from_predicate(bc1, lambda n: PTp.match_name(n, 'NonExist'), default=bc1)
  assert node_default == bc1
  with pytest.raises(CGNSNodeFromPredicateNotFoundError):
      PT.request_child_from_predicate(bc1, 'NonExist')

def test_request_child_from_name():
  tree = PTy.to_node(yt)
  bc2 = PT.get_node_from_name(tree, 'bc2')
  node = PT.request_child_from_name(bc2, 'FamilyName')
  assert node is not None
  assert PT.get_name(node) == 'FamilyName'
  with pytest.raises(CGNSNodeFromPredicateNotFoundError):
      PT.request_child_from_name(bc2, 'NonExist')

def test_request_child_from_label():
  tree = PTy.to_node(yt)
  bc2 = PT.get_node_from_name(tree, 'bc2')
  node = PT.request_child_from_label(bc2, 'FamilyName_t')
  assert node is not None
  assert PT.get_label(node) == 'FamilyName_t'
  with pytest.raises(CGNSNodeFromPredicateNotFoundError):
    PT.request_child_from_label(bc2, 'NonExistLabel')

def test_request_child_from_value():
  tree = PTy.to_node(yt)
  bc1 = PT.get_node_from_name(tree, 'bc1')
  node = PT.request_child_from_value(bc1, 'BC1')
  assert node is not None
  assert PT.get_value(node) == 'BC1'
  node_default = PT.request_child_from_value(bc1, 'NonExist', default=bc1)
  assert node_default == bc1
  with pytest.raises(CGNSNodeFromPredicateNotFoundError):
    PT.request_child_from_value(bc1, 'NonExist')

def test_request_child_from_name_and_label():
  tree = PTy.to_node(yt)
  bc1 = PT.get_node_from_name(tree, 'bc1')
  node = PT.request_child_from_name_and_label(bc1, 'FamilyName', 'FamilyName_t')
  assert node is not None
  assert PT.get_name(node) == 'FamilyName'
  with pytest.raises(CGNSNodeFromPredicateNotFoundError):
    PT.request_child_from_name_and_label(bc1, 'NonExist', 'FamilyName_t')

# ---------------------------------------------------------------------------
# Tests des fonctions de recherche multiples avec warning sur le paramètre caching

def test_get_nodes_from_predicate_warning(capsys):
  tree = PTy.to_node(yt)
  nodes = PT.get_nodes_from_predicate(tree, 'bc*', caching=False)
  captured = capsys.readouterr().out
  assert "Warning: get_nodes_from_predicate forces caching to True." in captured
  assert isinstance(nodes, list)

def test_iter_nodes_from_predicate_warning(capsys):
  tree = PTy.to_node(yt)
  _ = list(PT.iter_nodes_from_predicate(tree, 'bc*', caching=True))
  captured = capsys.readouterr().out
  assert "Warning: iter_nodes_from_predicate forces caching to False." in captured

# ---------------------------------------------------------------------------
# Tests itératifs sur la valeur

def test_iter_nodes_from_value():
  tree = PTy.to_node(yt)
  bc1 = PT.get_node_from_name(tree, 'bc1')
  nodes = list(PT.iter_nodes_from_value(bc1, 'BC1'))
  assert len(nodes) == 1
  assert PT.get_value(nodes[0]) == 'BC1'

def test_iter_children_from_value():
  tree = PTy.to_node(yt)
  bc1 = PT.get_node_from_name(tree, 'bc1')
  nodes = list(PT.iter_children_from_value(bc1, 'BC1'))
  assert len(nodes) == 1
  assert PT.get_value(nodes[0]) == 'BC1'

# ---------------------------------------------------------------------------
# Tests des fonctions sur les prédicats combinés par nom et label

def test_iter_nodes_from_name_and_label():
  tree = PTy.to_node(yt)
  nodes = list(PT.iter_nodes_from_name_and_label(tree, 'FamilyName', 'FamilyName_t'))
  values = sorted([PT.get_value(n) for n in nodes])
  for val in values:
      assert val in ['ROW1', 'BC1', 'BC2']

# La fonction alias pour une recherche en "child" devrait retourner un itérateur.
# Or, il apparaît qu'elle ne retourne rien (None). On marque ce test comme xfail.
@pytest.mark.xfail(reason="iter_children_from_name_and_label manque le return de l'itérateur")
def test_iter_children_from_name_and_label():
  tree = PTy.to_node(yt)
  bc2 = PT.get_node_from_name(tree, 'bc2')
  nodes = list(PT.iter_children_from_name_and_label(bc2, 'FamilyName', 'FamilyName_t'))
  assert len(nodes) == 1
  assert PT.get_name(nodes[0]) == 'FamilyName'

# ---------------------------------------------------------------------------
# Tests sur les fonctions de recherche par chaînes de noms/valeurs

def test_get_node_from_names():
  tree = PTy.to_node(yt)
  node = PT.get_node_from_names(tree, ['Zone', 'ZoneBC', 'bc1'])
  assert node is not None
  assert PT.get_name(node) == 'bc1'

def test_get_child_from_names():
  tree = PTy.to_node(yt)
  node = PT.get_child_from_names(tree, ['ZoneBC'])
  assert node is not None
  assert PT.get_name(node) == 'ZoneBC'

def test_get_node_from_values():
  tree = PTy.to_node(yt)
  bc1 = PT.get_node_from_name(tree, 'bc1')
  node = PT.get_node_from_values(bc1, ['BC1'])
  assert node is not None
  assert PT.get_value(node) == 'BC1'

def test_get_child_from_values():
  tree = PTy.to_node(yt)
  bc1 = PT.get_node_from_name(tree, 'bc1')
  node = PT.get_child_from_values(bc1, ['BC1'])
  assert node is not None
  assert PT.get_value(node) == 'BC1'

def test_get_node_from_name_and_labels():
  tree = PTy.to_node(yt)
  bc1 = PT.get_node_from_name(tree, 'bc1')
  node = PT.get_node_from_name_and_labels(bc1, ['FamilyName'], ['FamilyName_t'])
  assert node is not None
  assert PT.get_name(node) == 'FamilyName'

def test_get_child_from_name_and_labels():
  tree = PTy.to_node(yt)
  bc2 = PT.get_node_from_name(tree, 'bc2')
  node = PT.get_child_from_name_and_labels(bc2, ['FamilyName'], ['FamilyName_t'])
  assert node is not None
  assert PT.get_name(node) == 'FamilyName'

# ---------------------------------------------------------------------------
# Tests itératifs sur les recherches par prédicats multiples

def test_iter_nodes_from_predicates_warning(capsys):
  tree = PTy.to_node(yt)
  _ = list(PT.iter_nodes_from_predicates(tree, 'bc*', caching=True))
  captured = capsys.readouterr().out
  assert "Warning: iter_nodes_from_predicates forces caching to False." in captured

def test_iter_nodes_from_names():
  tree = PTy.to_node(yt)
  nodes = list(PT.iter_nodes_from_names(tree, ['Zone', 'ZoneBC', 'bc1']))
  assert len(nodes) == 1
  assert PT.get_name(nodes[0]) == 'bc1'

def test_iter_children_from_names():
  tree = PTy.to_node(yt)
  nodes = list(PT.iter_children_from_names(tree, ['ZoneBC']))
  assert len(nodes) == 1
  assert PT.get_name(nodes[0]) == 'ZoneBC'

def test_iter_nodes_from_values():
  tree = PTy.to_node(yt)
  bc1 = PT.get_node_from_name(tree, 'bc1')
  nodes = list(PT.iter_nodes_from_values(bc1, ['BC1']))
  assert len(nodes) == 1
  assert PT.get_value(nodes[0]) == 'BC1'

def test_iter_children_from_values():
  tree = PTy.to_node(yt)
  bc1 = PT.get_node_from_name(tree, 'bc1')
  nodes = list(PT.iter_children_from_values(bc1, ['BC1']))
  assert len(nodes) == 1
  assert PT.get_value(nodes[0]) == 'BC1'

def test_iter_nodes_from_name_and_labels():
  tree = PTy.to_node(yt)
  nodes = list(PT.iter_nodes_from_name_and_labels(tree, ['FamilyName'], ['FamilyName_t']))
  assert len(nodes) >= 1
  for n in nodes:
      assert PT.get_name(n) == 'FamilyName'

# Une deuxième version de la fonction alias de recherche child par noms et labels.
# On la marque xfail car elle présente le même problème que l'autre.
@pytest.mark.xfail(reason="iter_children_from_name_and_labels manque le return de l'itérateur")
def test_iter_children_from_name_and_labels_duplicate():
  tree = PTy.to_node(yt)
  bc2 = PT.get_node_from_name(tree, 'bc2')
  nodes = list(PT.iter_children_from_name_and_labels(bc2, ['FamilyName'], ['FamilyName_t']))
  assert len(nodes) == 1
  assert PT.get_name(nodes[0]) == 'FamilyName'

# ---------------------------------------------------------------------------
# Tests sur les fonctions de recherche multiples (nodes)

def test_get_nodes_from_predicates_warning(capsys):
  tree = PTy.to_node(yt)
  nodes = PT.get_nodes_from_predicates(tree, 'bc*', caching=False)
  captured = capsys.readouterr().out
  assert "Warning: get_nodes_from_predicates forces caching to True." in captured
  assert isinstance(nodes, list)

def test_get_nodes_from_values():
  tree = PTy.to_node(yt)
  bc1 = PT.get_node_from_name(tree, 'bc1')
  nodes = PT.get_nodes_from_values(bc1, ['BC1'])
  assert isinstance(nodes, list)
  assert len(nodes) == 1
  assert PT.get_value(nodes[0]) == 'BC1'

def test_get_children_from_values():
  tree = PTy.to_node(yt)
  bc1 = PT.get_node_from_name(tree, 'bc1')
  nodes = PT.get_children_from_values(bc1, ['BC1'])
  assert isinstance(nodes, list)
  assert len(nodes) == 1
  assert PT.get_value(nodes[0]) == 'BC1'

def test_get_nodes_from_name_and_labels():
  tree = PTy.to_node(yt)
  bc1 = PT.get_node_from_name(tree, 'bc1')
  nodes = PT.get_nodes_from_name_and_labels(bc1, ['FamilyName'], ['FamilyName_t'])
  assert isinstance(nodes, list)
  assert len(nodes) == 1
  assert PT.get_name(nodes[0]) == 'FamilyName'

def test_get_children_from_name_and_labels():
  tree = PTy.to_node(yt)
  bc2 = PT.get_node_from_name(tree, 'bc2')
  nodes = PT.get_children_from_name_and_labels(bc2, ['FamilyName'], ['FamilyName_t'])
  assert isinstance(nodes, list)
  assert len(nodes) == 1
  assert PT.get_name(nodes[0]) == 'FamilyName'

# ---------------------------------------------------------------------------
# Test sur l'alias hérité getNodeFromPredicates
# Celui-ci passe par défaut le paramètre 'explore' qui n'est plus accepté.
# On marque ce test comme xfail pour signaler le problème.
@pytest.mark.xfail(reason="Legacy alias getNodeFromPredicates passe 'explore' et lève une TypeError")
def test_getNodeFromPredicates():
    tree = PTy.to_node(yt)
    node1 = PT.get_node_from_predicates(tree, 'FamilyName_t')
    node2 = PT.getNodeFromPredicates(tree, 'FamilyName_t')
    assert node1 == node2

def test_get_all_zone_t_from_zone_t():
  yaml_str = "MyZone Zone_t:"
  tree = PTy.to_node(yaml_str)
  zones = PT.get_all_Zone_t(tree)
  assert len(zones) == 1
  assert PT.get_name(zones[0]) == "MyZone"

def test_get_all_zone_t_from_cgnsbase_t():
  yaml_str = """
  Base CGNSBase_t:
    Zone1 Zone_t:
    Other Something_t:
    Zone2 Zone_t:
  """
  tree = PTy.to_node(yaml_str)
  zones = PT.get_all_Zone_t(tree)
  zone_names = [PT.get_name(n) for n in zones]
  assert len(zones) == 2
  assert "Zone1" in zone_names
  assert "Zone2" in zone_names

def test_get_all_zone_t_from_cgnstree_t():
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
  zone_names = [PT.get_name(n) for n in zones]
  assert len(zones) == 3
  assert set(zone_names) == {"Zone1", "ZoneExtra", "Zone2"}

def test_get_all_cgnsbase_t_from_cgnsbase_t():
    yaml_str = "BaseA CGNSBase_t:"
    tree = PTy.to_node(yaml_str)
    bases = PT.get_all_CGNSBase_t(tree)
    # On s'attend à retrouver un unique nœud, celui-ci étant la racine
    assert len(bases) == 1
    assert PT.get_name(bases[0]) == "BaseA"

def test_get_all_cgnsbase_t_from_cgnstree_t():
    yaml_str = """
    Tree CGNSTree_t:
      BaseA CGNSBase_t:
      BaseB CGNSBase_t:
      NotBase SomethingElse_t:
    """
    tree = PTy.to_node(yaml_str)
    bases = PT.get_all_CGNSBase_t(tree)
    base_names = [PT.get_name(b) for b in bases]
    assert len(bases) == 2
    assert set(base_names) == {"BaseA", "BaseB"}



# ---------------------------------------------------------------------------


def test_get_child_from_name_and_label():
  tree = PTy.to_node(yt)
  zonebc = get_ZoneBC(tree)
  bc1 = get_bc1(zonebc)
  node = PT.get_child_from_name_and_label(bc1, 'FamilyName', 'FamilyName_t')
  assert node is not None
  assert PT.get_name(node) == 'FamilyName'
  assert PT.get_label(node) == 'FamilyName_t'
  assert PT.get_child_from_name_and_label(bc1, 'NonExistent', 'FamilyName_t') is None

def test_get_children_from_name_and_label():
  tree = PTy.to_node(yt)
  zonebc = get_ZoneBC(tree)
  bc1 = get_bc1(zonebc)
  nodes = PT.get_children_from_name_and_label(bc1, 'FamilyName', 'FamilyName_t')
  assert isinstance(nodes, list)
  assert len(nodes) == 1
  node = nodes[0]
  assert PT.get_name(node) == 'FamilyName'
  assert PT.get_label(node) == 'FamilyName_t'

def test_iter_children_from_predicate():
  tree = PTy.to_node(yt)
  zonebc = get_ZoneBC(tree)
  it = PT.iter_children_from_predicate(zonebc, lambda n: PTp.match_name(n, "bc*"))
  nodes = list(it)
  assert isinstance(nodes, list)
  assert len(nodes) == 2
  names = [PT.get_name(n) for n in nodes]
  assert set(names) == {"bc1", "bc2"}

def test_get_child_from_predicates():
  tree = PTy.to_node(yt)
  zonebc = get_ZoneBC(tree)
  node = PT.get_child_from_predicates(zonebc, "BC_t")
  assert node is not None
  assert PT.get_name(node) == "bc1"

def test_get_node_from_labels():
  tree = PTy.to_node(yt)
  node = PT.get_node_from_labels(tree, ['FamilyName_t'])
  assert node is not None
  assert PT.get_value(node) == 'BC1'

def test_get_child_from_labels():
    tree = PTy.to_node(yt)
    zonebc = get_ZoneBC(tree)
    node = PT.get_child_from_labels(zonebc, ['BC_t'])
    assert node is not None
    assert PT.get_label(node) == 'BC_t'
    assert PT.get_name(node) == 'bc1'

def test_get_children_from_predicates():
  tree = PTy.to_node(yt)
  zonebc = get_ZoneBC(tree)
  nodes = PT.get_children_from_predicates(zonebc, "BC_t")
  assert isinstance(nodes, list)
  assert len(nodes) == 2
  names = [PT.get_name(n) for n in nodes]
  assert set(names) == {"bc1", "bc2"}

def test_get_nodes_from_names():
  tree = PTy.to_node(yt)
  nodes = PT.get_nodes_from_names(tree, ['bc1'])
  assert isinstance(nodes, list)
  assert len(nodes) == 1
  assert PT.get_name(nodes[0]) == "bc1"

def test_get_children_from_names():
  tree = PTy.to_node(yt)
  zonebc = get_ZoneBC(tree)
  nodes = PT.get_children_from_names(zonebc, ['bc1'])
  assert isinstance(nodes, list)
  assert len(nodes) == 1
  assert PT.get_name(nodes[0]) == "bc1"

def test_get_nodes_from_labels():
  tree = PTy.to_node(yt)
  nodes = PT.get_nodes_from_labels(tree, ['BC_t'])
  assert isinstance(nodes, list)
  assert len(nodes) == 2
  names = sorted([PT.get_name(n) for n in nodes])
  assert names == ["bc1", "bc2"]

def test_get_children_from_labels():
  tree = PTy.to_node(yt)
  zonebc = get_ZoneBC(tree)
  nodes = PT.get_children_from_labels(zonebc, ['BC_t'])
  assert isinstance(nodes, list)
  assert len(nodes) == 2
  names = sorted([PT.get_name(n) for n in nodes])
  assert names == ["bc1", "bc2"]
