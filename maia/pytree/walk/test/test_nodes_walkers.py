import pytest

import maia.pytree as PT
from   maia.pytree.walk import predicate as PD

from maia.pytree.yaml import parse_yaml_cgns

yt = """
FamilyBC FamilyBC_t:
  FamilyBCDataSet FamilyBCDataSet_t:
    RefStateFamilyBCDataSet ReferenceState_t:
      Density DataArray_t [1.1]:
        Density2 DataArray_t [1.1]:
      MomentumX DataArray_t [1.1]:
    SomeData DataArray_t:
"""

def test_create():
  node = parse_yaml_cgns.to_node(yt)
  pattern = [lambda n : PT.get_label(n, "FamilyBCDataSet"), lambda n : PT.get_name(n).startswith("Momentum")]
  
  walker = PT.NodesWalkers(node, pattern, caching=True)

  assert walker.root == node
  assert walker.predicates == pattern

  walker.root = node[2][0] #Change root is allowed
  walker.predicates = [lambda n : PT.get_name(n) == "MomentumY"] #Change predicate is allowed
  # But predicates must be callable
  walker.predicates = ["MomentumY"]
  with pytest.raises(TypeError):
    walker()

  assert walker.ancestors == False
  walker.ancestors = True
  with pytest.raises(TypeError):
    walker.ancestors = "Nope"
  

def test_simple():
  node = parse_yaml_cgns.to_node(yt)

  walker = PT.NodesWalker(node, lambda n: PT.get_label(n) == "DataArray_t", explore='deep')
  assert PT.get_names(walker()) == ['Density', 'Density2', 'MomentumX', 'SomeData']

  predicates = [lambda n : PT.get_label(n) == "ReferenceState_t", lambda n : PT.get_label(n) == "DataArray_t"]
  walker = PT.NodesWalkers(node, predicates)
  assert 'SomeData' not in PT.get_names(walker())

@pytest.mark.parametrize("caching", [False, True])
def test_ancestors(caching):
  node = parse_yaml_cgns.to_node(yt)
  predicates = [lambda n : PT.get_label(n) == "ReferenceState_t", lambda n : PT.get_label(n) == "DataArray_t"]

  assert PT.get_names(PT.NodesWalkers(node, predicates, caching=caching, ancestors=False)()) == ['Density', 'MomentumX']
  for nodes in PT.NodesWalkers(node, predicates, caching=caching, ancestors=True)():
    assert len(nodes) == 2 and PT.get_name(nodes[0]) == "RefStateFamilyBCDataSet"

def test_kwargs():
  node = parse_yaml_cgns.to_node(yt)
  # By default kwargs are applied to each predicate
  predicates = [lambda n : PT.get_label(n) == "ReferenceState_t", lambda n : PT.get_label(n) == "DataArray_t"]

  assert PT.get_names(PT.NodesWalkers(node, predicates, depth=1)()) == [] #First predicate is not found
  root = PT.get_node_from_name(node, "RefStateFamilyBCDataSet")
  assert PT.get_names(PT.NodesWalkers(root, predicates, depth=1)()) == ['Density', 'MomentumX'] 

  #We can set specific kwargs for each predicate
  predicates = [{'predicate':lambda n : PT.get_label(n) == "ReferenceState_t", 'depth':None},
                {'predicate':lambda n : PT.get_label(n) == "DataArray_t", 'depth':1, 'explore':'deep'}]
  assert PT.get_names(PT.NodesWalkers(node, predicates)()) == ['Density', 'MomentumX'] 
  
  for nodes in PT.NodesWalkers(node, predicates, ancestors=True)():
    assert len(nodes) == 2 and PT.get_name(nodes[0]) == "RefStateFamilyBCDataSet"


def test_nodes_walkers_pattern():
  yt = """
FamilyBCDataSet FamilyBCDataSet_t:
  RefStateFamilyBCDataSet ReferenceState_t:
    Density DataArray_t [1.1]:
    MomentumX DataArray_t [1.1]:
    MomentumY DataArray_t [0.]:
    MomentumZ DataArray_t [0.]:
    EnergyStagnationDensity DataArray_t [2.51]:
  """
  tree = parse_yaml_cgns.to_cgns_tree(yt)

  names = ["Density", "MomentumX", "MomentumY", "MomentumZ", "EnergyStagnationDensity"]
  check_name = lambda name : [
    {'predicate': lambda n: PD.match_label(n, "ReferenceState_t"), 'depth':0, 'caching':False},
    {'predicate': lambda n: PD.match_name(n, name), 'depth':1, 'caching':False},
  ]
  patterns = [check_name(name) for name in names]

  root = PT.request_node_from_label(tree, "FamilyBCDataSet_t")
  results = [n for pattern in patterns for n in PT.NodesWalkers(root, pattern)() ]
  assert(not bool(results))

  root = PT.request_node_from_label(tree, "ReferenceState_t")
  results = [n for pattern in patterns for n in PT.NodesWalkers(root, pattern)() ]
  assert([PT.get_name(n) for n in results] == names)

  check_name = lambda name : [
    {'predicate': lambda n: PD.match_label(n, "ReferenceState_t"), 'depth':1, 'caching':False},
    {'predicate': lambda n: PD.match_name(n, name), 'depth':1, 'caching':False},
  ]
  patterns = [check_name(name) for name in names]
  root = PT.request_node_from_label(tree, "FamilyBCDataSet_t")
  results = [n for pattern in patterns for n in PT.NodesWalkers(root, pattern)() ]
  assert([PT.get_name(n) for n in results] == names)

  check_name = lambda name : [
    {'predicate': lambda n: PD.match_str_label(n, "ReferenceState_t"), 'depth':None, 'caching':False},
    {'predicate': lambda n: PD.match_name(n, name), 'depth':1, 'caching':False},
  ]
  patterns = [check_name(name) for name in names]
  root = PT.request_node_from_label(tree, "FamilyBCDataSet_t")
  results = [n for pattern in patterns for n in PT.NodesWalkers(root, pattern)() ]
  assert([PT.get_name(n) for n in results] == names)

  check_name = lambda name : [
    {'predicate': lambda n: PD.match_label(n, "ReferenceState_t"), 'caching':False},
    {'predicate': lambda n: PD.match_name(n, name), 'depth':1, 'caching':False},
  ]
  patterns = [check_name(name) for name in names]
  root = PT.request_node_from_label(tree, "FamilyBCDataSet_t")
  results = [n for pattern in patterns for n in PT.NodesWalkers(root, pattern)() ]
  assert([PT.get_name(n) for n in results] == names)
