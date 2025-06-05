import pytest

import maia.pytree as PT
from   maia.pytree import predicate as PD

from maia.pytree.yaml import parse_yaml_cgns

yt = """
FamilyBC FamilyBC_t:
  FamilyBCDataSet FamilyBCDataSet_t:
    RefStateFamilyBCDataSet ReferenceState_t:
      Density DataArray_t [1.1]:
      MomentumX DataArray_t [1.1]:
      MomentumY DataArray_t [0.]:
      MomentumZ DataArray_t [0.]:
      EnergyStagnationDensity DataArray_t [2.51]:
  """

def test_create():
  node = parse_yaml_cgns.to_node(yt)
  pattern = [lambda n : PT.get_label(n) == "ReferenceState_t", lambda n : PT.get_name(n) == "MomentumX"]
  
  walker = PT.NodeWalkers(node, pattern)

  assert walker.root == node
  assert walker.predicates == pattern

  walker.root = node[2][0] #Change root is allowed
  walker.predicates = [lambda n : PT.get_name(n) == "MomentumY"] #Change predicate is allowed
  with pytest.raises(TypeError):
    walker.predicates = "MomentumZ" #But non callable predicate are not allowed
    walker()

def test_search():
  yt = """
FamilyBC FamilyBC_t:
  FamilyBCDataSet FamilyBCDataSet_t:
    RefStateFamilyBCDataSet ReferenceState_t:
      Density DataArray_t [1.1]:
      MomentumX DataArray_t [1.1]:
      MomentumY DataArray_t [0.]:
      MomentumZ DataArray_t [0.]:
      EnergyStagnationDensity DataArray_t [2.51]:
    SomeData DataArray_t:
    """
  node = parse_yaml_cgns.to_node(yt)
  predicates = [lambda n : PT.get_label(n) == "ReferenceState_t", lambda n : PT.get_name(n) == "MomentumX"]

  walker = PT.NodeWalkers(node, predicates)
  assert walker()[0] == "MomentumX"

  predicates = [lambda n : PT.get_label(n) == "ReferenceState_t", lambda n : PT.get_label(n) == "DataArray_t"]
  walker = PT.NodeWalkers(node, predicates)
  assert walker()[0] == "Density" # Only first found is returned

  # Optionnal kwargs
  predicates = [lambda n : PT.get_label(n) == "ReferenceState_t", lambda n : PT.get_label(n) == "DataArray_t"]
  assert PT.NodeWalkers(node, predicates, depth=1)() is None #Not found because of depth=1
  root = PT.find_node_from_label(node, "FamilyBCDataSet_t")
  assert PT.NodeWalkers(root, predicates, depth=1)()[0] == "Density"

  # Specific options for each predicate
  patterns = [
    {'predicate': PD.label_matches("ReferenceState_t"), 'search':'dfs'},
    {'predicate': PD.name_matches('Density'), 'depth':1},
  ]
  assert PT.NodeWalkers(node, patterns)()[0] == 'Density'

  # With ancestors (specific options for each predicate)
  nodes = PT.NodeWalkers(node, patterns, ancestors=True)()
  assert [n[0] for n in nodes] == ['RefStateFamilyBCDataSet', 'Density']

  # With ancestors (shared options for all predicates)
  predicates = [lambda n : PT.get_label(n) == "ReferenceState_t", lambda n : PT.get_label(n) == "DataArray_t"]
  # None is found
  assert PT.NodeWalkers(node, predicates, depth=1, ancestors=True)() is None
  # First is found, but not second -> return None
  predicates = [lambda n : PT.get_label(n) == "FamilyBCDataSet_t", lambda n : PT.get_name(n) == "Pressure"]
  nodes = PT.NodeWalkers(node, predicates, depth=1, ancestors=True)()
  assert nodes is None