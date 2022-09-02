import pytest

import maia.pytree as PT
from maia.utils.yaml import parse_yaml_cgns

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
  pattern = lambda n : PT.get_name(n).startswith("Momentum")
  
  walker = PT.NodesWalker(node, pattern)

  assert walker.root == node
  assert walker.predicate == pattern

  walker.root = node[2][0] #Change root is allowed
  walker.predicate = lambda n : PT.get_name(n) == "MomentumY" #Change predicate is allowed
  with pytest.raises(TypeError):
    walker.predicate = "MomentumZ" #But non callable predicate are not allowed

  assert walker.depth == [0, None] #Default depth
  walker.depth = [2,8] #Depth can be changed
  walker.depth = 1
  with pytest.raises(ValueError):
    walker.depth = "Toto" #But must be a range

  assert walker.search == "dfs"
  walker.search = "bfs"
  with pytest.raises(ValueError):
    walker.search = "Fake"
  assert walker.explore == "shallow"
  walker.explore = "deep"
  with pytest.raises(ValueError):
    walker.explore = "Fake"
  assert walker.caching == False
  walker.caching = True
  with pytest.raises(TypeError):
    walker.caching = "Fake"

@pytest.mark.parametrize("explore", ["deep", "shallow"])
def test_explore(explore):
  node = parse_yaml_cgns.to_node(yt)
  predicate = lambda n: PT.get_label(n) == "DataArray_t"
  walker = PT.NodesWalker(node, predicate, explore=explore)
  if explore == "deep":
    assert [n[0] for n in walker()] == ["Density", "Density2", "MomentumX", "SomeData"]
  else:
    assert [n[0] for n in walker()] == ["Density", "MomentumX", "SomeData"]


@pytest.mark.parametrize("search", ["bfs", "dfs"])
def test_search(search):
  node = parse_yaml_cgns.to_node(yt)
  predicate = lambda n: PT.get_label(n) == "DataArray_t"

  walker = PT.NodesWalker(node, predicate, explore="deep", search=search)
  if search == "dfs":
    assert [n[0] for n in walker()] == ["Density", "Density2", "MomentumX", "SomeData"]
  elif search == "bfs":
    assert [n[0] for n in walker()] == ["SomeData", "Density", "MomentumX", "Density2"]

@pytest.mark.parametrize("caching", [False, True])
def test_caching(caching):
  node = parse_yaml_cgns.to_node(yt)
  predicate = lambda n: 'FamilyBC' in PT.get_name(n) and not 'Ref' in PT.get_name(n)
  #TODO si explore == shallow, on en capte 2 et pas 1 ??
  walker = PT.NodesWalker(node, predicate, explore='deep', caching=caching)
  if caching:
    assert [PT.get_name(n) for n in walker()] == ['FamilyBC', 'FamilyBCDataSet']
  else:
    assert walker.cache == []

@pytest.mark.parametrize("caching", [False, True])
def test_apply(caching):
  node = parse_yaml_cgns.to_node(yt)
  predicate = lambda n: PT.get_label(n) == "DataArray_t"
  
  walker = PT.NodesWalker(node, predicate, caching=caching)
  walker.apply(lambda n : PT.set_name(n, PT.get_name(n).upper()))
  assert [PT.get_name(n) for n in walker()] == ['DENSITY', 'MOMENTUMX', 'SOMEDATA']
  if caching: #Cache is updated too
    assert [PT.get_name(n) for n in walker.cache] == ['DENSITY', 'MOMENTUMX', 'SOMEDATA']

def test_sort():
  node = parse_yaml_cgns.to_node(yt)
  predicate = lambda n: PT.get_label(n) == "DataArray_t"

  walker = PT.NodesWalker(node, predicate, explore='deep', search='dfs')
  assert [PT.get_name(n) for n in walker()] == ['Density', 'Density2', 'MomentumX', 'SomeData']
  walker.sort = PT.NodesWalker.BACKWARD
  assert [PT.get_name(n) for n in walker()] == ['SomeData', 'MomentumX', 'Density', 'Density2']
  
  walker = PT.NodesWalker(node, predicate, explore='deep', search='bfs')
  assert [PT.get_name(n) for n in walker()] == ['SomeData', 'Density', 'MomentumX', 'Density2']
  walker.sort = PT.NodesWalker.BACKWARD
  assert [PT.get_name(n) for n in walker()] == ['SomeData', 'MomentumX', 'Density', 'Density2']

def test_depth():
  node = parse_yaml_cgns.to_node(yt)
  predicate = lambda n: PT.get_label(n) == "DataArray_t"

  walker = PT.NodesWalker(node, predicate, explore='deep', search='bfs')
  assert [PT.get_name(n) for n in walker()] == ['SomeData', 'Density', 'MomentumX', 'Density2']
  walker.depth = [0, 2]
  assert [PT.get_name(n) for n in walker()] == ['SomeData']
  walker.depth = [3, None]
  assert [PT.get_name(n) for n in walker()] == ['Density', 'MomentumX', 'Density2']
  walker.depth = [3, 3]
  assert [PT.get_name(n) for n in walker()] == ['Density', 'MomentumX']

