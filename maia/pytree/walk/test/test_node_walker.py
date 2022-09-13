import pytest

import maia.pytree as PT
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
  pattern = lambda n : PT.get_name(n) == "MomentumX"
  
  walker = PT.NodeWalker(node, pattern)

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
    walker = PT.NodeWalker(node, pattern, search="Fake") #Search must be in dfs, bfs

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
  predicate = lambda n: PT.get_label(n) == "DataArray_t"

  walker = PT.NodeWalker(node, predicate, search="dfs")
  assert walker()[0] == "Density" #Good deep first
  walker.search = "bfs"
  assert walker()[0] == "SomeData" #Explore level first

def test_sort():
  node = parse_yaml_cgns.to_node(yt)
  predicate = lambda n: PT.get_label(n) == "DataArray_t"

  walker = PT.NodeWalker(node, predicate, search="dfs")
  assert walker()[0] == "Density"
  walker.sort = PT.NodeWalker.BACKWARD
  assert walker()[0] == "EnergyStagnationDensity"

@pytest.mark.parametrize("search", ['dfs', 'bfs'])
def test_sort(search):
  node = parse_yaml_cgns.to_node(yt)
  predicate = lambda n: PT.get_name(n) == "RefStateFamilyBCDataSet"

  assert PT.NodeWalker(node, predicate, search)() is not None 
  assert PT.NodeWalker(node, predicate, search, depth=1)() is None  #Until 1
  assert PT.NodeWalker(node, predicate, search, depth=2)() is not None  #Until 2
  assert PT.NodeWalker(node, predicate, search, depth=[0,1])() is None  
  assert PT.NodeWalker(node, predicate, search, depth=[1,None])() is not None  #From 1 to end

  root = PT.NodeWalker(node, predicate, search)() #Change root
  assert PT.NodeWalker(root, predicate, search)() is not None  #Default include level 0
  assert PT.NodeWalker(root, predicate, search, depth=[1,None])() is None  #Default include level 0

