import pytest
import Converter.Internal as I
from maia.utils   import parse_yaml_cgns
from maia.cgns_io import fix_tree

def test_fix_point_ranges():
  yt = """
Base0 CGNSBase_t [3,3]:
  ZoneA Zone_t:
    ZGC ZoneGridConnectivity_t:
      matchAB GridConnectivity1to1_t "ZoneB":
        PointRange IndexArray_t [[17,17],[3,9],[1,5]]:
        PointRangeDonor IndexArray_t [[7,1],[9,9],[1,5]]:
        Transform "int[IndexDimension]" [-2,-1,-3]:
  ZoneB Zone_t:
    ZGC ZoneGridConnectivity_t:
      matchBA GridConnectivity1to1_t "Base0/ZoneA":
        PointRange IndexArray_t [[7,1],[9,9],[1,5]]:
        PointRangeDonor IndexArray_t [[17,17],[3,9],[1,5]]:
        Transform "int[IndexDimension]" [-2,-1,-3]:
"""
  size_tree = parse_yaml_cgns.to_cgns_tree(yt)
  fix_tree.fix_point_ranges(size_tree)
  gcA = I.getNodeFromName(size_tree, 'matchAB')
  gcB = I.getNodeFromName(size_tree, 'matchBA')
  assert (I.getNodeFromName1(gcA, 'PointRange')[1]      == [[17,17], [3,9], [1,5]]).all()
  assert (I.getNodeFromName1(gcA, 'PointRangeDonor')[1] == [[ 7, 1], [9,9], [5,1]]).all()
  assert (I.getNodeFromName1(gcB, 'PointRange')[1]      == [[ 7, 1], [9,9], [5,1]]).all()
  assert (I.getNodeFromName1(gcB, 'PointRangeDonor')[1] == [[17,17], [3,9], [1,5]]).all()

#def test_load_grid_connectivity_property():
  #Besoin de charger depuis un fichier, comment tester ?
