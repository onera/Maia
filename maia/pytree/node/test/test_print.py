import sys
import pytest

from maia.pytree.yaml import parse_yaml_cgns
from maia.pytree.node import print_tree

class Test_print_tree:
  yt = """MyBase CGNSBase_t [3,3]:
  Wall Family_t:
    BCWall FamilyBC_t:
  MyZone Zone_t I4 [16,6,0]:
    ZoneType ZoneType_t "Unstructured":
    GridCoordinates GridCoordinates_t:
      Descriptor Descriptor_t "A very long description for this data":
      CoordinateX DataArray_t [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]:
        DimensionalUnit DimensionalUnits_t ['Kilogram', 'Meter', 'Second', 'Kelvin', 'Radian']:
"""
  tree = parse_yaml_cgns.to_cgns_tree(yt)

  def test_plain(self, capsys):
    expected_print_str = """\
CGNSTree CGNSTree_t 
├───MyBase CGNSBase_t I4 [3 3]
│   ├───Wall Family_t 
│   │   └───BCWall FamilyBC_t 
│   └───MyZone Zone_t I4 [16  6  0]
│       ├───ZoneType ZoneType_t "Unstructured"
│       └───GridCoordinates GridCoordinates_t 
│           ├───Descriptor Descriptor_t "A very lo[...]data"
│           └───CoordinateX DataArray_t I4 (16,)
│               └───DimensionalUnit DimensionalUnits_t ["Kilogram" ... "Radian"]
└───CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]
"""
    # We have to reput sys.stdout otherwise pytest does not capture output
    print_tree(self.tree, sys.stdout, colors=False)
    out, err = capsys.readouterr()
    assert out == expected_print_str

  def test_file(self, tmp_path):
    print_tree(self.tree, tmp_path/'tree.txt')
    
    expected_str = """\
CGNSTree CGNSTree_t 
├───MyBase CGNSBase_t I4 [3 3]
│   ├───Wall Family_t 
│   │   └───BCWall FamilyBC_t 
│   └───MyZone Zone_t I4 [16  6  0]
│       ├───ZoneType ZoneType_t "Unstructured"
│       └───GridCoordinates GridCoordinates_t 
│           ├───Descriptor Descriptor_t "A very lo[...]data"
│           └───CoordinateX DataArray_t I4 (16,)
│               └───DimensionalUnit DimensionalUnits_t ["Kilogram" ... "Radian"]
└───CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]
"""
    with open(tmp_path / 'tree.txt') as f:
      assert f.read() == expected_str

  def test_verbose(self, capsys):
    expected_print_str = """\
CGNSTree CGNSTree_t 
├───MyBase CGNSBase_t I4 [3 3]
│   ├───Wall Family_t 
│   │   └───BCWall FamilyBC_t 
│   └───MyZone Zone_t I4 [16  6  0]
│       ├───ZoneType ZoneType_t "Unstructured"
│       └───GridCoordinates GridCoordinates_t 
│           ├───Descriptor Descriptor_t 
│           │   "A very long description for this data"
│           └───CoordinateX DataArray_t I4 (16,)
│               [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3]
│               └───DimensionalUnit DimensionalUnits_t 
│                   ["Kilogram" "Meter" "Second" "Kelvin" "Radian"]
└───CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]
"""
    print_tree(self.tree, sys.stdout, colors=False, verbose=True)
    out, err = capsys.readouterr()
    assert out == expected_print_str

  def test_maxdepth(self, capsys):
    expected_print_str = """\
CGNSTree CGNSTree_t 
├───MyBase CGNSBase_t I4 [3 3]
│   ├───Wall Family_t 
│   │   ╵╴╴╴ (1 child masked)
│   └───MyZone Zone_t I4 [16  6  0]
│       ╵╴╴╴ (2 children masked)
└───CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]
"""
    # We have to reput sys.stdout otherwise pytest does not capture output
    print_tree(self.tree, sys.stdout, colors=False, max_depth=2)
    out, err = capsys.readouterr()
    assert out == expected_print_str

  def test_predicate(self, capsys):
    expected_print_str = """\
CGNSTree CGNSTree_t 
└───MyBase CGNSBase_t I4 [3 3]
    └───MyZone Zone_t I4 [16  6  0]
        └───GridCoordinates GridCoordinates_t 
            └───Descriptor Descriptor_t "A very lo[...]data"
"""
    # We have to reput sys.stdout otherwise pytest does not capture output
    print_tree(self.tree, sys.stdout, colors=False, print_if = lambda n: n[3] == 'Descriptor_t')
    out, err = capsys.readouterr()
    assert out == expected_print_str




def test_string_that_is_long_but_not_a_lot(capsys):
  y_desc = 'Descriptor Descriptor_t "My description node":' # 19 chars, but length is 20 in CGNS (to account for the ending \0)
  desc = parse_yaml_cgns.to_node(y_desc)

  expected_print_str = """\
Descriptor Descriptor_t "My descri[...]node"
"""

  # We have to reput sys.stdout otherwise pytest does not capture output
  print_tree(desc, sys.stdout, colors=False)
  out, err = capsys.readouterr()
  assert out == expected_print_str

@pytest.mark.parametrize('verbose', [False, True])
def test_string_3d(verbose, capsys):
  node = parse_yaml_cgns.to_node("""
    BaseIterativeData BaseIterativeData_t [2]:
      TimeValues DataArray_t [0., 1.]:
      FamilyPointers DataArray_t [["F1", "F2"], ["F1", "F2", "F3"]]:
      ZonePointers DataArray_t [["Zone.P0.N0", "Zone.P0.N1"], ["Zone.P0.N0", "Zone.P0.N1", "Zone.P0.N2"]]:
      NumberOfZones DataArray_t [2,3]:
  """)
  if verbose:
    zone_pointer = """\
├───ZonePointers DataArray_t [
│   ["Zone.P0.N0" "Zone.P0.N1" ""]
│   ["Zone.P0.N0" "Zone.P0.N1" "Zone.P0.N2"]]"""
  else:
    zone_pointer = """├───ZonePointers DataArray_t [["Zone.P0.N0" ...] ...]]"""

  expected_print_str = f"""\
BaseIterativeData BaseIterativeData_t I4 [2]
├───TimeValues DataArray_t R4 [0. 1.]
├───FamilyPointers DataArray_t [["F1" "F2" ""] ["F1" "F2" "F3"]]
{zone_pointer}
└───NumberOfZones DataArray_t I4 [2 3]
"""

  print_tree(node, sys.stdout, verbose=verbose, colors=False)
  out, err = capsys.readouterr()
  assert out == expected_print_str