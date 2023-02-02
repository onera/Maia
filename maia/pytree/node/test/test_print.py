import sys
import pytest

from maia.pytree.yaml import parse_yaml_cgns
from maia.pytree.node import print_tree

class Test_print_tree:
  yt = """MyBase CGNSBase_t [3,3]:
  Wall Family_t:
    BCWall FamilyBC_t:
  MyZone Zone_t I4 [16,6,0]:
    GridCoordinates GridCoordinates_t:
      Descriptor Descriptor_t "A very long description for this data":
      CoordinateX DataArray_t [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]:
"""
  tree = parse_yaml_cgns.to_cgns_tree(yt)

  def test_plain(self, capsys):
    expected_print_str = """\
CGNSTree CGNSTree_t 
├───MyBase CGNSBase_t I4 [3 3]
│   ├───Wall Family_t 
│   │   └───BCWall FamilyBC_t 
│   └───MyZone Zone_t I4 [16  6  0]
│       └───GridCoordinates GridCoordinates_t 
│           ├───Descriptor Descriptor_t "A very long descrip[...]data"
│           └───CoordinateX DataArray_t I4 (16,)
└───CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]
"""
    # We have to reput sys.stdout otherwise pytest does not capture output
    print_tree(self.tree, sys.stdout, colors=False)
    out, err = capsys.readouterr()
    assert out == expected_print_str

  def test_verbose(self, capsys):
    expected_print_str = """\
CGNSTree CGNSTree_t 
├───MyBase CGNSBase_t I4 [3 3]
│   ├───Wall Family_t 
│   │   └───BCWall FamilyBC_t 
│   └───MyZone Zone_t I4 [16  6  0]
│       └───GridCoordinates GridCoordinates_t 
│           ├───Descriptor Descriptor_t 
│           │   "A very long description for this data"
│           └───CoordinateX DataArray_t I4 (16,)
│               [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3]
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
│       ╵╴╴╴ (1 child masked)
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
            └───Descriptor Descriptor_t "A very long descrip[...]data"
"""
    # We have to reput sys.stdout otherwise pytest does not capture output
    print_tree(self.tree, sys.stdout, colors=False, print_if = lambda n: n[3] == 'Descriptor_t')
    out, err = capsys.readouterr()
    assert out == expected_print_str
