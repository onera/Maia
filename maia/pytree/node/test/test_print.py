import sys
import pytest

from maia.pytree.yaml import parse_yaml_cgns
from maia.pytree.node import print_tree

@pytest.mark.parametrize("verbose", [True, False])
def test_print_tree(capsys, verbose):
  yt = """MyBase CGNSBase_t [3,3]:
  Wall Family_t:
    BCWall FamilyBC_t:
  MyZone Zone_t I4 [16,6,0]:
    GridCoordinates GridCoordinates_t:
      Descriptor Descriptor_t "A very long description for this data":
      CoordinateX DataArray_t [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]:
"""
  if verbose:
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
  else:
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

  tree = parse_yaml_cgns.to_cgns_tree(yt)
  # We have to reput sys.stdout otherwise pytest does not capture output
  print_tree(tree, sys.stdout, no_colors=True, verbose=verbose)
  out, err = capsys.readouterr()
  assert out == expected_print_str
