from maia.utils.yaml.pretty_tree import pretty_string

def test_pretty_tree():
  yt = """MyBase [3,3]:
  Wall Family_t:
    BCWall FamilyBC_t:
  MyZone [9 4 0]:
    GridCoordinates:
      CoordinateX [0 1 2 0 1 2 0 1 2]:
"""

  expected_print_str = """MyBase [3,3]
├───Wall Family_t
│   └───BCWall FamilyBC_t
└───MyZone [9 4 0]
    └───GridCoordinates
        └───CoordinateX [0 1 2 0 1 2 0 1 2]
"""
  assert(pretty_string(yt) == expected_print_str)
