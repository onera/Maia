import pytest
import pytest_parallel
import numpy as np

import maia.pytree        as PT

import maia
from maia.pytree.yaml   import parse_yaml_cgns

from maia.algo.part import wall_distance as WD

def test_detect_wall_families():
  yt = """
  BaseA CGNSBase_t:
    SomeWall Family_t:
      FamilyBC FamilyBC_t "BCWallViscous":
    SomeNoWall Family_t:
      FamilyBC FamilyBC_t "BCFarfield":
  BaseB CGNSBase_t:
    SomeOtherWall Family_t:
      FamilyBC FamilyBC_t "BCWall":
  BaseC CGNSBase_t:
  """
  tree = parse_yaml_cgns.to_cgns_tree(yt)

  assert WD.detect_wall_families(tree) == ['SomeWall', 'SomeOtherWall']


# For U, we reuse the meshes defined in test_interpolate
from maia.algo.part.test.test_interpolate import src_part_0, src_part_1

@pytest.mark.skipif(not maia.pdma_enabled, reason="Require ParaDiGMA")
@pytest.mark.parametrize("perio", [True, False])
@pytest_parallel.mark.parallel(2)
class Test_wallDistance:
  def test_U(self, perio, comm):
    if comm.Get_rank() == 0:
      part_tree = parse_yaml_cgns.to_cgns_tree(src_part_0)
      expected_wd = [0.75, 0.25, 0.25, 0.75]
      expected_gnum = [1, 1, 2, 2]
    elif comm.Get_rank() == 1:
      part_tree = parse_yaml_cgns.to_cgns_tree(src_part_1)
      expected_wd = [0.75, 0.25, 0.75, 0.25]
      expected_gnum = [3, 3, 4, 4]
    base = PT.get_all_CGNSBase_t(part_tree)[0]
    base_family = PT.new_Family('WALL', family_bc='BCWall', parent=base)
    zone = PT.get_all_Zone_t(part_tree)[0]
    zone[0] += f'.P{comm.Get_rank()}.N0'

    # Add BC
    zone_bc = parse_yaml_cgns.to_node("""
      ZoneBC ZoneBC_t:
        BC BC_t "FamilySpecified":
          PointList IndexArray_t [[13,14]]:
          GridLocation GridLocation_t "FaceCenter":
          FamilyName FamilyName_t "WALL":
      """)
    PT.add_child(zone, zone_bc)

    # Test with family specification + propagation method + default out_fs_name
    WD.compute_wall_distance(part_tree, comm, method="propagation", families=["WALL"], perio=perio)

    fs = PT.get_child_from_name(zone, 'WallDistance')
    assert fs is not None and PT.Subset.GridLocation(fs) == 'CellCenter'
    for array in PT.iter_children_from_label(fs, 'DataArray_t'):
      assert array[1].shape == (4,)
    assert (PT.get_child_from_name(fs, 'TurbulentDistance')[1] == expected_wd).all()
    assert (PT.get_child_from_name(fs, 'ClosestEltGnum')[1] == expected_gnum).all()

    #Test with family detection + cloud method + custom fs name
    PT.rm_nodes_from_name(part_tree, 'WallDistance')
    WD.compute_wall_distance(part_tree, comm, method="cloud", out_fs_name='MyWallDistance', perio=perio)

    fs = PT.get_child_from_name(zone, 'MyWallDistance')
    assert fs is not None and PT.Subset.GridLocation(fs) == 'CellCenter'
    for array in PT.iter_children_from_label(fs, 'DataArray_t'):
      assert array[1].shape == (4,)
    assert (PT.get_child_from_name(fs, 'TurbulentDistance')[1] == expected_wd).all()
    assert (PT.get_child_from_name(fs, 'ClosestEltGnum')[1] == expected_gnum).all()

@pytest_parallel.mark.parallel(2)
def test_walldistance_vtx(comm):
  if comm.Get_rank() == 0:
    part_tree = parse_yaml_cgns.to_cgns_tree(src_part_0)
    expected_wd = [1, 0.5, 0, 1., 0.5, 0, 1., 0.5, 0, 1., 0.5, 0, 1., 0.5, 0, 1., 0.5, 0 ]
    expected_gnum = [1,1,1,1,1,1,2,2,2,1,1,1,1,1,1,2,2,2]
  elif comm.Get_rank() == 1:
    part_tree = parse_yaml_cgns.to_cgns_tree(src_part_1)
    expected_wd = [1, 0.5, 0, 1., 0.5, 0, 1., 0.5, 0, 1., 0.5, 0, 1., 0.5, 0, 1., 0.5, 0 ]
    expected_gnum = [3,3,3,3,3,3,4,4,4,1,1,1,1,1,1,2,2,2]
  base = PT.get_all_CGNSBase_t(part_tree)[0]
  base_family = PT.new_Family('WALL', family_bc='BCWall', parent=base)
  zone = PT.get_all_Zone_t(part_tree)[0]
  zone[0] += f'.P{comm.Get_rank()}.N0'

  # Add BC
  zone_bc = parse_yaml_cgns.to_node("""
    ZoneBC ZoneBC_t:
      BC BC_t "FamilySpecified":
        PointList IndexArray_t [[13,14]]:
        GridLocation GridLocation_t "FaceCenter":
        FamilyName FamilyName_t "WALL":
    """)
  PT.add_child(zone, zone_bc)

  WD.compute_wall_distance(part_tree, comm, method="cloud", point_cloud="Vertex", out_fs_name='MyWallDistance')

  fs = PT.get_child_from_name(zone, 'MyWallDistance')
  assert fs is not None and PT.Subset.GridLocation(fs) == 'Vertex'
  assert (PT.get_child_from_name(fs, 'TurbulentDistance')[1] == expected_wd).all()
  assert (PT.get_child_from_name(fs, 'ClosestEltGnum')[1] == expected_gnum).all()

