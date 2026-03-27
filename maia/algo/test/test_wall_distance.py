import pytest
import pytest_parallel

import numpy as np

import maia
import maia.pytree        as PT

from maia.algo import wall_distance as WD

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
  tree = PT.yaml.to_cgns_tree(yt)

  assert WD.detect_wall_families(tree) == ['SomeWall', 'SomeOtherWall']

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("partitioned", [False, True])
def test_wall_distance_no_wall(partitioned, comm):
  tree = maia.factory.generate_dist_block(4, "Poly", comm)
  if partitioned:
    tree = maia.factory.partition_dist_tree(tree, comm)
  WD.compute_wall_distance(tree, comm)

  for zone in PT.get_all_Zone_t(tree):
    fs_node = PT.get_child_from_name(zone, 'WallDistance')
    assert PT.Container.GridLocation(fs_node) == 'CellCenter'
    assert (PT.get_child_from_name(fs_node, 'TurbulentDistance')[1] == np.inf).all()
    assert (PT.get_child_from_name(fs_node, 'ClosestEltGnum')[1] == -1).all()
    assert (PT.get_child_from_name(fs_node, 'ClosestEltDomId')[1] == -1).all()
    


@pytest_parallel.mark.parallel(1)
@pytest.mark.parametrize("partitioned", [False, True])
def test_twice(partitioned, comm):
  tree = maia.factory.generate_dist_block([5,4], 'S', comm)
  # Set some BC wall
  bc = PT.get_node_from_name(tree, 'Xmax')
  PT.set_value(bc, 'BCWall')

  if partitioned:
    tree = maia.factory.partition_dist_tree(tree, comm)

  WD.compute_wall_distance(tree, comm)
  WD.compute_wall_distance(tree, comm)
