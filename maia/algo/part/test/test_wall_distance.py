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
def test_wall_distance_U(perio, comm):
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

  # Test with propagation method + default out_fs_name
  WD.compute_wall_distance(part_tree, comm, method="propagation", perio=perio)

  fs = PT.get_child_from_name(zone, 'WallDistance')
  assert fs is not None and PT.Subset.GridLocation(fs) == 'CellCenter'
  for array in PT.iter_children_from_label(fs, 'DataArray_t'):
    assert array[1].shape == (4,)
  assert (PT.get_child_from_name(fs, 'TurbulentDistance')[1] == expected_wd).all()
  assert (PT.get_child_from_name(fs, 'ClosestEltGnum')[1] == expected_gnum).all()

  #Test with cloud method + custom fs name
  PT.rm_nodes_from_name(part_tree, 'WallDistance')
  WD.compute_wall_distance(part_tree, comm, method="cloud", out_fs_name='MyWallDistance', perio=perio)

  fs = PT.get_child_from_name(zone, 'MyWallDistance')
  assert fs is not None and PT.Subset.GridLocation(fs) == 'CellCenter'
  for array in PT.iter_children_from_label(fs, 'DataArray_t'):
    assert array[1].shape == (4,)
  assert (PT.get_child_from_name(fs, 'TurbulentDistance')[1] == expected_wd).all()
  assert (PT.get_child_from_name(fs, 'ClosestEltGnum')[1] == expected_gnum).all()

@pytest_parallel.mark.parallel(2)
def test_projection_to(comm):
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

  WD.compute_projection_to(part_tree, lambda n: PT.get_label(n) == 'BC_t', comm)
  PT.print_tree(part_tree)

  fs = PT.get_child_from_name(zone, 'SurfDistance')
  assert fs is not None and PT.Subset.GridLocation(fs) == 'CellCenter'
  assert (PT.get_child_from_name(fs, 'Distance')[1] == expected_wd).all()
  assert (PT.get_child_from_name(fs, 'ClosestEltGnum')[1] == expected_gnum).all()

@pytest_parallel.mark.parallel(2)
def test_walldistance_elts(comm):
  tree = maia.factory.generate_dist_block(3, 'TETRA_4', comm)
  # Set some BC wall
  for name in ['Xmin', 'Ymin', 'Zmin']:
    bc = PT.get_node_from_name(tree, 'Xmin')
    PT.set_value(bc, 'BCWall')

  ptree = maia.factory.partition_dist_tree(tree, comm)
  WD.compute_wall_distance(ptree, comm)
  maia.transfer.part_tree_to_dist_tree_all(tree, ptree, comm)
  
  if comm.Get_rank() == 0:
    expected_wd = [0.125,0.625,0.75, 0.375,0.875,0.875,0.875,0.75, 0.875,0.375,0.625,0.375,
                   0.875,0.375,0.25, 0.875,0.75, 0.625,0.625,0.875]
    expected_gnum = [1, 5, 3, 5, 6, 4, 5, 5, 3, 3, 3, 1, 1, 1, 1, 2, 1, 1, 1, 7]
  elif comm.Get_rank() == 1:
    expected_wd = [0.125,0.25, 0.125,0.375,0.875,0.125,0.625,0.625,0.375,0.25, 0.625,0.375,
                   0.25, 0.125,0.125,0.625,0.125,0.375,0.125,0.75]
    expected_gnum = [2, 7, 8, 7, 8, 7, 7, 7, 7, 5, 5, 3, 3, 3, 4, 3, 6, 5, 5, 7]

  assert (PT.get_node_from_name(ptree, 'TurbulentDistance')[1] == expected_wd).all()
  assert (PT.get_node_from_name(ptree, 'ClosestEltGnum')[1] == expected_gnum).all()
    

@pytest_parallel.mark.parallel(2)
def test_walldistance_perio(comm):
  #Case generation
  dist_treeU = maia.factory.generate_dist_block(3, "Poly", comm, edge_length=2.)
  coordX, coordY, coordZ = PT.Zone.coordinates(PT.get_node_from_label(dist_treeU, 'Zone_t'))
  coordX += coordY
  coordY -= coordZ

  for bc in PT.get_nodes_from_label(dist_treeU, "BC_t"):
    family_bc = "BCWall" if PT.get_name(bc) == 'Ymin' else None
    PT.new_Family(f"Fam_{PT.get_name(bc)}", family_bc=family_bc, parent=PT.get_child_from_label(dist_treeU, "CGNSBase_t"))
    PT.new_node(name='FamilyName', label='FamilyName_t', value=f"Fam_{PT.get_name(bc)}", parent=bc)

  maia.algo.dist.connect_1to1_families(dist_treeU, ('Fam_Xmin', 'Fam_Xmax'), comm,
          periodic={'translation' : np.array([2.,0.,0.])})

  maia.algo.dist.connect_1to1_families(dist_treeU, ('Fam_Zmin', 'Fam_Zmax'), comm,
          periodic={'translation' : np.array([0.,-2.,2.])})

  zone_paths1 = maia.pytree.predicates_to_paths(dist_treeU, "CGNSBase_t/Zone_t")
  jn_paths_for_dupl1 = [['Base/zone/ZoneGridConnectivity/Xmin_0'],['Base/zone/ZoneGridConnectivity/Xmax_0']]
  maia.algo.dist.duplicate_from_periodic_jns(dist_treeU, zone_paths1, jn_paths_for_dupl1, 1, comm)

  zone_paths2 = maia.pytree.predicates_to_paths(dist_treeU, "CGNSBase_t/Zone_t")
  jn_paths_for_dupl2 = [['Base/zone.D0/ZoneGridConnectivity/Zmin_0','Base/zone.D1/ZoneGridConnectivity/Zmin_0'],
                        ['Base/zone.D0/ZoneGridConnectivity/Zmax_0','Base/zone.D1/ZoneGridConnectivity/Zmax_0']]
  maia.algo.dist.duplicate_from_periodic_jns(dist_treeU, zone_paths2, jn_paths_for_dupl2, 1, comm)

  part_tree = maia.factory.partition_dist_tree(dist_treeU, comm)

  # Test with family specification
  WD.compute_wall_distance(part_tree, comm)

  expected_wd     = [0.35355339, 0.35355339, 1.06066017, 1.06066017,
                     0.35355339, 0.35355339, 1.06066017, 1.06066017]
  if comm.rank == 0:
    expected_gnum   = [[ 1, 4, 3, 126,  2, 3, 92, 37],
                       [37, 6, 5,  98, 38, 5, 98,  5]]
    expected_dom_id = [[0, 0, 0, 3, 0, 0, 2, 1],
                       [1, 0, 0, 2, 1, 0, 2, 0]]
  elif comm.rank == 1:
    expected_gnum   = [[ 73, 76, 75, 38,  74, 75, 4, 109],
                       [109, 78, 77,  6, 110, 77, 6,  77]]
    expected_dom_id = [[2, 2, 2, 1, 2, 2, 0, 3],
                       [3, 2, 2, 0, 3, 2, 0, 2]]

  for z, zone in enumerate(PT.get_all_Zone_t(part_tree)):
    fs = PT.get_child_from_name_and_label(zone, 'WallDistance', 'FlowSolution_t')
    assert fs is not None and PT.Subset.GridLocation(fs) == 'CellCenter'
    assert np.allclose(PT.get_value(PT.get_child_from_name(fs, 'TurbulentDistance')), expected_wd, rtol=1e-10)
    assert (PT.get_value(PT.get_child_from_name(fs, 'ClosestEltGnum'))  == expected_gnum[z]).all()
    assert (PT.get_value(PT.get_child_from_name(fs, 'ClosestEltDomId')) == expected_dom_id[z]).all()

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

