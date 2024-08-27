import pytest
import pytest_parallel
import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.algo import geometry
from maia.utils import np_utils

def generate_dist_line(comm):
  tree = PT.yaml.to_cgns_tree("""
  zone Zone_t [[6,5,0]]:
    ZoneType ZoneType_t "Unstructured":
    GridCoordinates GridCoordinates_t:
      CoordinateX DataArray_t R8 [0, 0.2, 0.4, 0.6, 0.8, 1.]:
      CoordinateY DataArray_t R8 [0, 0.2, 0.4, 0.6, 0.8, 1.]:
      CoordinateZ DataArray_t R8 [0, 0.2, 0.4, 0.6, 0.8, 1.]:
    BAR_2 Elements_t [3,0]:
      ElementRange IndexRange_t [1, 5]:
      ElementConnectivity DataArray_t [1,2, 2,3, 3,4, 4,5, 5,6]:
  """)
  return maia.factory.full_to_dist_tree(tree, comm)

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("parallel", ["dist", "part"])
def test_compute_centers(parallel, comm):

  def rename_append(parent, child_dict):
    for zname, t in child_dict.items():
      zone = PT.get_node_from_label(t, 'Zone_t')
      PT.set_name(zone, zname)
      PT.add_child(parent, zone)

  # Create a fake tree containing all zone / mesh kind
  tree = PT.new_CGNSTree() 

  trees3d = {'3d_poly'   : maia.factory.generate_dist_block(11, 'Poly', comm),
             '3d_elmts'  : maia.factory.generate_dist_block(11, 'TETRA_4', comm),
             '3d_struct' : maia.factory.generate_dist_block([5,3,6], 'S', comm)}
  trees2d = {'2d_poly'   : maia.factory.generate_dist_block(11, 'QUAD_4', comm),
             '2d_elmts'  : maia.factory.generate_dist_block(11, 'TRI_3', comm),
             '2d_struct' : maia.factory.generate_dist_block([11,11,1], 'S', comm)}
  trees1d = {'1d_elmts'  : generate_dist_line(comm)}

  maia.algo.dist.convert_elements_to_ngon(trees2d['2d_poly'], comm)

  trees2d_xy = {f"{key}_xy" : PT.deep_copy(tree) for key, tree in trees2d.items()}
  trees1d_xy = {f"{key}_xy" : PT.deep_copy(tree) for key, tree in trees1d.items()}
  trees1d_x  = {f"{key}_x"  : PT.deep_copy(tree) for key, tree in trees1d.items()}

  base = PT.new_CGNSBase('Base3D', cell_dim=3, phy_dim=3, parent=tree)
  rename_append(base, trees3d)
  base = PT.new_CGNSBase('Base2D', cell_dim=2, phy_dim=3, parent=tree)
  rename_append(base, trees2d)
  base = PT.new_CGNSBase('Base2D_XY', cell_dim=2, phy_dim=2, parent=tree)
  rename_append(base, trees2d_xy)
  base = PT.new_CGNSBase('Base1D', cell_dim=1, phy_dim=3, parent=tree)
  rename_append(base, trees1d)
  base = PT.new_CGNSBase('Base1D_XY', cell_dim=1, phy_dim=2, parent=tree)
  rename_append(base, trees1d_xy)
  base = PT.new_CGNSBase('Base1D_X', cell_dim=1, phy_dim=1, parent=tree)
  rename_append(base, trees1d_x)

  # Cleanup (to speed up test)
  PT.rm_nodes_from_label(tree, 'ZoneBC_t')

  # Add few 1D elts (BAR) to 3D elts meshes  (Line Y=0.5 on plane Z=0)
  z = PT.get_node_from_name(tree, '3d_elmts')
  if comm.rank == 0:
    ec = np.array([56,57, 57,58, 58,59, 59,60, 60,61, 61,62, 62,63, 63,64, 64,65, 65,66], z[1].dtype)
    distri = np.array([0, 10, 10], z[1].dtype)
  else:
    ec= np.empty(0, z[1].dtype)
    distri =  10*np.ones(3, z[1].dtype)
  bar_n = PT.new_Elements('BAR_2.0', 'BAR_2', erange=[6201, 6210], econn=ec, parent=z)
  MT.newDistribution({'Element' : distri}, bar_n)

  if parallel == 'part':
    tree = maia.factory.partition_dist_tree(tree, comm)

  # Remove coord Z after split, because split does not like it
  for name in ['Base2D_XY', 'Base1D_XY', 'Base1D_X']:
    base = PT.get_child_from_name(tree, name)
    PT.rm_nodes_from_name(base, 'CoordinateZ')
  # Last base is Base1D_X -> rm Y coord
  PT.rm_nodes_from_name(base, 'CoordinateY')


  geometry.compute_centers(tree, 3, comm) 
  geometry.compute_centers(tree, 2, comm) 

  # For edges, we need to remove poly3d and structured meshes (not implemented)
  mask = PT.shallow_copy(tree)
  PT.rm_nodes_from_name(mask, '3d_poly*')
  PT.rm_nodes_from_name(mask, '3d_str*')
  PT.rm_nodes_from_name(mask, '2d_str*')
  geometry.compute_centers(mask, 1, comm) 
  tree = PT.union(tree, mask)

  for base in PT.get_all_CGNSBase_t(tree):
    cell_dim, phy_dim = PT.get_value(base)
    for zone in PT.get_all_Zone_t(base):
      # Check number of produced arrays
      for sol in PT.get_children_from_name(zone, f'Geometry_*'):
        assert len(PT.get_children_from_label(sol, 'DataArray_t')) == phy_dim
      # All zones have computed CellCenter
      cc_sol = PT.get_child_from_name(zone, f'Geometry_{cell_dim}d')
      assert PT.Subset.GridLocation(cc_sol) == 'CellCenter'
      # Test existance of splitted containers for S3D zone
      if PT.Zone.Type(zone) == 'Structured':
        if cell_dim == 3:
          assert PT.get_child_from_name(zone, f'Geometry_2d') is None
          assert PT.get_child_from_name(zone, f'Geometry_2d_I') is not None
          assert PT.get_child_from_name(zone, f'Geometry_2d_J') is not None
          assert PT.get_child_from_name(zone, f'Geometry_2d_K') is not None
        for sol in PT.get_children_from_name(zone, f'Geometry_*'):
          assert MT.getGlobalNumbering(sol) is None
          dim = int(PT.get_name(sol)[9])
          if cell_dim == dim:
            arrays = [PT.get_child_from_name(sol, f'Center{dir}')[1] for dir in 'XYZ'[:phy_dim]]
            computed = np.zeros(3*arrays[0].size, arrays[0].dtype)
            for i,array in enumerate(arrays):
              computed[i::3] = array.reshape(-1, order='F')
            expected = geometry._compute_zone_centers(zone, dim, comm)
            assert np.allclose(computed, expected)


      # Compare with elementary func, who does not add data in tree (correctness of results
      # should be done at lower level)
      elif PT.Zone.Type(zone) == 'Unstructured':
        for sol in PT.get_children_from_name(zone, f'Geometry_*'):
          dim = int(PT.get_name(sol)[-2])
          arrays = [PT.get_child_from_name(sol, f'Center{dir}')[1] for dir in 'XYZ'[:phy_dim]]
          computed = np.zeros(3*arrays[0].size, arrays[0].dtype)
          for i,array in enumerate(arrays):
            computed[i::3] = array
          expected = geometry._compute_zone_centers(zone, dim, comm)
          assert np.allclose(computed, expected)

          if PT.Subset.GridLocation(sol) in ['FaceCenter', 'EdgeCenter']:
            pl = PT.get_child_from_name(sol, 'PointList')[1][0]
            elt_d_range = PT.Zone.get_elt_range_per_dim(zone)[dim]
            if parallel == 'part':
              assert np.array_equal(pl, np.arange(elt_d_range[0], elt_d_range[1]+1))
            else:
              distri = MT.getDistribution(sol, 'Index')[1]
              # Works but probably because only one section per dim, otherwise PL may mix elements
              assert np.array_equal(pl, np.arange(elt_d_range[0], elt_d_range[1]+1)[distri[0]:distri[1]])


@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("parallel", ["part", "dist"])
def test_compute_measures(parallel, comm):

  def rename_append(parent, child_dict):
    for zname, t in child_dict.items():
      zone = PT.get_node_from_label(t, 'Zone_t')
      PT.set_name(zone, zname)
      PT.add_child(parent, zone)

  # Create a fake tree containing all zone / mesh kind
  tree = PT.new_CGNSTree() 

  trees3d = {'3d_poly'   : maia.factory.generate_dist_block(11, 'Poly', comm),
             '3d_elmts'  : maia.factory.generate_dist_block(11, 'PENTA_6', comm),
             '3d_struct' : maia.factory.generate_dist_block([5,3,6], 'S', comm)}
  trees2d = {'2d_poly'   : maia.factory.generate_dist_block(11, 'QUAD_4', comm),
             '2d_elmts'  : maia.factory.generate_dist_block(11, 'TRI_3', comm),
             '2d_struct' : maia.factory.generate_dist_block([11,11,1], 'S', comm)}
  trees1d = {'1d_elmts'  : generate_dist_line(comm)}

  maia.algo.dist.convert_elements_to_ngon(trees2d['2d_poly'], comm)

  trees2d_xy = {f"{key}_xy" : PT.deep_copy(tree) for key, tree in trees2d.items()}
  trees1d_xy = {f"{key}_xy" : PT.deep_copy(tree) for key, tree in trees1d.items()}
  trees1d_x  = {f"{key}_x"  : PT.deep_copy(tree) for key, tree in trees1d.items()}

  base = PT.new_CGNSBase('Base3D', cell_dim=3, phy_dim=3, parent=tree)
  rename_append(base, trees3d)
  base = PT.new_CGNSBase('Base2D', cell_dim=2, phy_dim=3, parent=tree)
  rename_append(base, trees2d)
  base = PT.new_CGNSBase('Base2D_XY', cell_dim=2, phy_dim=2, parent=tree)
  rename_append(base, trees2d_xy)
  base = PT.new_CGNSBase('Base1D', cell_dim=1, phy_dim=3, parent=tree)
  rename_append(base, trees1d)
  base = PT.new_CGNSBase('Base1D_XY', cell_dim=1, phy_dim=2, parent=tree)
  rename_append(base, trees1d_xy)
  base = PT.new_CGNSBase('Base1D_X', cell_dim=1, phy_dim=1, parent=tree)
  rename_append(base, trees1d_x)

  # Cleanup (to speed up test)
  PT.rm_nodes_from_label(tree, 'ZoneBC_t')

  # Add few 1D elts (BAR) to 3D elts meshes  (Line Y=0.5 on plane Z=0)
  z = PT.get_node_from_name(tree, '3d_elmts')
  last_range = PT.Element.Range(PT.Zone.get_ordered_elements(z)[-1])[1]
  if comm.rank == 0:
    ec = np.array([56,57, 57,58, 58,59, 59,60, 60,61, 61,62, 62,63, 63,64, 64,65, 65,66], z[1].dtype)
    distri = np.array([0, 10, 10], z[1].dtype)
  else:
    ec= np.empty(0, z[1].dtype)
    distri =  10*np.ones(3, z[1].dtype)
  bar_n = PT.new_Elements('BAR_2.0', 'BAR_2', erange=[last_range+1, last_range+10], econn=ec, parent=z)
  MT.newDistribution({'Element' : distri}, bar_n)

  if parallel == 'part':
    tree = maia.factory.partition_dist_tree(tree, comm, preserve_orientation=True)

  # Remove coord Z after split, because split does not like it
  for name in ['Base2D_XY', 'Base1D_XY', 'Base1D_X']:
    base = PT.get_child_from_name(tree, name)
    PT.rm_nodes_from_name(base, 'CoordinateZ')
  # Last base is Base1D_X -> rm Y coord
  PT.rm_nodes_from_name(base, 'CoordinateY')

  geometry.compute_measures(tree, 3, comm)
  geometry.compute_measures(tree, 2, comm)

  # For edges, we need to remove poly3d and structured meshes (not implemented)
  mask = PT.shallow_copy(tree)
  PT.rm_nodes_from_name(mask, '3d_poly*')
  PT.rm_nodes_from_name(mask, '3d_str*')
  PT.rm_nodes_from_name(mask, '2d_str*')
  geometry.compute_measures(mask, 1, comm) 
  tree = PT.union(tree, mask)

  for base in PT.get_all_CGNSBase_t(tree):
    cell_dim, phy_dim = PT.get_value(base)
    for zone in PT.get_all_Zone_t(base):
      # Check number of produced arrays
      for sol in PT.get_children_from_name(zone, f'Geometry_*'):
        assert len(PT.get_children_from_label(sol, 'DataArray_t')) == 1
      # All zones have computed CellCenter
      cc_sol = PT.get_child_from_name(zone, f'Geometry_{cell_dim}d')
      assert PT.Subset.GridLocation(cc_sol) == 'CellCenter'
      # Test existance of splitted containers for S3D zone
      if PT.Zone.Type(zone) == 'Structured':
        if cell_dim == 3:
          assert PT.get_child_from_name(zone, f'Geometry_2d') is None
          assert PT.get_child_from_name(zone, f'Geometry_2d_I') is not None
          assert PT.get_child_from_name(zone, f'Geometry_2d_J') is not None
          assert PT.get_child_from_name(zone, f'Geometry_2d_K') is not None
        for sol in PT.get_children_from_name(zone, f'Geometry_*'):
          assert MT.getGlobalNumbering(sol) is None
          dim = int(PT.get_name(sol)[9])
          if cell_dim == dim:
            computed = PT.get_child_from_name(sol, f'Measure')[1].reshape(-1, order='F')
            expected = geometry._compute_zone_measures(zone, dim, comm)
            assert np.allclose(computed, expected)


      # Compare with elementary func, who does not add data in tree (correctness of results
      # should be done at lower level)
      elif PT.Zone.Type(zone) == 'Unstructured':
        for sol in PT.get_children_from_name(zone, f'Geometry_*'):
          dim = int(PT.get_name(sol)[-2])
          computed = PT.get_child_from_name(sol, f'Measure')[1]
          expected = geometry._compute_zone_measures(zone, dim, comm)
          assert np.allclose(computed, expected)

          if PT.Subset.GridLocation(sol) in ['FaceCenter', 'EdgeCenter']:
            pl = PT.get_child_from_name(sol, 'PointList')[1][0]
            elt_d_range = PT.Zone.get_elt_range_per_dim(zone)[dim]
            if parallel == 'part':
              assert np.array_equal(pl, np.arange(elt_d_range[0], elt_d_range[1]+1))
            else:
              starts, ends = [], []
              for elt in PT.Zone.get_ordered_elements_per_dim(zone)[dim]:
                starts.append(MT.getDistribution(elt, 'Element')[1][0] + PT.Element.Range(elt)[0])
                ends  .append(MT.getDistribution(elt, 'Element')[1][1] + PT.Element.Range(elt)[0])
              assert np.array_equal(pl, np_utils.multi_arange(starts, ends))
