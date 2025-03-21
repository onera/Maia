import os
import pytest
import pytest_parallel

import maia
import maia.utils.test_utils as TU

from maia.io.cgns_io_tree import create_tree_hdf_filter, add_distribution_info

know_cassiopee = True
try:
  from maia.io import _hdf_io_cass as LC
except ImportError:
  know_cassiopee = False

import maia.pytree as PT

@pytest.mark.skipif(not know_cassiopee, reason="Require Cassiopee")
def test_add_sizes_to_zone_tree():
  yt = """
Zone Zone_t:
  Hexa Elements_t [17, 0]:
    ElementConnectivity DataArray_t None:
  ZBC ZoneBC_t:
    bc BC_t "farfield":
      PointList IndexArray_t None:
    bc_withds BC_t "farfield":
      PointList IndexArray_t None:
      BCDataSet BCDataSet_t:
        PointList IndexArray_t None:
  ZGC ZoneGridConnectivity_t:
    gc GridConnectivity_t:
      PointList IndexArray_t None:
      PointListDonor IndexArray_t None:
  ZSR ZoneSubRegion_t:
    PointList IndexArray_t None:
  FS FlowSolution_t:
  FSPL FlowSolution_t:
    PointList IndexArray_t None:
"""
  zone = PT.yaml.to_node(yt)
  size_data = {'/Zone/Hexa/ElementConnectivity' : (1, 'I4', 160),
               '/Zone/ZBC/bc/PointList' : (1, 'I4', (1,30)),
               '/Zone/ZBC/bc_withds/PointList' : (1, 'I4', (1,100)),
               '/Zone/ZBC/bc_withds/BCDataSet/PointList' : (1, 'I4', (1,10)),
               '/Zone/ZGC/gc/PointList' : (1, 'I4', (1,20)),
               '/Zone/ZGC/gc/PointListDonor' : (1, 'I4', (1,20)),
               '/Zone/ZSR/PointList' : (1, 'I4', (1,34)),
               '/Zone/FSPL/PointList' : (1, 'I4', (1,10)),
              }

  LC.add_sizes_to_zone_tree(zone, '/Zone', size_data)

  assert PT.get_node_from_path(zone, 'Hexa/ElementConnectivity#Size') is None

  assert (PT.get_node_from_path(zone, 'ZBC/bc/PointList#Size')[1] == [1,30]).all()
  assert (PT.get_node_from_path(zone, 'ZBC/bc_withds/PointList#Size')[1] == [1,100]).all()
  assert (PT.get_node_from_path(zone, 'ZBC/bc_withds/BCDataSet/PointList#Size')[1] == [1,10]).all()

  assert (PT.get_node_from_path(zone, 'ZGC/gc/PointList#Size')[1] == [1,20]).all()

  assert (PT.get_node_from_path(zone, 'ZSR/PointList#Size')[1] == [1,34]).all()

  assert (PT.get_node_from_path(zone, 'FSPL/PointList#Size')[1] == [1,10]).all()
  assert (PT.get_node_from_path(zone, 'FS/PointList#Size') is None)

@pytest.mark.skipif(not know_cassiopee, reason="Require Cassiopee")
def test_add_sizes_to_tree():
  yt = """
BaseA CGNSBase_t:
  Zone Zone_t:
    Hexa Elements_t [17, 0]:
      ElementConnectivity DataArray_t None:
    ZBC ZoneBC_t:
      bc BC_t "farfield":
        PointList IndexArray_t None:
      bc_withds BC_t "farfield":
        PointList IndexArray_t None:
        BCDataSet BCDataSet_t:
          PointList IndexArray_t None:
    ZGC ZoneGridConnectivity_t:
      gc GridConnectivity_t:
        PointList IndexArray_t None:
        PointListDonor IndexArray_t None:
    ZSR ZoneSubRegion_t:
      PointList IndexArray_t None:
"""
  tree = PT.yaml.to_cgns_tree(yt)
  size_data_tree = {'/BaseA/Zone/Hexa/ElementConnectivity' : (1, 'I4', 160),
                    '/BaseA/Zone/ZBC/bc/PointList' : (1, 'I4', (1,30)),
                    '/BaseA/Zone/ZBC/bc_withds/PointList' : (1, 'I4', (1,100)),
                    '/BaseA/Zone/ZBC/bc_withds/BCDataSet/PointList' : (1, 'I4', (1,10)),
                    '/BaseA/Zone/ZGC/gc/PointList' : (1, 'I4', (1,20)),
                    '/BaseA/Zone/ZGC/gc/PointListDonor' : (1, 'I4', (1,20)),
                    '/BaseA/Zone/ZSR/PointList' : (1, 'I4', (1,34)),
                   }
  LC.add_sizes_to_tree(tree, size_data_tree)
  assert len(PT.get_nodes_from_name(tree, '*#Size')) == 6
  

@pytest.mark.skipif(not know_cassiopee, reason="Require Cassiopee")
@pytest_parallel.mark.parallel(2)
def test_load_size_tree(comm):
  filename = str(TU.sample_mesh_dir / 'only_coords.hdf')
  dist_tree = LC.load_size_tree(filename, comm)
  if comm.Get_rank() == 0:
    iso_zone = PT.get_all_Zone_t(dist_tree)[0]
    assert PT.Zone.Type(iso_zone)=="Unstructured"
    assert PT.Zone.n_cell(iso_zone)==0 and PT.Zone.n_vtx(iso_zone)==6
  else:
    iso_zone = PT.get_all_Zone_t(dist_tree)[1]
    assert PT.Zone.Type(iso_zone)=="Structured"
    assert PT.Zone.n_cell(iso_zone) == 1 and PT.Zone.n_vtx(iso_zone) == 4
 
def create_tree_with_perio_jn(comm):
    tree = maia.factory.generate_dist_block(4, 'Poly', comm)
    zone = PT.get_node_from_label(tree, 'Zone_t')

    xmax = PT.pop_node_from_path(zone, 'ZoneBC/Xmax')
    xmin = PT.pop_node_from_path(zone, 'ZoneBC/Xmin')

    PT.update_node(xmax, label='GridConnectivity_t', value='zone')
    PT.update_node(xmin, label='GridConnectivity_t', value='zone')
    PT.new_IndexArray('PointListDonor', PT.get_child_from_name(xmax, 'PointList')[1].copy(), parent=xmin)
    PT.new_IndexArray('PointListDonor', PT.get_child_from_name(xmin, 'PointList')[1].copy(), parent=xmax)
    PT.new_GridConnectivityProperty(periodic={'translation':[1.,0,0]}, parent=xmin)
    PT.new_GridConnectivityProperty(periodic={'translation':[-1.,0,0]}, parent=xmax)

    PT.new_child(zone, 'ZoneGridConnectivity', 'ZoneGridConnectivity_t', children=[xmin, xmax])

    return tree


@pytest.mark.skipif(not know_cassiopee, reason="Require Cassiopee")
@pytest_parallel.mark.parallel(1)
def test_load_grid_connectivity_property(comm):
  dist_tree=create_tree_with_perio_jn(comm)
  tmp_dir = TU.create_collective_tmp_dir(comm)
  out_file = os.path.join(tmp_dir, 'yt.cgns')
  maia.io.dist_tree_to_file(dist_tree, out_file, comm)

  size_tree = LC.load_size_tree(out_file, comm)
  LC.load_grid_connectivity_property(out_file, size_tree)
  is_gc = lambda n : PT.get_label(n) in ['GridConnectivity_t']
  for gc in PT.get_children_from_predicates(size_tree, ['ZoneGridConnectivity_t', is_gc]):
      assert (PT.get_node_from_name(gc, 'RotationCenter') == [0., 0., 0.]).all()
      assert (PT.get_node_from_name(gc, 'RotationAngle') == [0., 0., 0.]).all()
      assert (PT.get_node_from_name(gc, 'Translation') == [-1., 0., 0.]).all()
  TU.rm_collective_dir(tmp_dir, comm)

@pytest.mark.skipif(not know_cassiopee, reason="Require Cassiopee")
@pytest_parallel.mark.parallel(2)  
def test_load_partial(comm):
  filename = str(TU.sample_mesh_dir / 'only_coords.hdf')
  size_tree = LC.load_size_tree(filename, comm)
  add_distribution_info(size_tree, comm)
  hdf_filter = create_tree_hdf_filter(size_tree) 
  hdf_filter = {key:val for key,val in hdf_filter.items() if not key.endswith('#Size')}
  
  LC.load_partial(filename, size_tree, hdf_filter,comm)
  if comm.rank == 0:
    assert (PT.get_node_from_path(size_tree, 'Base/ZoneU/GridCoordinates/CoordinateX')[1] == [1., 2, 3]).all()
    assert (PT.get_node_from_path(size_tree, 'Base/ZoneU/GridCoordinates/CoordinateY')[1] == [-1., -2, -3]).all()
    assert (PT.get_node_from_path(size_tree, 'Base/ZoneS/GridCoordinates/CoordinateX')[1] == [1., 3]).all()
    assert (PT.get_node_from_path(size_tree, 'Base/ZoneS/GridCoordinates/CoordinateY')[1] == [-1., 0]).all()
  elif comm.rank == 1:
    assert (PT.get_node_from_path(size_tree, 'Base/ZoneU/GridCoordinates/CoordinateX')[1] == [4., 5, 6]).all()
    assert (PT.get_node_from_path(size_tree, 'Base/ZoneU/GridCoordinates/CoordinateY')[1] == [-4., -5, -6]).all()
    assert (PT.get_node_from_path(size_tree, 'Base/ZoneS/GridCoordinates/CoordinateX')[1] == [2., 4]).all()
    assert (PT.get_node_from_path(size_tree, 'Base/ZoneS/GridCoordinates/CoordinateY')[1] == [-1., 0]).all()

@pytest.mark.skipif(not know_cassiopee, reason="Require Cassiopee")
def test_read_full():
  filename = str(TU.sample_mesh_dir / 'only_coords.hdf')
  tree = LC.read_full(filename)

  expected = PT.yaml.to_cgns_tree(f"""
  Base CGNSBase_t I4 [2, 2]:
    ZoneU Zone_t I4 [[6, 0, 0]]:
      ZoneType ZoneType_t 'Unstructured':
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t R8 [1,2,3,4,5,6]:
        CoordinateY DataArray_t R8 [-1,-2,-3,-4,-5,-6]:
    ZoneS Zone_t I4 [[2, 1, 0], [2, 1, 0]]:
      ZoneType ZoneType_t 'Structured':
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t R8 [[1,2],[3,4]]:
        CoordinateY DataArray_t R8 [[-1,-1],[0,0]]:
  """)
   # Converter casts CGNSLibraryVersion
  PT.rm_children_from_label(expected, 'CGNSLibraryVersion_t')
  PT.rm_children_from_label(tree, 'CGNSLibraryVersion_t')
  assert PT.is_same_tree(tree, expected)
  
@pytest.mark.skipif(not know_cassiopee, reason="Require Cassiopee")
@pytest_parallel.mark.parallel(1)
def test_write_full(tmp_path, comm):
  tree = maia.factory.generate_dist_block(4, 'TRI_3', comm)
  links = [['.', 'this/hdf/file.hdf', 'this/node', 'Base/zone/GridCoordinates/CoordinateX'],
           ['.', 'this/hdf/file.hdf', 'this/other_node', 'Base/zone/GridCoordinates/CoordinateY']] 
  filename = tmp_path / 'out.cgns'
  LC.write_full(str(filename), tree, links)

  assert filename.exists()
