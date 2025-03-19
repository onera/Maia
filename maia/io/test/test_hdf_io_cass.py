import pytest
import numpy as np
import os
import pytest_parallel
import maia.utils.test_utils as TU
#import Converter 
#from maia.io import _hdf_io_cass as LC 
import maia
from maia.io.cgns_io_tree import  create_tree_hdf_filter 
from maia.io.cgns_io_tree import add_distribution_info

know_cassiopee = True
try:
  import Converter
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
  tmp_dir = TU.create_collective_tmp_dir(comm)
  out_file = os.path.join(tmp_dir, 'yt.cgns')
  dist_tree=create_tree_with_perio_jn(comm)
  maia.io.dist_tree_to_file(dist_tree, out_file, comm)
  dist_tree= LC.load_size_tree(out_file, comm)
  LC.load_grid_connectivity_property(out_file, dist_tree)



@pytest.mark.skipif(not know_cassiopee, reason="Require Cassiopee")
@pytest_parallel.mark.parallel(2)  
def test_load_partial(comm):
  filename = str(TU.sample_mesh_dir / 'only_coords.hdf')
  dist_tree = LC.load_size_tree(filename, comm)
  add_distribution_info(dist_tree, comm)
  hdf_filter = create_tree_hdf_filter(dist_tree) 
  hdf_filter = {key:val for key,val in hdf_filter.items() if not key.endswith('#Size')}
  assert hdf_filter is not None
  LC.load_partial(filename, dist_tree, hdf_filter,comm)

@pytest.mark.skipif(not know_cassiopee, reason="Require Cassiopee")
def test_read_full():
  filename = str(TU.sample_mesh_dir / 'only_coords.hdf')
  LC.read_full(filename)
  
@pytest.mark.skipif(not know_cassiopee, reason="Require Cassiopee")
@pytest_parallel.mark.parallel(1)
def test_write_full(comm):
  filename = str(TU.sample_mesh_dir / 'only_coords.hdf')
  dist_tree = LC.load_size_tree(filename, comm)
  links = [['.', 'this/hdf/file.hdf', 'this/node', 'Base/ZoneA/GridCoordinates/CoordinateX'],
           ['.', 'this/hdf/file.hdf', 'this/other_node', 'Base/ZoneB/GridCoordinates/CoordinateY']] 
  filename_to_write = "write_tree.hdf"
  LC.write_full(filename_to_write, dist_tree, links)
  
 # TEST NOT CORRECT  
@pytest.mark.skipif(not know_cassiopee, reason="Require Cassiopee")
@pytest_parallel.mark.parallel(1)
def test_write_partial(comm):
  filename = str(TU.sample_mesh_dir / 'only_coords.hdf')
  dist_tree = LC.load_size_tree(filename, comm)
  filename_to_write = "write_tree.hdf"
  links = [['.', 'this/hdf/file.hdf', 'this/node', 'Base/ZoneA/GridCoordinates/CoordinateX'],
           ['.', 'this/hdf/file.hdf', 'this/other_node', 'Base/ZoneB/GridCoordinates/CoordinateY']] 
  add_distribution_info(dist_tree, comm)
  hdf_filter = create_tree_hdf_filter(dist_tree) 
  hdf_filter = {f'/{key}' : data for key, data in hdf_filter.items()} 
  #LC.write_partial(filename_to_write, dist_tree, hdf_filter, links, comm)
  


  