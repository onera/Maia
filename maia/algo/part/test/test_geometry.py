import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import maia.pytree as PT

from maia.factory.dcube_generator import dcube_generate

from maia.algo.part import geometry

@mark_mpi_test(1)
def test_compute_cell_center(sub_comm):
  #Test U
  tree = dcube_generate(3, 1., [0,0,0], sub_comm)
  zoneU = PT.get_all_Zone_t(tree)[0]
  #On partitions, element are supposed to be I4
  for elt_node in PT.iter_children_from_label(zoneU, 'Elements_t'):
    for name in ['ElementConnectivity', 'ParentElements', 'ElementStartOffset']:
      node = PT.get_child_from_name(elt_node, name)
      node[1] = node[1].astype(np.int32)

  cell_center = geometry.compute_cell_center(zoneU)
  expected_cell_center = np.array([0.25, 0.25, 0.25, 
                                   0.75, 0.25, 0.25, 
                                   0.25, 0.75, 0.25, 
                                   0.75, 0.75, 0.25, 
                                   0.25, 0.25, 0.75, 
                                   0.75, 0.25, 0.75, 
                                   0.25, 0.75, 0.75, 
                                   0.75, 0.75, 0.75])
  assert (cell_center == expected_cell_center).all()

  #Test S
  cx_s = PT.get_node_from_name(zoneU, 'CoordinateX')[1].reshape((3,3,3), order='F')
  cy_s = PT.get_node_from_name(zoneU, 'CoordinateY')[1].reshape((3,3,3), order='F')
  cz_s = PT.get_node_from_name(zoneU, 'CoordinateZ')[1].reshape((3,3,3), order='F')

  zoneS = PT.new_Zone(size=[[3,2,0], [3,2,0], [3,2,0]], type='Structured')
  grid_coords = PT.new_GridCoordinates(parent=zoneS)
  PT.new_DataArray('CoordinateX', cx_s, parent=grid_coords)
  PT.new_DataArray('CoordinateY', cy_s, parent=grid_coords)
  PT.new_DataArray('CoordinateZ', cz_s, parent=grid_coords)
  cell_center = geometry.compute_cell_center(zoneS)
  assert (cell_center == expected_cell_center).all()

  #Test wrong case
  PT.rm_children_from_label(zoneU, 'Elements_t')
  with pytest.raises(NotImplementedError):
    geometry.compute_cell_center(zoneU)
