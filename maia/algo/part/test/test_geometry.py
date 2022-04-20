import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import Converter.Internal     as I

from maia.factory.dcube_generator import dcube_generate

from maia.algo.part import geometry

@mark_mpi_test(1)
def test_compute_cell_center(sub_comm):
  #Test U
  tree = dcube_generate(3, 1., [0,0,0], sub_comm)
  zoneU = I.getZones(tree)[0]
  #On partitions, element are supposed to be I4
  for elt_node in I.getNodesFromType1(zoneU, 'Elements_t'):
    for name in ['ElementConnectivity', 'ParentElements', 'ElementStartOffset']:
      node = I.getNodeFromName1(elt_node, name)
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
  cx_s = I.getNodeFromName(zoneU, 'CoordinateX')[1].reshape((3,3,3), order='F')
  cy_s = I.getNodeFromName(zoneU, 'CoordinateY')[1].reshape((3,3,3), order='F')
  cz_s = I.getNodeFromName(zoneU, 'CoordinateZ')[1].reshape((3,3,3), order='F')

  zoneS = I.newZone(zsize=[[3,2,0], [3,2,0], [3,2,0]], ztype='Structured')
  grid_coords = I.newGridCoordinates(parent=zoneS)
  I.newDataArray('CoordinateX', cx_s, parent=grid_coords)
  I.newDataArray('CoordinateY', cy_s, parent=grid_coords)
  I.newDataArray('CoordinateZ', cz_s, parent=grid_coords)
  cell_center = geometry.compute_cell_center(zoneS)
  assert (cell_center == expected_cell_center).all()

  #Test wrong case
  zone_no_ng = I.rmNodesByType(zoneU, 'Elements_t')
  with pytest.raises(NotImplementedError):
    geometry.compute_cell_center(zone_no_ng)
