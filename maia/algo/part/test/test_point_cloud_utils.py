import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import Converter.Internal as I
import maia.pytree.maia   as MT

from maia.factory      import dcube_generator as DCG

from maia.algo.part import point_cloud_utils as PCU

@mark_mpi_test(1)
def test_get_point_cloud(sub_comm):
  tree = DCG.dcube_generate(3, 1., [0.,0.,0.], sub_comm)
  zone = I.getZones(tree)[0]
  #On partitions, element are supposed to be I4
  for elt_node in I.getNodesFromType1(zone, 'Elements_t'):
    for name in ['ElementConnectivity', 'ParentElements', 'ElementStartOffset']:
      node = I.getNodeFromName1(elt_node, name)
      node[1] = node[1].astype(np.int32)
  I._rmNodesByType(zone, 'ZoneBC_t')
  I._rmNodesByName(zone, ':CGNS#Distribution')
  fs = I.newFlowSolution('MyOwnCoords', 'CellCenter', parent=zone)
  I.newDataArray('CoordinateX', 1*np.ones(8), parent=fs)
  I.newDataArray('CoordinateY', 2*np.ones(8), parent=fs)
  I.newDataArray('CoordinateZ', 3*np.ones(8), parent=fs)
  vtx_gnum = np.arange(3**3) + 1
  cell_gnum = np.arange(2**3) + 1
  MT.newGlobalNumbering({'Vertex' : vtx_gnum, 'Cell' : cell_gnum}, parent=zone)

  expected_vtx_co = np.array([0., 0. , 0. , 0.5, 0. , 0. , 1., 0. , 0. , 0., 0.5, 0. , 0.5, 0.5, 0. , 1., 0.5, 0.,
                              0., 1. , 0. , 0.5, 1. , 0. , 1., 1. , 0. , 0., 0. , 0.5, 0.5, 0. , 0.5, 1., 0. , 0.5,
                              0., 0.5, 0.5, 0.5, 0.5, 0.5, 1., 0.5, 0.5, 0., 1. , 0.5, 0.5, 1. , 0.5, 1., 1. , 0.5,
                              0., 0. , 1. , 0.5, 0. , 1. , 1., 0. , 1. , 0., 0.5, 1. , 0.5, 0.5, 1. , 1., 0.5, 1.,
                              0., 1. , 1. , 0.5, 1. , 1. , 1., 1. , 1. ])
  expected_cell_co = np.array(
    [0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.75, 0.25, 0.75, 0.75, 0.25, 0.25, 0.25,
     0.75, 0.75, 0.25, 0.75, 0.25, 0.75, 0.75, 0.75, 0.75, 0.75])

  
  coords, gnum = PCU.get_point_cloud(zone, 'Vertex')
  assert (gnum == vtx_gnum).all()
  assert (coords == expected_vtx_co).all()

  coords, gnum = PCU.get_point_cloud(zone, 'CellCenter')
  assert (gnum == cell_gnum).all()
  assert (coords == expected_cell_co).all()

  coords, gnum = PCU.get_point_cloud(zone, 'MyOwnCoords')
  assert (gnum == cell_gnum).all()
  assert (coords == np.tile([1,2,3], 8)).all() #Repeat motif

  with pytest.raises(RuntimeError):
    coords, gnum = PCU.get_point_cloud(zone, 'FaceCenter')

