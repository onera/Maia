import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia              import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.factory      import dcube_generator as DCG

from maia.algo.part import point_cloud_utils as PCU

def as_partitioned(zone):
  #On partitions, element are supposed to be I4
  for elt_node in PT.iter_children_from_label(zone, 'Elements_t'):
    for name in ['ElementConnectivity', 'ParentElements', 'ElementStartOffset']:
      node = PT.get_child_from_name(elt_node, name)
      node[1] = node[1].astype(np.int32)
  I._rmNodesByName(zone, ':CGNS#Distribution')

@mark_mpi_test(1)
def test_get_zone_ln_to_gn_from_loc(sub_comm):
  tree = DCG.dcube_generate(3, 1., [0.,0.,0.], sub_comm)
  zone = I.getZones(tree)[0]
  as_partitioned(zone)
  vtx_gnum = np.arange(3**3) + 1
  cell_gnum = np.arange(2**3) + 1
  MT.newGlobalNumbering({'Vertex' : vtx_gnum, 'Cell' : cell_gnum}, parent=zone)

  assert (PCU._get_zone_ln_to_gn_from_loc(zone, 'Vertex') == vtx_gnum).all()
  assert (PCU._get_zone_ln_to_gn_from_loc(zone, 'CellCenter') == cell_gnum).all()

@mark_mpi_test(1)
def test_get_point_cloud(sub_comm):
  tree = DCG.dcube_generate(3, 1., [0.,0.,0.], sub_comm)
  zone = I.getZones(tree)[0]
  as_partitioned(zone)
  I._rmNodesByType(zone, 'ZoneBC_t')
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

def test_extract_sub_cloud():
  coords = np.array([0,0,0, .5,0,0, 1,0,0, 0,1,0, .5,1,0, 1,1,0])
  lngn   = np.array([42,9,1,4,55,3], pdm_gnum_dtype)

  sub_coords, sub_lngn = PCU.extract_sub_cloud(coords, lngn, np.empty(0, np.int32))
  assert sub_coords.size == sub_lngn.size == 0
  sub_coords, sub_lngn = PCU.extract_sub_cloud(coords, lngn, np.array([0,1,2,3,4,5], np.int32))
  assert (sub_coords == coords).all()
  assert (sub_lngn == lngn).all()
  sub_coords, sub_lngn = PCU.extract_sub_cloud(coords, lngn, np.array([1,3], np.int32))
  assert (sub_coords == [.5,0,0, 0,1,0]).all()
  assert (sub_lngn == [9,4]).all()
  sub_coords, sub_lngn = PCU.extract_sub_cloud(coords, lngn, np.array([3,1], np.int32))
  assert (sub_coords == [0,1,0, .5,0,0]).all()
  assert (sub_lngn == [4,9]).all()
  
@mark_mpi_test(2)
def test_create_sub_numbering(sub_comm):
  if sub_comm.Get_rank() == 0:
    lngn_l =  [ np.array([3,9], pdm_gnum_dtype), np.array([], pdm_gnum_dtype) ]
    expected_sub_lngn_l  =  [ np.array([1,5]), np.array([]) ]
  elif sub_comm.Get_rank() == 1:
    lngn_l =  [ np.array([7,5,6], pdm_gnum_dtype) ]
    expected_sub_lngn_l  =  [ np.array([4,2,3]) ]

  sub_gnum_l = PCU.create_sub_numbering(lngn_l, sub_comm)

  for sub_gnum, expected_sub_lngn in zip(sub_gnum_l, expected_sub_lngn_l):
    assert (sub_gnum == expected_sub_lngn).all()

