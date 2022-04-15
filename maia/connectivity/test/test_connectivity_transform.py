import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import Converter.Internal as I
from maia.generate     import dcube_generator as DCG
from maia.utils        import parse_yaml_cgns
from maia.sids         import pytree as PT

from maia.connectivity import connectivity_transform as CNT

@pytest.mark.parametrize("dtype", ["float"])
def test_pe_cgns_to_pdm_face_cell_bad_type(dtype):
  """
  """
  ngon_pe = np.array([[1, 0],
                       [1, 2],
                       [2, 3]], dtype=dtype, order='F')
  size = np.prod(ngon_pe.shape)
  dface_cell = np.empty(2*ngon_pe.shape[0], order='F', dtype=dtype)
  try:
    CNT.pe_cgns_to_pdm_face_cell(ngon_pe, dface_cell)
    assert False
  except TypeError:
    assert True

@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_pe_cgns_to_pdm_face_cell(dtype):
  """
  """
  ngon_pe = np.array([[1, 0],
                      [1, 2],
                      [2, 3]], dtype=dtype, order='F')
  size = np.prod(ngon_pe.shape)
  dface_cell = np.empty(2*ngon_pe.shape[0], order='F', dtype=dtype)
  CNT.pe_cgns_to_pdm_face_cell(ngon_pe, dface_cell)

  np.testing.assert_equal(dface_cell, np.array([1, 0, 1, 2, 2, 3], order='F', dtype=dtype))

@pytest.mark.parametrize("dtype", ["float"])
def test_compute_idx_local_bad_type(dtype):
  """
  """
  connect_g_idx = np.array([100, 104, 108, 112], dtype=dtype  , order='F')
  distrib       = np.array([100, 116, 100000  ], dtype=dtype  , order='F')
  connect_l_idx = np.empty((connect_g_idx.shape[0]+1), dtype='int32', order='F')
  try:
    CNT.compute_idx_local(connect_l_idx, connect_g_idx, distrib)
    assert False
  except TypeError:
    assert True

@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_compute_idx_local(dtype):
  """
  """
  connect_g_idx = np.array([100, 104, 108, 112], dtype=dtype  , order='F')
  distrib       = np.array([100, 116, 100000  ], dtype=dtype  , order='F')
  connect_l_idx = np.empty((connect_g_idx.shape[0]+1), dtype='int32', order='F')
  CNT.compute_idx_local(connect_l_idx, connect_g_idx, distrib)
  #print(connect_l_idx)
  np.testing.assert_equal(connect_l_idx, np.array([0, 4, 8, 12, 16], dtype=dtype, order='F'))

def test_get_ngon_pe_local():
  yt = """
  NGonElements Elements_t [22, 0]:
    ElementRange IndexRange_t [1, 8]:
    ParentElements DataArray_t [[9, 0], [10, 0], [11, 12], [0, 12]]:
  """
  ngon = parse_yaml_cgns.to_node(yt)
  assert (CNT.get_ngon_pe_local(ngon) == np.array([[9-8,0], [10-8,0], [11-8,12-8], [0,12-8]])).all()

  yt = """
  NGonElements Elements_t [22, 0]:
    ElementRange IndexRange_t [1, 8]:
    ParentElements DataArray_t [[1, 0], [2, 0], [3, 4], [0, 4]]:
  """
  ngon = parse_yaml_cgns.to_node(yt)
  assert (CNT.get_ngon_pe_local(ngon) == np.array([[1,0], [2,0], [3,4], [0,4]])).all()

  yt = """
  NGonElements Elements_t [22, 0]:
    ElementRange IndexRange_t [1, 8]:
  """
  ngon = parse_yaml_cgns.to_node(yt)
  with pytest.raises(RuntimeError):
    CNT.get_ngon_pe_local(ngon)

@mark_mpi_test(1)
def test_enforce_boundary_pe_left(sub_comm):
  tree = DCG.dcube_generate(3, 1., [0., 0., 0.], sub_comm)
  zone = I.getZones(tree)[0]
  #### Add nface (todo : use maia func when available; reput in eso form after)
  #Careful, fix ngon do crazy stuff if pe start at 1
  pe_node = I.getNodeFromPath(zone, 'NGonElements/ParentElements')
  pe_node[1] = CNT.get_ngon_pe_local(I.getNodeFromName(zone, 'NGonElements'))
  I._fixNGon(zone)
  for node_name, lsize in zip(['NFaceElements', 'NGonElements'], [6,4]):
    node = I.getNodeFromName1(zone, node_name)
    ec = I.getNodeFromName1(node, 'ElementConnectivity')
    er = I.getNodeFromName1(node, 'ElementRange')
    n_elem = er[1][1] - er[1][0] + 1
    eso = np.arange(0, (n_elem+1)*lsize, lsize, np.int32)
    ec[1] = np.delete(ec[1], eso[:-1] + np.arange(n_elem))
    I.newDataArray('ElementStartOffset', eso, node)
  pe_node[1] += n_elem*(pe_node[1] > 0)
  #### Todo : remove preceding stuff when we have pe -> nface conversion
  pe_bck = pe_node[1].copy()
  zone_bck = I.copyTree(zone)
  CNT.enforce_boundary_pe_left(zone)
  assert PT.is_same_tree(zone, zone_bck)

  #Test with swapped pe
  pe_node[1][2] = pe_node[1][2][::-1]
  CNT.enforce_boundary_pe_left(zone)

  assert I.getNodeFromPath(zone, 'NFaceElements/ElementConnectivity')[1][12] == -3
  expt_ng_ec = I.getNodeFromPath(zone_bck, 'NGonElements/ElementConnectivity')[1].copy()
  expt_ng_ec[4*2 : 4*3] = [5, 4, 7, 8]
  assert (I.getNodeFromPath(zone, 'NGonElements/ElementConnectivity')[1] == expt_ng_ec).all()

  #Test with no NFace
  zone = I.copyTree(zone_bck)
  pe_node = I.getNodeFromPath(zone, 'NGonElements/ParentElements')
  pe_node[1][2] = pe_node[1][2][::-1]
  I._rmNodesByName(zone, 'NFaceElements')
  CNT.enforce_boundary_pe_left(zone)
  assert (I.getNodeFromPath(zone, 'NGonElements/ElementConnectivity')[1] == expt_ng_ec).all()
