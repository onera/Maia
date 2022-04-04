import pytest
import numpy as np

from maia.connectivity import connectivity_transform as CNT
from maia.utils        import parse_yaml_cgns

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
  assert (CNT.get_ngon_pe_local(ngon) == np.array([[1,0], [2,0], [3,4], [0,4]])).all()

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
