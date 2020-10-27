import pytest
import numpy             as NPY

from   maia.connectivity import connectivity_transform as CNT

@pytest.mark.parametrize("dtype", ["float"])
def test_pe_cgns_to_pdm_face_cell_bad_type(dtype):
  """
  """
  ngon_pe = NPY.array([[1, 0],
                       [1, 2],
                       [2, 3]], dtype=dtype, order='F')
  size = NPY.prod(ngon_pe.shape)
  dface_cell = NPY.empty(2*ngon_pe.shape[0], order='F', dtype=dtype)
  try:
    CNT.pe_cgns_to_pdm_face_cell(ngon_pe, dface_cell)
    assert False
  except TypeError:
    assert True

@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_pe_cgns_to_pdm_face_cell(dtype):
  """
  """
  ngon_pe = NPY.array([[1, 0],
                       [1, 2],
                       [2, 3]], dtype=dtype, order='F')
  size = NPY.prod(ngon_pe.shape)
  dface_cell = NPY.empty(2*ngon_pe.shape[0], order='F', dtype=dtype)
  CNT.pe_cgns_to_pdm_face_cell(ngon_pe, dface_cell)

  NPY.testing.assert_equal(dface_cell, NPY.array([1, 0, 1, 2, 2, 3], order='F', dtype=dtype))

@pytest.mark.parametrize("dtype", ["float"])
def test_compute_idx_local_bad_type(dtype):
  """
  """
  connect_g_idx = NPY.array([100, 104, 108, 112], dtype=dtype  , order='F')
  distrib       = NPY.array([100, 116, 100000  ], dtype=dtype  , order='F')
  connect_l_idx = NPY.empty((connect_g_idx.shape[0]+1), dtype='int32', order='F')
  try:
    CNT.compute_idx_local(connect_l_idx, connect_g_idx, distrib)
    assert False
  except TypeError:
    assert True

@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_compute_idx_local(dtype):
  """
  """
  connect_g_idx = NPY.array([100, 104, 108, 112], dtype=dtype  , order='F')
  distrib       = NPY.array([100, 116, 100000  ], dtype=dtype  , order='F')
  connect_l_idx = NPY.empty((connect_g_idx.shape[0]+1), dtype='int32', order='F')
  CNT.compute_idx_local(connect_l_idx, connect_g_idx, distrib)
  print(connect_l_idx)
  NPY.testing.assert_equal(connect_l_idx, NPY.array([0, 4, 8, 12, 16], order='F', dtype=dtype))
