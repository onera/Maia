import pytest
import pytest_parallel
import numpy as np

from maia       import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.utils import layouts

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
    layouts.pe_cgns_to_pdm_face_cell(ngon_pe, dface_cell)
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
  layouts.pe_cgns_to_pdm_face_cell(ngon_pe, dface_cell)

  np.testing.assert_equal(dface_cell, np.array([1, 0, 1, 2, 2, 3], order='F', dtype=dtype))

def test_strided_connectivity_to_pe():
  connect_idx = np.array([0, 1, 3, 4, 6], dtype=np.int32)  
  connect     = np.array([12,  16,-17,   -4,   -8,7], dtype=int)  
  pe = np.empty((4,2), order='F', dtype=int)
  layouts.strided_connectivity_to_pe(connect_idx, connect, pe)
  assert (pe == np.array([[12,0], [16,17], [0,4], [7,8]])).all()

def test_extract_from_indicesint():
  a = np.array([0, 1, 2, 3, 4], dtype='int32')
  indices = np.array([0, 2, 4], dtype='int32')
  extract_a = layouts.extract_from_indices(a, indices, 1, 0)
  expected_extract_a = np.array([0, 2, 4], dtype='int32')
  assert( (extract_a == expected_extract_a ).all() )


def test_extract_from_indices_pdm():
  a = np.array([0, 1, 2, 3, 4], dtype=pdm_gnum_dtype)
  indices = np.array([0, 2, 4], dtype='int32')
  extract_a = layouts.extract_from_indices(a, indices, 1, 0)
  expected_extract_a = np.array([0, 2, 4], dtype=pdm_gnum_dtype)
  assert( (extract_a == expected_extract_a ).all() )

def test_extract_from_indices_double():
  #             |   0     |     1     |     2    |     3       |      4       |
  a = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.], dtype='double')
  indices = np.array([0, 2, 4], dtype='int32')
  extract_a = layouts.extract_from_indices(a, indices, 3, 0)
  expected_extract_a = np.array([0., 1., 2., 6., 7., 8., 12., 13., 14.], dtype='double')
  assert( (extract_a == expected_extract_a ).all() )

