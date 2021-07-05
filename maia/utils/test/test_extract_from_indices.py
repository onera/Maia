import pytest
import numpy as np
import cmaia.utils.extract_from_indices as EX
from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype

def test_extract_from_indicesint():
  a = np.array([0, 1, 2, 3, 4], dtype='int32')
  indices = np.array([0, 2, 4], dtype='int32')
  extract_a = EX.extract_from_indices(a, indices, 1, 0)
  expected_extract_a = np.array([0, 2, 4], dtype='int32')
  assert( (extract_a == expected_extract_a ).all() )


def test_extract_from_indices_pdm():
  a = np.array([0, 1, 2, 3, 4], dtype=pdm_gnum_dtype)
  indices = np.array([0, 2, 4], dtype='int32')
  extract_a = EX.extract_from_indices(a, indices, 1, 0)
  expected_extract_a = np.array([0, 2, 4], dtype=pdm_gnum_dtype)
  assert( (extract_a == expected_extract_a ).all() )

def test_extract_from_indices_double():
  #             |   0     |     1     |     2    |     3       |      4       |
  a = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.], dtype='double')
  indices = np.array([0, 2, 4], dtype='int32')
  extract_a = EX.extract_from_indices(a, indices, 3, 0)
  expected_extract_a = np.array([0., 1., 2., 6., 7., 8., 12., 13., 14.], dtype='double')
  assert( (extract_a == expected_extract_a ).all() )
