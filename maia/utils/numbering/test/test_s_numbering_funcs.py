import pytest
import numpy as np
from maia.utils.numbering import s_numbering_funcs as s_numb

def test_ijk_to_index():
  assert s_numb.ijk_to_index(1,1,1,[3,3,3]) ==   1
  assert s_numb.ijk_to_index(1,2,3,[7,5,3]) == 1+1*7+2*7*5
  assert s_numb.ijk_to_index(7,5,3,[7,5,3]) == 7+4*7+2*7*5

def test_index_to_ijk():
  assert s_numb.index_to_ijk(1,[7,5,3]) == (1,1,1)
  assert s_numb.index_to_ijk(78,[7,5,3]) == (1,2,3)
  assert s_numb.index_to_ijk(105,[7,5,3]) == (7,5,3)
  idx = np.random.randint(1, 3*5*7+1, size=20)
  assert (s_numb.ijk_to_index(*s_numb.index_to_ijk(idx, [7,5,3]), [7,5,3]) == idx).all()

def test_ijk_to_faceiIndex():
  assert s_numb.ijk_to_faceiIndex(1,1,1,[7,5,3],[8,6,4]) ==   1
  assert s_numb.ijk_to_faceiIndex(3,2,1,[7,5,3],[8,6,4]) ==  11
  assert s_numb.ijk_to_faceiIndex(7,5,3,[7,5,3],[8,6,4]) == 119
  assert s_numb.ijk_to_faceiIndex(8,5,3,[7,5,3],[8,6,4]) == 8+4*8+2*8*5

def test_ijk_to_facejIndex():
  assert s_numb.ijk_to_facejIndex(1,1,1,[7,5,3],[8,6,4]) == 121
  assert s_numb.ijk_to_facejIndex(3,2,1,[7,5,3],[8,6,4]) == 130
  assert s_numb.ijk_to_facejIndex(7,5,3,[7,5,3],[8,6,4]) == 239
  assert s_numb.ijk_to_facejIndex(7,6,3,[7,5,3],[8,6,4]) == 7+5*7+2*6*7+120

def test_ijk_to_facekIndex():
  assert s_numb.ijk_to_facekIndex(1,1,1,[7,5,3],[8,6,4]) == 247
  assert s_numb.ijk_to_facekIndex(3,2,1,[7,5,3],[8,6,4]) == 256
  assert s_numb.ijk_to_facekIndex(7,5,3,[7,5,3],[8,6,4]) == 351
  assert s_numb.ijk_to_facekIndex(7,5,4,[7,5,3],[8,6,4]) == 7+4*7+3*7*5+120+126

def test_PE_idx_from_i_face_idx():
  PE = s_numb.PE_idx_from_i_face_idx(np.arange(1,21), [4,2,2], [5,3,3])
  n_face = 5*2*2 + 4*3*2 + 4*2*3
  expected_pe = np.array([ 1,0,  1,2,   2,3,   3,4,   4,0,  5,0,    5,6,
                           6,7,  7,8,   8,0,   9,0,  9,10, 10,11, 11,12,
                          12,0, 13,0, 13,14, 14,15, 15,16,  16,0       ]).reshape(20,2)
  expected_pe += n_face * (expected_pe > 0)
  assert(PE.shape == (20,2))
  assert (PE == expected_pe).all()

def test_facevtx_from_i_face_idx():
  facevtx = s_numb.facevtx_from_i_face_idx(np.arange(1,21), [4,2,2], [5,3,3])
  assert (facevtx[4* 0:4* 1] == [1,16,21,6]).all() #Bnd min
  assert (facevtx[4*10:4*11] == [16,31,36,21]).all() #Bnd max
  assert (facevtx[4*13:4*15] == [19,24,39,34,20,25,40,35]).all() #Internal

def test_PE_idx_from_j_face_idx():
  PE = s_numb.PE_idx_from_j_face_idx(np.arange(1,25), [4,2,2], [5,3,3])
  n_face = 5*2*2 + 4*3*2 + 4*2*3
  expected_pe = np.array([ 1,0,   2,0,   3,0,   4,0,  1,5,  2,6,  3,7,  4,8,
                           5,0,   6,0,   7,0,   8,0,  9,0, 10,0, 11,0, 12,0,
                          9,13, 10,14, 11,15, 12,16, 13,0, 14,0, 15,0, 16,0]).reshape(24,2)
  expected_pe += n_face * (expected_pe > 0)
  assert(PE.shape == (24,2))
  assert (PE == expected_pe).all()

def test_facevtx_from_j_face_idx():
  facevtx = s_numb.facevtx_from_j_face_idx(np.arange(1,25), [4,2,2], [5,3,3])
  assert (facevtx[4* 0:4* 1] == [1,2,17,16]).all() #Bnd min
  assert (facevtx[4*10:4*11] == [13,28,29,14]).all() #Bnd max
  assert (facevtx[4*16:4*18] == [21,36,37,22,22,37,38,23]).all() #Internal

def test_PE_idx_from_k_face_idx():
  PE = s_numb.PE_idx_from_k_face_idx(np.arange(1,25), [4,2,2], [5,3,3])
  n_face = 5*2*2 + 4*3*2 + 4*2*3
  expected_pe = np.array([1,0,  2,0,  3,0,  4,0,  5,0,  6,0,  7,0,  8,0,
                          1,9, 2,10, 3,11, 4,12, 5,13, 6,14, 7,15, 8,16,
                          9,0, 10,0, 11,0, 12,0, 13,0, 14,0, 15,0, 16,0]).reshape(24,2)
  expected_pe += n_face * (expected_pe > 0)
  assert(PE.shape == (24,2))
  assert (PE == expected_pe).all()

def test_facevtx_from_k_face_idx():
  facevtx = s_numb.facevtx_from_k_face_idx(np.arange(1,25), [4,2,2], [5,3,3])
  assert (facevtx[4* 4:4* 5] == [6,11,12,7]).all() #Bnd min
  assert (facevtx[4*22:4*23] == [38,39,44,43]).all() #Bnd max
  assert (facevtx[4*11:4*13] == [19,20,25,24,21,22,27,26]).all() #Internal
