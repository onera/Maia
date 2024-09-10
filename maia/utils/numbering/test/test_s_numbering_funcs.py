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

def test_faceindex_to_idx():
  n_vtx = [7,5,3]
  n_cell = [6,4,2]
  idx = np.random.randint(1, 3*5*7+1, size=20)
  assert (s_numb.ijk_to_faceiIndex(*s_numb.faceiIndex_to_ijk(idx, n_cell, n_vtx), n_cell, n_vtx) == idx).all()
  assert (s_numb.ijk_to_facejIndex(*s_numb.facejIndex_to_ijk(idx, n_cell, n_vtx), n_cell, n_vtx) == idx).all()
  assert (s_numb.ijk_to_facekIndex(*s_numb.facekIndex_to_ijk(idx, n_cell, n_vtx), n_cell, n_vtx) == idx).all()

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





