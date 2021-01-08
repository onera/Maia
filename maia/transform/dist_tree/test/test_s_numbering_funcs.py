import pytest
from maia.transform.dist_tree import s_numbering_funcs as s_numb

def test_ijk_to_index():
  assert s_numb.ijk_to_index(1,1,1,[3,3,3]) ==   1
  assert s_numb.ijk_to_index(1,2,3,[7,5,3]) ==  78
  assert s_numb.ijk_to_index(7,5,3,[7,5,3]) == 105

def test_ijk_to_faceiIndex():
  assert s_numb.ijk_to_faceiIndex(1,1,1,[7,5,3],[8,6,4]) ==   1
  assert s_numb.ijk_to_faceiIndex(3,2,1,[7,5,3],[8,6,4]) ==  11
  assert s_numb.ijk_to_faceiIndex(7,5,3,[7,5,3],[8,6,4]) == 119
  assert s_numb.ijk_to_faceiIndex(8,5,3,[7,5,3],[8,6,4]) == 120

def test_ijk_to_facejIndex():
  assert s_numb.ijk_to_facejIndex(1,1,1,[7,5,3],[8,6,4]) == 121
  assert s_numb.ijk_to_facejIndex(3,2,1,[7,5,3],[8,6,4]) == 130
  assert s_numb.ijk_to_facejIndex(7,5,3,[7,5,3],[8,6,4]) == 239
  assert s_numb.ijk_to_facejIndex(7,6,3,[7,5,3],[8,6,4]) == 246

def test_ijk_to_facekIndex():
  assert s_numb.ijk_to_facekIndex(1,1,1,[7,5,3],[8,6,4]) == 247
  assert s_numb.ijk_to_facekIndex(3,2,1,[7,5,3],[8,6,4]) == 256
  assert s_numb.ijk_to_facekIndex(7,5,3,[7,5,3],[8,6,4]) == 351
  assert s_numb.ijk_to_facekIndex(7,5,4,[7,5,3],[8,6,4]) == 386

def test_ijk_to_faceIndices():
  assert s_numb.ijk_to_faceIndices(1,1,1,[7,5,3],[8,6,4]) == (  1,121,247)
  assert s_numb.ijk_to_faceIndices(3,2,1,[7,5,3],[8,6,4]) == ( 11,130,256)
  assert s_numb.ijk_to_faceIndices(7,5,3,[7,5,3],[8,6,4]) == (119,239,351)

def test_compute_fi_from_ijk():
  assert s_numb.compute_fi_from_ijk(5,4,3)    == ((5,4,3),(5,5,3),(5,5,4),(5,4,4),(4,4,3),(5,4,3))
  assert s_numb.compute_fi_from_ijk(6,4,3,is_max=True) == ((6,4,3),(6,5,3),(6,5,4),(6,4,4),(5,4,3),0)
  assert s_numb.compute_fi_from_ijk(1,4,3,is_min=True) == ((1,4,3),(1,4,4),(1,5,4),(1,5,3),(1,4,3),0)

def test_compute_fj_from_ijk():
  assert s_numb.compute_fj_from_ijk(5,4,3)    == ((5,4,3),(5,4,4),(6,4,4),(6,4,3),(5,3,3),(5,4,3))
  assert s_numb.compute_fj_from_ijk(5,5,3,is_max=True) == ((5,5,3),(5,5,4),(6,5,4),(6,5,3),(5,4,3),0)
  assert s_numb.compute_fj_from_ijk(5,1,3,is_min=True) == ((5,1,3),(6,1,3),(6,1,4),(5,1,4),(5,1,3),0)

def test_compute_fk_from_ijk():
  assert s_numb.compute_fk_from_ijk(5,4,3)    == ((5,4,3),(6,4,3),(6,5,3),(5,5,3),(5,4,2),(5,4,3))
  assert s_numb.compute_fk_from_ijk(5,4,4,is_max=True) == ((5,4,4),(6,4,4),(6,5,4),(5,5,4),(5,4,3),0)
  assert s_numb.compute_fk_from_ijk(5,4,1,is_min=True) == ((5,4,1),(5,5,1),(6,5,1),(6,4,1),(5,4,1),0)

