import pytest

import maia.utils.py_utils as py_utils

def test_camel_to_snake():
  assert py_utils.camel_to_snake("already_snake") == "already_snake"
  assert py_utils.camel_to_snake("stringInCamelCase") == "string_in_camel_case"
  assert py_utils.camel_to_snake("StringInCamelCase") == "string_in_camel_case"
  assert py_utils.camel_to_snake("stringINCamelCase", keep_upper=True) == "string_IN_camel_case"

def test_flatten():
  l = [1,2,3,4,5,6]
  assert list(py_utils.flatten(l)) == [1,2,3,4,5,6]
  l = [[1,2,3],[4,5],[6]]
  assert list(py_utils.flatten(l)) == [1,2,3,4,5,6]
  l = [[1,[2,3]],[4,5],[6]]
  assert list(py_utils.flatten(l)) == [1,2,3,4,5,6]
  l = [[1,[2,[3]]],[4,[5]],[6]]
  assert list(py_utils.flatten(l)) == [1,2,3,4,5,6]

def test_list_or_only_elt():
  assert py_utils.list_or_only_elt([42]) == 42
  input = [1,2,3, "nous irons au bois"]
  assert py_utils.list_or_only_elt(input) is input

def test_bucket_split():
  l = ["apple", "banana", "orange", "watermelon", "grappe", "pear"]
  assert py_utils.bucket_split(l, lambda e: len(e)) == \
      [ [], [], [], [], ["pear"], ["apple"], ["banana", "orange", "grappe"], [], [], [], ["watermelon"]]
  assert py_utils.bucket_split(l, lambda e: len(e), compress=True) == \
      [["pear"], ["apple"], ["banana", "orange", "grappe"], ["watermelon"]]
  assert py_utils.bucket_split(l, lambda e: len(e), size=13) == \
      [ [], [], [], [], ["pear"], ["apple"], ["banana", "orange", "grappe"], [], [], [], ["watermelon"], [], []]
  with pytest.raises(IndexError):
    py_utils.bucket_split(l, lambda e: len(e), size=4) #To short return list

def test_are_overlapping():
  assert py_utils.are_overlapping([1,6], [2,9]) == True
  assert py_utils.are_overlapping([5,9], [1,3]) == False
  assert py_utils.are_overlapping([1,4], [4,9], strict=False) == True
  assert py_utils.are_overlapping([1,4], [4,9], strict=True) == False
  assert py_utils.are_overlapping([4,4], [2,4], strict=False) == True
  assert py_utils.are_overlapping([4,4], [2,4], strict=True) == False

def test_is_subset_l():
  L = [2,8,10,3,3]
  assert py_utils.is_subset_l([2],        L) == True
  assert py_utils.is_subset_l([10,3],     L) == True
  assert py_utils.is_subset_l([10,3,3],   L) == True
  assert py_utils.is_subset_l([3,2,8],    L) == True
  assert py_utils.is_subset_l([1],        L) == False
  assert py_utils.is_subset_l([3,8,2],    L) == False
  assert py_utils.is_subset_l([10,3,3,1], L) == False

def test_append_unique():
  L = [1,2,3]
  py_utils.append_unique(L, 4)
  assert L == [1,2,3,4]
  py_utils.append_unique(L, 4)
  assert L == [1,2,3,4]

def test_expects_one():
  assert py_utils.expects_one([42]) == 42
  with pytest.raises(RuntimeError) as e:
    py_utils.expects_one([42,43])
  assert str(e.value) == 'Multiple elem found in list'
  with pytest.raises(RuntimeError) as e:
    py_utils.expects_one([], ('fruit', 'grocery list'))
  assert str(e.value) == 'fruit not found in grocery list'

def test_loop_from():
  L = ["apple", "banana", "orange", "mango"]
  assert list(py_utils.loop_from(L, 0)) == L
  assert list(py_utils.loop_from(L, 1)) == ["banana", "orange", "mango", "apple"]
  assert list(py_utils.loop_from(L, 3)) == ["mango", "apple", "banana", "orange"]


def test_find_cartesian_vector_names():
  names = ["Tata","TotoY","TotoZ","Titi","totoX"]
  assert py_utils.find_cartesian_vector_names(names) == []
  names.append("TotoX")
  assert py_utils.find_cartesian_vector_names(names) == ["Toto"]

def test_get_ordered_subset():
  L = [2,8,10,3,3]
  assert py_utils.get_ordered_subset([10,8,3], L) == (8,10,3)
  assert py_utils.get_ordered_subset([10,2], L)   == None
  assert py_utils.get_ordered_subset([10,3], L)   == (10,3)
  assert py_utils.get_ordered_subset([3,2], L)    == py_utils.get_ordered_subset([2,3], L) == (3,2)
  assert py_utils.get_ordered_subset([8], L)      == (8,)
  assert py_utils.get_ordered_subset([], L)       == ()
  assert py_utils.get_ordered_subset([3,8,2,10,3], L) == (3,2,8,10,3)

def test_uniform_distribution_at():
  assert py_utils.uniform_distribution_at(15,0,3) == (0,5)
  assert py_utils.uniform_distribution_at(15,1,3) == (5,10)
  assert py_utils.uniform_distribution_at(15,2,3) == (10,15)

  assert py_utils.uniform_distribution_at(17,0,3) == (0,6)
  assert py_utils.uniform_distribution_at(17,1,3) == (6,12)
  assert py_utils.uniform_distribution_at(17,2,3) == (12,17)

def test_is_before():
  L = [2,8,10,3,3]
  assert py_utils.is_before(L, 8, 10) == True
  assert py_utils.is_before(L, 8, 9 ) == True
  assert py_utils.is_before(L, 8, 2 ) == False
  assert py_utils.is_before(L, 7, 2 ) == False
  assert py_utils.is_before(L, 7, 14) == False

def test_str_to_bools():
  assert py_utils.str_to_bools(4, 'none') == [False, False, False, False]
  assert py_utils.str_to_bools(2, 'all') == [True, True]
  assert py_utils.str_to_bools(2, 'leaf') == [False, True]
  assert py_utils.str_to_bools(3, 'ancestors') == [True, True, False]
  with pytest.raises(ValueError):
    py_utils.str_to_bools(4, 'wrong')

