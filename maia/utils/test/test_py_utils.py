import pytest
import Converter.Internal as I
import numpy as np
import maia.utils.py_utils as py_utils
from   maia.utils        import parse_yaml_cgns

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

def test_get_ordered_subset():
  L = [2,8,10,3,3]
  assert py_utils.get_ordered_subset([10,8,3], L) == (8,10,3)
  assert py_utils.get_ordered_subset([10,2], L)   == None
  assert py_utils.get_ordered_subset([10,3], L)   == (10,3)
  assert py_utils.get_ordered_subset([3,2], L)    == py_utils.get_ordered_subset([2,3], L) == (3,2)
  assert py_utils.get_ordered_subset([8], L)      == (8,)
  assert py_utils.get_ordered_subset([], L)       == ()
  assert py_utils.get_ordered_subset([3,8,2,10,3], L) == (3,2,8,10,3)

def test_is_before():
  L = [2,8,10,3,3]
  assert py_utils.is_before(L, 8, 10) == True
  assert py_utils.is_before(L, 8, 9 ) == True
  assert py_utils.is_before(L, 8, 2 ) == False
  assert py_utils.is_before(L, 7, 2 ) == False
  assert py_utils.is_before(L, 7, 14) == False

def test_interweave_arrays():
  first  = np.array([1,2,3], dtype=np.int32)
  second = np.array([11,22,33], dtype=np.int32)
  third  = np.array([111,222,333], dtype=np.int32)
  assert (py_utils.interweave_arrays([first]) == [1,2,3]).all()
  assert (py_utils.interweave_arrays([second, third]) == \
      [11,111,22,222,33,333]).all()
  assert (py_utils.interweave_arrays([first, second, third]) == \
      [1,11,111,2,22,222,3,33,333]).all()

def test_single_dim_pr_to_pl():
  no_dist = py_utils.single_dim_pr_to_pl(np.array([[20, 25]]))
  assert no_dist.dtype == np.array([[20,25]]).dtype
  assert (no_dist == np.arange(20, 25+1)).all()
  assert no_dist.ndim == 2 and no_dist.shape[0] == 1
  dist = py_utils.single_dim_pr_to_pl(np.array([[20, 25]], dtype=np.int32), np.array([10,15,20]))
  assert dist.dtype == np.int32
  assert (dist == np.arange(10+20, 15+20)).all()
  assert dist.ndim == 2 and dist.shape[0] == 1
  with pytest.raises(AssertionError):
    py_utils.single_dim_pr_to_pl(np.array([[20, 25], [1,1]]))

def test_sizes_to_indices():
  assert(py_utils.sizes_to_indices([]) == np.zeros(1))
  assert(py_utils.sizes_to_indices([5,3,5,10]) == np.array([0,5,8,13,23])).all()
  assert(py_utils.sizes_to_indices([5,0,0,10]) == np.array([0,5,5,5,15])).all()
  assert py_utils.sizes_to_indices([5,0,0,10], np.int32).dtype == np.int32
  assert py_utils.sizes_to_indices([5,0,0,10], np.int64).dtype == np.int64

def test_reverse_connectivity():
  ids   = np.array([8,51,6,30,29])
  idx   = np.array([0,3,6,10,13,17])
  array = np.array([7,29,32, 32,11,13, 4,32,29,61, 32,4,13, 44,11,32,7])

  r_ids, r_idx, r_array = py_utils.reverse_connectivity(ids, idx, array)

  assert (r_ids == [4,7,11,13,29,32,44,61]).all()
  assert (r_idx == [0,2,4,6,8,10,15,16,17]).all()
  assert (r_array == [30,6, 8,29, 51,29, 51,30,  8, 6, 8,6,29,30,51, 29, 6]).all()

def test_multi_arange():
  # With only one start/stop, same as np.arange
  assert (py_utils.multi_arange([0], [10]) == [0,1,2,3,4,5,6,7,8,9]).all()

  assert (py_utils.multi_arange([0,100], [10,105]) == [0,1,2,3,4,5,6,7,8,9,  100,101,102,103,104]).all()

  # Empty aranges produce no values
  assert (py_utils.multi_arange([1,3,4,6], [1,5,7,6]) == [ 3,4, 4,5,6 ]).all()

  # No start/stop
  assert py_utils.multi_arange(np.empty(0, np.int64), np.empty(0, np.int64)).size == 0

def test_arange_with_jumps():
  assert (py_utils.arange_with_jumps([0         ,5   , 10      , 13  , 18   , 20], \
                                     [False     ,True, False   , True, False]) == \
                                     [0,1,2,3,4      , 10,11,12      , 18,19]).all()



def test_roll_from():
  assert (py_utils.roll_from(np.array([2,4,8,16]), start_idx = 1) == [4,8,16,2]).all()
  assert (py_utils.roll_from(np.array([2,4,8,16]), start_value = 4) == [4,8,16,2]).all()
  assert (py_utils.roll_from(np.array([2,4,8,16]), start_value = 8, reverse=True) == [8,4,2,16]).all()
  with pytest.raises(AssertionError):
    py_utils.roll_from(np.array([2,4,8,16]), start_idx = 1, start_value = 8)

def test_others_mask():
  array = np.array([2,4,6,1,3,5])
  assert (py_utils.others_mask(array, np.empty(0, np.int32)) == [1,1,1,1,1,1]).all()
  assert (py_utils.others_mask(array, np.array([2,1]))       == [1,0,0,1,1,1]).all()
  assert (py_utils.others_mask(array, np.array([0,1,3,4,5])) == [0,0,1,0,0,0]).all()

def test_concatenate_np_arrays():
  a1 = np.array([2, 4, 6, 8])
  a2 = np.array([10, 20, 30, 40, 50, 60])
  a3 = np.array([100])
  av = np.empty(0, np.int64)
  array_idx, array = py_utils.concatenate_np_arrays([a1,a3,av,a2])
  assert (array_idx == [0,4,5,5,11]).all()
  assert (array == [2,4,6,8,100,10,20,30,40,50,60]).all()
  assert array.dtype == np.int64

  array_idx, array = py_utils.concatenate_np_arrays([av])
  assert (array_idx == [0,0]).all()
  assert (array == []).all()

def test_concatenate_point_list():
  pl1 = np.array([[2, 4, 6, 8]])
  pl2 = np.array([[10, 20, 30, 40, 50, 60]])
  pl3 = np.array([[100]])
  plvoid = np.empty((1,0))

  #No pl at all in the mesh
  none_idx, none = py_utils.concatenate_point_list([])
  assert none_idx == [0]
  assert isinstance(none, np.ndarray)
  assert none.shape == (0,)

  #A pl, but with no data
  empty_idx, empty = py_utils.concatenate_point_list([plvoid])
  assert (none_idx == [0,0]).all()
  assert isinstance(empty, np.ndarray)
  assert empty.shape == (0,)

  # A pl with data
  one_idx, one = py_utils.concatenate_point_list([pl1])
  assert (one_idx == [0,4]).all()
  assert (one     == pl1[0]).all()

  # Several pl
  merged_idx, merged = py_utils.concatenate_point_list([pl1, pl2, pl3])
  assert (merged_idx == [0, pl1.size, pl1.size+pl2.size, pl1.size+pl2.size+pl3.size]).all()
  assert (merged[0:pl1.size]                 == pl1[0]).all()
  assert (merged[pl1.size:pl1.size+pl2.size] == pl2[0]).all()
  assert (merged[pl1.size+pl2.size:]         == pl3[0]).all()
  # Several pl, some with no data
  merged_idx, merged = py_utils.concatenate_point_list([pl1, plvoid, pl2])
  assert (merged_idx == [0, 4, 4, 10]).all()
  assert (merged[0:4 ] == pl1[0]).all()
  assert (merged[4:10] == pl2[0]).all()

def test_any_in_range():
  assert py_utils.any_in_range([3,4,1,6,12,3], 2, 20, strict=False)
  assert not py_utils.any_in_range([3,4,2,6,12,3], 15, 20, strict=False)
  assert py_utils.any_in_range([3,4,1,6,12,3], 12, 20, strict=False)
  assert not py_utils.any_in_range([3,4,1,6,12,3], 12, 20, strict=True)

def test_all_in_range():
  assert py_utils.all_in_range([3,4,5,6,12,3], 2, 20, strict=False)
  assert not py_utils.all_in_range([18,4,2,17,16,3], 15, 20, strict=False)
  assert not py_utils.all_in_range([3,4,1,6,12,3], 3, 20, strict=True)
