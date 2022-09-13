import pytest

from maia.pytree.sids import utils

def test_bucket_split():
  l = ["apple", "banana", "orange", "watermelon", "grappe", "pear"]
  assert utils.bucket_split(l, lambda e: len(e)) == \
      [ [], [], [], [], ["pear"], ["apple"], ["banana", "orange", "grappe"], [], [], [], ["watermelon"]]
  assert utils.bucket_split(l, lambda e: len(e), compress=True) == \
      [["pear"], ["apple"], ["banana", "orange", "grappe"], ["watermelon"]]
  assert utils.bucket_split(l, lambda e: len(e), size=13) == \
      [ [], [], [], [], ["pear"], ["apple"], ["banana", "orange", "grappe"], [], [], [], ["watermelon"], [], []]
  with pytest.raises(IndexError):
    utils.bucket_split(l, lambda e: len(e), size=4) #To short return list

def test_are_overlapping():
  assert utils.are_overlapping([1,6], [2,9]) == True
  assert utils.are_overlapping([5,9], [1,3]) == False
  assert utils.are_overlapping([1,4], [4,9], strict=False) == True
  assert utils.are_overlapping([1,4], [4,9], strict=True) == False
  assert utils.are_overlapping([4,4], [2,4], strict=False) == True
  assert utils.are_overlapping([4,4], [2,4], strict=True) == False

def test_append_unique():
  L = [1,2,3]
  utils.append_unique(L, 4)
  assert L == [1,2,3,4]
  utils.append_unique(L, 4)
  assert L == [1,2,3,4]

def test_expects_one():
  assert utils.expects_one([42]) == 42
  with pytest.raises(RuntimeError) as e:
    utils.expects_one([42,43])
  assert str(e.value) == 'Multiple elem found in list'
  with pytest.raises(RuntimeError) as e:
    utils.expects_one([], ('fruit', 'grocery list'))
  assert str(e.value) == 'fruit not found in grocery list'

