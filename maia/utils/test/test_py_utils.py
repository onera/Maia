import numpy as np
import maia.utils.py_utils as py_utils

def test_list_or_only_elt():
  assert py_utils.list_or_only_elt([42]) == 42
  input = [1,2,3, "nous irons aux bois"]
  assert py_utils.list_or_only_elt(input) is input

def test_interweave_arrays():
  first  = np.array([1,2,3], dtype=np.int32)
  second = np.array([11,22,33], dtype=np.int32)
  third  = np.array([111,222,333], dtype=np.int32)
  assert (py_utils.interweave_arrays([first]) == [1,2,3]).all()
  assert (py_utils.interweave_arrays([second, third]) == \
      [11,111,22,222,33,333]).all()
  assert (py_utils.interweave_arrays([first, second, third]) == \
      [1,11,111,2,22,222,3,33,333]).all()
