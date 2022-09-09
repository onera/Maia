import pytest
import numpy as np

from maia.utils.yaml   import parse_yaml_cgns
from maia.pytree.node  import access as NA

def test_convert_value():

  #Basic values
  assert NA.convert_value(None) is None
  converted = NA.convert_value(12)
  assert  converted == np.array([12]) and converted.dtype==np.int32
  converted = NA.convert_value(-12.0)
  assert converted == np.array([-12.0]) and converted.dtype==np.float64
  converted = NA.convert_value("Spam, eggs")
  assert isinstance(converted, np.ndarray) and converted.dtype=='S1'
  assert converted.tobytes().decode() == "Spam, eggs"

  # Arrays
  np_array = np.array([[1,2,3], [4,5,6]], order='F')
  assert NA.convert_value(np_array) is np_array
  np_array = np.array([[1,2,3], [4,5,6]], order='C')
  converted = NA.convert_value(np_array)
  assert (converted == np_array).all() and converted.flags.f_contiguous == True

  # Iterables
  converted = NA.convert_value(["Spaaaaaam", "eggs"])
  assert isinstance(converted, np.ndarray) and converted.dtype=='S1'and converted.shape == (32,2)
  node = ['FakeNode', converted, [], 'FakeNode_t']
  assert NA.get_value(node) == ["Spaaaaaam", "eggs"]
  assert NA.convert_value([]).shape == (0,)
  assert NA.convert_value([[]]).shape == (1,0,)
  converted = NA.convert_value([13, 14, 15])
  assert (converted == np.array([13, 14, 15])).all() and converted.dtype==np.int32
  converted = NA.convert_value([13.3, 14.2, 15.3])
  assert (converted == np.array([13.3, 14.2, 15.3])).all() and converted.dtype==np.float64
  converted = NA.convert_value([['Spam', 'eggs'], ["Bacon"]])
  assert isinstance(converted, np.ndarray) and converted.dtype=='S1' and converted.shape == (32,2,2)

yt = """
MyNode UserDefinedData_t I4 [0]:
  SomeOtherNode DataArray_t:
"""
def test_name():
  node = parse_yaml_cgns.to_node(yt)
  assert NA.get_name(node) == 'MyNode'
  NA.set_name(node, 'NewName')
  assert NA.get_name(node) == 'NewName'
  with pytest.warns(RuntimeWarning):
    NA.set_name(node, "AVeryLongNameButCNGSIsLimitedTo32Characters")
  with pytest.raises(ValueError):
    NA.set_name(node, 12)

def test_value():
  node = parse_yaml_cgns.to_node(yt)
  assert NA.get_value(node) == np.array([0], np.int32)
  NA.set_value(node, "Toto")
  assert NA.get_value(node) == "Toto"
  NA.set_value(node, ["Spaaaaaam", "eggs"])
  assert NA.get_value(node) == ["Spaaaaaam", "eggs"]
  NA.set_value(node, [['Spam'], ["Bacon"]])
  assert NA.get_value(node) == [['Spam'], ['Bacon']]
  NA.set_value(node, [['Spam', 'eggs'], ["Bacon"], ['', 'bar']])
  assert NA.get_value(node) == [['Spam', 'eggs'], ['Bacon', ''], ['', 'bar']]

def test_label():
  node = parse_yaml_cgns.to_node(yt)
  assert NA.get_label(node) == 'UserDefinedData_t'
  NA.set_label(node, 'DataArray_t')
  assert NA.get_label(node) == 'DataArray_t'
  with pytest.warns(RuntimeWarning):
    NA.set_label(node, "FakeLabel_t")
  with pytest.raises(ValueError):
    NA.set_label(node, 12)

def test_children():
  node = parse_yaml_cgns.to_node(yt)
  child = NA.get_children(node)[0]
  assert child[0] == "SomeOtherNode"
  NA.set_children(node, [])
  assert NA.get_children(node) == []
  NA.add_child(node, child)
  assert len(NA.get_children(node)) == 1
  with pytest.raises(TypeError):
    NA.set_children(node, 12)

  other_child = ['SomeOtherNode', None, [], 'UserDefinedData_t']
  with pytest.raises(RuntimeError):
    NA.add_child(node, other_child)
