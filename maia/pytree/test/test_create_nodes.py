import numpy as np
import Converter.Internal as I

from maia.pytree import create_nodes as CN

def test_convert_value():

  #Basic values
  assert CN.convert_value(None) is None
  converted = CN.convert_value(12)
  assert  converted == np.array([12]) and converted.dtype==np.int32
  converted = CN.convert_value(-12.0)
  assert converted == np.array([-12.0]) and converted.dtype==np.float64
  converted = CN.convert_value("Spam, eggs")
  assert isinstance(converted, np.ndarray) and converted.dtype=='S1'
  assert converted.tobytes().decode() == "Spam, eggs"

  # Arrays
  np_array = np.array([[1,2,3], [4,5,6]], order='F')
  assert CN.convert_value(np_array) is np_array
  np_array = np.array([[1,2,3], [4,5,6]], order='C')
  converted = CN.convert_value(np_array)
  assert (converted == np_array).all() and converted.flags.f_contiguous == True

  # Iterables
  converted = CN.convert_value(["Spaaaaaam", "eggs"])
  assert isinstance(converted, np.ndarray) and converted.dtype=='S1'and converted.shape == (32,2)
  node = ['FakeNode', converted, [], 'FakeNode_t']
  assert I.getValue(node) == ["Spaaaaaam", "eggs"]
  assert CN.convert_value([]).shape == (0,)
  assert CN.convert_value([[]]).shape == (1,0,)
  converted = CN.convert_value([13, 14, 15])
  assert (converted == np.array([13, 14, 15])).all() and converted.dtype==np.int32
  converted = CN.convert_value([13.3, 14.2, 15.3])
  assert (converted == np.array([13.3, 14.2, 15.3])).all() and converted.dtype==np.float64
  converted = CN.convert_value([['Spam', 'eggs'], ["Bacon"]])
  assert isinstance(converted, np.ndarray) and converted.dtype=='S1' and converted.shape == (32,2,2)
