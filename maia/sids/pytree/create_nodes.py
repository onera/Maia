import numpy as np

import maia.sids.cgns_keywords as CGK
import maia.utils.py_utils as PYU

def convert_value(value):
  """
  Convert a Python input to a compliant pyCGNS value
  """
  result = None
  if value is None:
    return result
  # value as a numpy: convert to fortran order
  if isinstance(value, np.ndarray):
    # print(f"-> value as a numpy")
    if value.flags.f_contiguous:
      result = value
    else:
      result = np.asfortranarray(value)
  # value as a single value
  elif isinstance(value, float):   # "R8"
    # print(f"-> value as float")
    result = np.array([value],'d')
  elif isinstance(value, int):     # "I4"
    # print(f"-> value as int")
    result = np.array([value],'i')
  elif isinstance(value, str):     # "C1"
    # print(f"-> value as str with {CGK.cgns_to_dtype[CGK.C1]}")
    result = np.array([c for c in value], CGK.cgns_to_dtype[CGK.C1])
  elif isinstance(value, CGK.dtypes):
    # print(f"-> value as CGK.dtypes with {np.dtype(value)}")
    result = np.array([value], np.dtype(value))
  # value as an iterable (list, tuple, set, ...)
  elif isinstance(value, PYU.Iterable):
    # print(f"-> value as PYU.Iterable : {PYU.flatten(value)}")
    try:
      first_value = next(PYU.flatten(value))
      if isinstance(first_value, float):                       # "R8"
        # print(f"-> first_value as float")
        result = np.array(value, dtype=np.float64, order='F')
      elif isinstance(first_value, int):                       # "I4"
        # print(f"-> first_value as int")
        result = np.array(value, dtype=np.int32, order='F')
      elif isinstance(first_value, str):                       # "C1"
        # print(f"-> first_value as with {CGK.cgns_to_dtype[CGK.C1]}")
        # WARNING: string numpy is limited to rank=2
        size = max([len(v) for v in PYU.flatten(value)])
        if isinstance(value[0], str):
          v = np.empty( (size,len(value) ), dtype='c', order='F')
          for c, i in enumerate(value):
            s = min(len(i),size)
            v[:,c] = ' '
            v[0:s,c] = i[0:s]
          result = v
        else:
          v = np.empty( (size,len(value[0]),len(value) ), dtype='c', order='F')
          for c in range(len(value)):
            for d in range(len(value[c])):
              s = min(len(value[c][d]),size)
              v[:,d,c] = ' '
              v[0:s,d,c] = value[c][d][0:s]
          result = v
      elif isinstance(first_value, CGK.dtypes):
        result = np.array(value, dtype=np.dtype(first_value), order='F')
    except StopIteration:
      # empty iterable
      result = np.array(value, dtype=np.int32, order='F')
  else:
    # print(f"-> value as unknown type")
    result = np.array([value], order='F')
  return result
