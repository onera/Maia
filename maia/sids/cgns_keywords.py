import numpy as np
from cmaia.sids.cgns_keywords import *
import cmaia.sids.cgns_name as Name

B1, C1, I4, U4, I8, U8, R4, R8, X4, X8 = "B1", "C1", "I4", "U4", "I8", "U8", "R4", "R8", "X4", "X8"

cgns_to_dtype = {
  B1 : np.dtype(np.int8),        # 'b'
  C1 : np.dtype('S1'),           # 'c'
  I4 : np.dtype(np.int32),       # 'i'
  U4 : np.dtype(np.uint32),      # 'I'
  I8 : np.dtype(np.int64),       # 'l'
  U8 : np.dtype(np.uint64),      # 'L'
  R4 : np.dtype(np.float32),     # 'f'
  R8 : np.dtype(np.float64),     # 'd'
  X4 : np.dtype(np.complex64),   # 'F'
  X8 : np.dtype(np.complex128),  # 'D'
}
dtype_to_cgns = dict([(v,k) for k,v in cgns_to_dtype.items()])

cgns_types = tuple(cgns_to_dtype.keys())

dtypes = (np.int8,
          np.int32,
          np.uint32,
          np.int64,
          np.uint64,
          np.float32,
          np.float64,
          np.complex64,
          np.complex128,)
