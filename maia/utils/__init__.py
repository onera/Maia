from cmaia.utils import layouts

from .numbering import s_numbering_funcs as s_numbering

from .          import py_utils
from .parallel  import utils    as par_utils
from .ndarray   import np_utils as np_utils

def as_pdm_gnum(array):
  import Pypdm.Pypdm as PDM
  return np_utils.safe_int_cast(array, PDM.npy_pdm_gnum_dtype)
