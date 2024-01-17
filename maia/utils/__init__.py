from functools   import wraps

from cmaia       import cpp20_enabled 
from cmaia.utils import layouts

from .numbering import s_numbering_funcs as s_numbering
from .numbering import pr_utils

from .          import py_utils
from .parallel  import utils    as par_utils
from .ndarray   import np_utils as np_utils

def require_cpp20(f):
  """ A decorator checking if Maia has been compiled with CXX20 """
  @wraps(f)
  def inner(*args, **kwargs):
    if not cpp20_enabled:
      raise Exception(f"Functionnality {f.__name__} is unavailable because "
      "Maia has been compiled without C++ 20 support")
    return f(*args, **kwargs)
  return inner

def as_pdm_gnum(array):
  import Pypdm.Pypdm as PDM
  return np_utils.safe_int_cast(array, PDM.npy_pdm_gnum_dtype)
