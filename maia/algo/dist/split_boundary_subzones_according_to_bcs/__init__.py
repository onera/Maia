import warnings
import maia.pytree as PT

from cmaia import dist_algo as cdist_algo

from maia.utils import require_cpp20

@require_cpp20
def split_boundary_subzones_according_to_bcs(t, comm):
  warnings.warn('This function is deprecated and will be removed in the next release', DeprecationWarning, stacklevel=3)
  for base in PT.iter_all_CGNSBase_t(t):
    cdist_algo.split_boundary_subzones_according_to_bcs(base, comm)
