from cmaia import dist_algo as cdist_algo

import maia.pytree as PT
from maia.utils import require_cpp20

@require_cpp20
def put_boundary_first(t, comm):
  for base in PT.iter_all_CGNSBase_t(t):
    cdist_algo.put_boundary_first(base)

