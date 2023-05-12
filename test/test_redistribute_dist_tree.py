import pytest
import pytest_parallel
import os

import maia
import maia.io            as Mio
import maia.pytree        as PT

from   maia.utils         import test_utils   as TU
from   maia.algo.dist     import redistribute_tree

@pytest.mark.parametrize("policy", ["gather", "gather.0", "gather.1", "gather.2"])
@pytest_parallel.mark.parallel([2,3])
def test_redistribute_tree_U(policy, comm, write_output):
  
  # Reference directory and file
  ref_file = os.path.join(TU.mesh_dir, 'U_Naca0012_multizone.yaml')

  # Loading file
  dist_tree     = Mio.file_to_dist_tree(ref_file, comm)
  dist_tree_ref = PT.deep_copy(dist_tree)

  if not (comm.Get_size() == 2 and policy =="gather.2"):
    # Gather and uniform
    redistribute_tree(dist_tree  , comm, policy=policy)
    redistribute_tree(dist_tree, comm, policy='uniform')

    if write_output:
      out_dir   = maia.utils.test_utils.create_pytest_output_dir(comm)
      Mio.dist_tree_to_file(dist_tree, os.path.join(out_dir, 'out_tree.cgns'), comm)
    
    assert PT.is_same_tree(dist_tree, dist_tree_ref)

