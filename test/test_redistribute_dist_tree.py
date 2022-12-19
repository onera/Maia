from   pytest_mpi_check._decorator import mark_mpi_test
import pytest
import os
import numpy as np

import maia
import maia.io            as Mio
import maia.pytree        as PT
import maia.pytree.maia   as MT

from   maia.utils         import par_utils
from   maia.utils         import test_utils   as TU
from   maia.algo.dist     import redistribute as RDT
from   maia.pytree.yaml   import parse_yaml_cgns


from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype
dtype = 'I4' if pdm_gnum_dtype == np.int32 else 'I8'


@pytest.mark.parametrize("policy", ["gather", "gather.0", "gather.1", "gather.2", 'uniform'])
@mark_mpi_test([1,2,3])
def test_redistribute_tree_U(policy, sub_comm, write_output):
  
  # Reference directory and file
  ref_file = os.path.join(TU.mesh_dir, 'U_Naca0012_multizone.yaml')

  # Loading file
  dist_tree     = Mio.file_to_dist_tree(ref_file, sub_comm)
  dist_tree_ref = PT.deep_copy(dist_tree)

  if  not((sub_comm.Get_size()==1 and policy in ["gather.1", "gather.2"]) or
          (sub_comm.Get_size()==2 and policy in [            "gather.2"])):
    # Gather and uniform
    gather_tree = RDT.redistribute_tree(dist_tree  , sub_comm, policy=policy)
    dist_tree   = RDT.redistribute_tree(gather_tree, sub_comm, policy='uniform')

    if write_output:
      out_dir   = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
      Mio.dist_tree_to_file(dist_tree    , os.path.join(out_dir, 'out_tree.cgns'), sub_comm)
    
    assert PT.is_same_tree(dist_tree, dist_tree_ref)

