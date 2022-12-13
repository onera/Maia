from   pytest_mpi_check._decorator import mark_mpi_test
import pytest
import os
import numpy as np

import maia
import maia.io            as Mio
import maia.pytree        as PT
import maia.pytree.maia   as MT

from   maia.utils         import par_utils
from   maia.algo.dist     import redistribute_tree as RDT
from   maia.pytree.yaml   import parse_yaml_cgns

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype
dtype = 'I4' if pdm_gnum_dtype == np.int32 else 'I8'


@mark_mpi_test([1, 2, 3])
def test_redistribute_zone_U(sub_comm):
  
  # Reference directory and file
  ref_dir  = os.path.join(os.path.dirname(__file__), 'references')
  ref_file = os.path.join(ref_dir, f'cube_bcdataset_and_periodic.yaml')
  
  print(ref_file)
  dist_tree   = Mio.file_to_dist_tree(ref_file, sub_comm)
  gather_tree = dist_tree

  for zone in PT.get_all_Zone_t(gather_tree):
    gather_zone = RDT.redistribute_zone(zone, par_utils.gathering_distribution, sub_comm)

  if sub_comm.Get_rank()==0:
    ref_tree = Mio.read_tree(ref_file)
    PT.rm_nodes_from_name(gather_tree, ':CGNS#Distribution')
    assert PT.is_same_tree(ref_tree[2][1], gather_tree[2][1])

    
