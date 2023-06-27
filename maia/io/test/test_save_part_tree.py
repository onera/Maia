import pytest
import pytest_parallel
import mpi4py.MPI as MPI
import numpy      as np
from pathlib import Path

import maia
import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia.utils.test_utils as TU
from   maia.pytree.yaml    import parse_yaml_cgns
from   maia.utils.parallel import utils as par_utils

from maia.io import save_part_tree as SPT

@pytest_parallel.mark.parallel(4)
@pytest.mark.parametrize('single_file', [False, True])
def test_write_part_tree(single_file, comm):
  dtree = maia.factory.generate_dist_block(4, 'Poly', comm)
  tree  = maia.factory.partition_dist_tree(dtree, comm)

  expected_n_files = 1 + comm.Get_size() * int(not single_file)

  with TU.collective_tmp_dir(comm) as tmpdir:
    filename = Path(tmpdir) / 'out.hdf'
    SPT.save_part_tree(tree, str(filename), comm, single_file)
    comm.barrier()
    assert filename.exists()
    assert len(list(Path(tmpdir).glob('*'))) == expected_n_files
    
@pytest_parallel.mark.parallel(4)
@pytest.mark.parametrize('single_file', [False, True])
def test_read_part_tree(single_file, comm):

  # Prepare test (produce part_tree_file)
  dtree = maia.factory.generate_dist_block(4, 'Poly', comm)
  tree  = maia.factory.partition_dist_tree(dtree, comm)

  with TU.collective_tmp_dir(comm) as tmpdir:
    filename = Path(tmpdir) / 'out.hdf'
    SPT.save_part_tree(tree, str(filename), comm, single_file)
    comm.barrier()
    tree = SPT.read_part_tree(str(filename), comm)
    zones = PT.get_all_Zone_t(tree)
    assert len(zones) == 1
    assert PT.get_name(zones[0]) == f'zone.P{comm.Get_rank()}.N0'
    # Data should have been loaded
    assert PT.get_node_from_name(zones[0], 'CoordinateX')[1].size > 0
    
