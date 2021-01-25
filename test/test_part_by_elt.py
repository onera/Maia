import pytest
from maia.cgns_io import save_part_tree as SPT

from mpi4py import MPI

from maia.partitioning.parallel_tree import load_partitioned_tree

@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("sub_comm", [2], indirect=['sub_comm'])
def test_load_partitioned_tree(tmpdir,sub_comm):
  if(sub_comm == MPI.COMM_NULL):
    return
  input_file = '/scratchm/bberthou/cases/CODA_tests/cube/data/in/cube.cgns'

  parallel_tree = load_partitioned_tree(input_file,sub_comm)

  SPT.save_part_tree(parallel_tree.part_tree, str(tmpdir)+'/part_tree', sub_comm)
