import pytest
from pytest_mpi_check._decorator import mark_mpi_test
from maia.cgns_io import save_part_tree as SPT

from mpi4py import MPI

from maia import parallel_tree 

@mark_mpi_test(1)
def test_load_partitioned_tree(tmpdir,sub_comm):
  #input_file = '/scratchm/bberthou/cases/CODA_tests/cube/data/in/cube.cgns'
  input_file = '/scratchm/bberthou/cases/CODA_tests/cube/data/in/cube_4.cgns'

  par_tree = parallel_tree.load(input_file,sub_comm)

  SPT.save_part_tree(par_tree.part_tree, str(tmpdir)+'/part_tree', sub_comm)
