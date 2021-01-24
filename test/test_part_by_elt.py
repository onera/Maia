from maia.cgns_io import save_part_tree as SPT

from mpi4py import MPI
comm = MPI.COMM_WORLD

from maia.partitioning.parallel_tree import load_partitioned_tree

def test_load_partitioned_tree(tmpdir):
  input_file = '/scratchm/bberthou/cases/CODA_tests/cube/data/in/cube.cgns'

  parallel_tree = load_partitioned_tree(input_file,comm)

  print(tmpdir)
  SPT.save_part_tree(parallel_tree.part_tree, tmpdir+'part_tree', comm)
