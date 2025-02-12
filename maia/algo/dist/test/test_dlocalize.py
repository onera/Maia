import pytest
import pytest_parallel
import os
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia
from maia              import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.factory      import dcube_generator as DCG
from maia.factory      import partition_dist_tree

from maia.utils     import test_utils as TU
from maia.utils     import np_utils
from maia.algo.part import localize as PLOC

from maia.algo.dist import localize as LOC

@pytest_parallel.mark.parallel(2)
def test_localize(comm):
  src_tree = maia.factory.generate_dist_block(501, 'QUAD_4', comm)
  tgt_tree = maia.factory.generate_dist_block([101,101], 'S', comm)
  
  LOC.localize_points(src_tree, tgt_tree, 'CellCenter', comm)
  
  for node in PT.get_nodes_from_label(tgt_tree, 'DiscreteData_t'):
    PT.set_label(node, 'FlowSolution_t')
  

  maia.io.dist_tree_to_file(src_tree, 'src.cgns', comm)
  maia.io.dist_tree_to_file(tgt_tree, 'tgt.cgns', comm)

  if comm.rank == 0:
    PT.print_tree(tgt_tree)

  PT.rm_nodes_from_label(tgt_tree, 'FlowSolution_t')

  return
  psrc = maia.factory.partition_dist_tree(src_tree, comm)
  ptgt = maia.factory.partition_dist_tree(tgt_tree, comm)
  PLOC.localize_points(psrc, ptgt, 'CellCenter', comm)
  maia.transfer.part_tree_to_dist_tree_all(tgt_tree, ptgt, comm)
  # Expected :
  # tgt 1 2 3 4 5  6  7  8
  # src 1 3 7 9 19 21 25 27
  #
  # 1 2 4 6 8 14 10 17 ??