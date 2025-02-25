import pytest
import pytest_parallel
import numpy as np

import maia.pytree as PT

import maia

from maia.algo import geosearch as GS


def test_is_distributed():
  tree = PT.yaml.to_cgns_tree("""
  Zone Zone_t:
    :CGNS#Distribution UserDefinedData_t:
      Cell DataArray_t [0, 10, 100]:
        Vertex DataArray_t [0, 20, 50]:
  """)
  assert GS.is_distributed(tree)
  tree = PT.yaml.to_cgns_tree("""
  Zone Zone_t:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Cell DataArray_t [1,2,3,4,5,6]:
  """)
  assert not GS.is_distributed(tree)

  # This case could represent partitioned tree with 0 part -> must return False
  tree = PT.new_CGNSTree()
  assert not GS.is_distributed(tree)

@pytest_parallel.mark.parallel(2)
def test_exclusive(comm):
  dtree_src = maia.factory.generate_dist_block(5, 'Poly', comm)
  dtree_tgt = maia.factory.generate_dist_block(5, 'Poly', comm)

  ptree_src = maia.factory.partition_dist_tree(dtree_src, comm) 
  ptree_tgt = maia.factory.partition_dist_tree(dtree_tgt, comm) 

  with pytest.raises(ValueError):
    GS.localize_points(dtree_src, ptree_tgt, 'Vertex', comm)
  with pytest.raises(ValueError):
    GS.find_closest_points(ptree_src, dtree_tgt, 'CellCenter', comm)