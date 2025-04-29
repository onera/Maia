import maia
import pytest
import pytest_parallel
import maia.pytree as PT
from mpi4py import MPI
import maia.pytree.maia.check_tree as CT


def prepare_trees():
  comm = MPI.COMM_SELF
  dist_tree = maia.factory.generate_dist_block([4,2,2], 'S', comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
  full_tree = maia.factory.dist_to_full_tree(dist_tree, comm)
  dist_part_tree = PT.union(dist_tree, part_tree)
  return dist_tree, part_tree, full_tree, dist_part_tree

def test_check_contains_zone():
  node = PT.new_Zone()
  CT.check_contain_zones(node)
  
  node = PT.yaml.to_node("""
  BCa BC_t "BCFarfield":
    GridLocation GridLocation_t "FaceCenter":
    PointList IndexArray_t [[1, 2, 3, 4]]:
    BCDataSet BCDataSet_t:
      BCData BCData_t:
        Data DataArray_t [10., 20., 30., 40.]:
  """)
  
  with pytest.raises(ValueError):
    CT.check_contain_zones(node)

def test_check_cgns_dist_tree():
  dist_tree, part_tree, full_tree, dist_part_tree = prepare_trees()
  
  CT.check_cgns_dist_tree(dist_tree)
  
  with pytest.raises(ValueError, match="The provided CGNS tree is not a distributed tree."):
    CT.check_cgns_dist_tree(part_tree)
  with pytest.raises(ValueError, match="The provided CGNS tree is not a distributed tree."):
    CT.check_cgns_dist_tree(full_tree)
  with pytest.raises(ValueError, match="The provided CGNS tree is not a distributed tree."):
    CT.check_cgns_dist_tree(dist_part_tree)
        
def test_check_cgns_part_tree():
  dist_tree, part_tree, full_tree, dist_part_tree = prepare_trees()
  
  CT.check_cgns_part_tree(part_tree)
  
  with pytest.raises(ValueError, match="The provided CGNS tree is not a partitioned tree."):
    CT.check_cgns_part_tree(dist_tree)
  with pytest.raises(ValueError, match="The provided CGNS tree is not a partitioned tree."):
    CT.check_cgns_part_tree(full_tree)
  with pytest.raises(ValueError, match="The provided CGNS tree is not a partitioned tree."):
    CT.check_cgns_part_tree(dist_part_tree)

def test_check_cgns_full_tree():
  dist_tree, part_tree, full_tree, dist_part_tree = prepare_trees()

  CT.check_cgns_full_tree(full_tree)
  
  with pytest.raises(ValueError, match="The provided CGNS tree is not a full CGNS tree."):
    CT.check_cgns_full_tree(dist_tree)
  with pytest.raises(ValueError, match="The provided CGNS tree is not a full CGNS tree."):
    CT.check_cgns_full_tree(part_tree)
  with pytest.raises(ValueError, match="The provided CGNS tree is not a full CGNS tree."):
    CT.check_cgns_full_tree(dist_part_tree)

def test_check_cgns_dist_part_tree():
  dist_tree, part_tree, full_tree, dist_part_tree = prepare_trees()
  
  CT.check_cgns_dist_part_tree(dist_tree)
  CT.check_cgns_dist_part_tree(part_tree)
  
  with pytest.raises(ValueError, match="The provided CGNS tree is neither a distributed nor partitionned CGNS tree."):
    CT.check_cgns_dist_part_tree(full_tree)
  with pytest.raises(ValueError):
    CT.check_cgns_dist_part_tree(dist_part_tree)