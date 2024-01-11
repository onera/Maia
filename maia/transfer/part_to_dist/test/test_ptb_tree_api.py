import pytest
import pytest_parallel
import numpy      as np

import maia.pytree      as PT

import maia.transfer.part_to_dist.tree_api as PTB
from   maia.pytree.yaml   import parse_yaml_cgns


@pytest_parallel.mark.parallel(2)
class Test_IterativeData:

  dist_tree = parse_yaml_cgns.to_cgns_tree("""
  Base CGNSBase_t:
    Zone Zone_t:
  """)
  part_trees = [
    parse_yaml_cgns.to_cgns_tree("""
    Base CGNSBase_t:
      Zone.P0.N0 Zone_t:
        ZoneIterativeData ZoneIterativeData_t:
          FlowSolutionPointers DataArray_t ["FS0", "FS1"]:
      BaseIterativeData BaseIterativeData_t [2]:
        TimeValues DataArray_t [0., 1]:
    """), # Rank 0
    parse_yaml_cgns.to_cgns_tree("""
    Base CGNSBase_t:
      Zone.P1.N0 Zone_t:
        ZoneIterativeData ZoneIterativeData_t:
          FlowSolutionPointers DataArray_t ["FS0", "FS1"]:
      BaseIterativeData BaseIterativeData_t [2]:
        TimeValues DataArray_t [0., 1]:
    """) # Rank 1
  ]
  def test_simple(self, comm):
    dist_tree = PT.deep_copy(self.dist_tree)
    part_tree = PT.deep_copy(self.part_trees[comm.rank])
    PTB.part_tree_to_dist_tree_all(dist_tree, part_tree, comm)
    assert (PT.get_node_from_name(dist_tree, 'TimeValues')[1] == [0., 1]).all()
    assert PT.get_value(PT.get_node_from_path(dist_tree, 'Base/Zone/ZoneIterativeData/FlowSolutionPointers')) \
           == ["FS0", "FS1"]
  def test_already_present(self, comm):
    dist_tree = PT.deep_copy(self.dist_tree)
    part_tree = PT.deep_copy(self.part_trees[comm.rank])
    PT.new_BaseIterativeData(time_values=[0., 1], parent=PT.get_all_CGNSBase_t(dist_tree)[0])
    zid = PT.new_node('ZoneIterativeData', 'ZoneIterativeData_t',  parent=PT.get_all_Zone_t(dist_tree)[0])
    PT.new_DataArray('FlowSolutionPointers', ["FS0", "FS1"], parent=zid)
    # Should not crash if IterativeData already exists on dist_tree
    PTB.part_tree_to_dist_tree_all(dist_tree, part_tree, comm)
    assert (PT.get_node_from_name(dist_tree, 'TimeValues')[1] == [0., 1]).all()
