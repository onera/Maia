import pytest
import pytest_parallel
import numpy      as np

import maia.pytree      as PT

import maia
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

@pytest.mark.parametrize('missing_part_node', [False, True])
@pytest_parallel.mark.parallel(1)
def test_recover_UDData(missing_part_node, comm):
  dist_tree = maia.factory.generate_dist_block(3, 'Poly', comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

  part_base = PT.get_all_CGNSBase_t(part_tree)[0]
  dist_base = PT.get_all_CGNSBase_t(dist_tree)[0]
  part_zone = PT.get_all_Zone_t(part_tree)[0]
  for i in range(3):
    part_family_n = PT.new_Family(f'WALL_{i}', family_bc='BCWall', parent=part_base)
    PT.new_Family(f'WALL_{i}', family_bc='BCWall', parent=dist_base)
    PT.new_node('.Solver#BC', label='UserDefinedData_t', value=np.array([i,i+1,i+2]), children=[], parent=part_family_n)
    PT.new_node('.Solver#Property', label='UserDefinedData_t', value=np.array([i,i+1,i+2]), children=[], parent=part_family_n)
  for i, bc_n  in enumerate(PT.get_nodes_from_predicates(part_zone, 'ZoneBC_t/BC_t')):
    PT.new_node('.Solver#BC', label='UserDefinedData_t', value=np.array([i,i+1,i+2]), children=[], parent=bc_n)
    PT.new_node('.Solver#Property', label='UserDefinedData_t', value=np.array([i,i+1,i+2]), children=[], parent=bc_n)

  if missing_part_node:
    PT.rm_nodes_from_label(part_tree, 'ZoneBC_t')

  ud_predicates = [['CGNSBase_t', 'Family_t', lambda n : PT.get_name(n).startswith('.Solver#')],
                  'CGNSBase_t/Zone_t/ZoneBC_t/BC_t/.Solver#*']
  for ud_predicate in ud_predicates:
    PTB.part_tree_to_dist_tree_copy(dist_tree, part_tree, ud_predicate, comm)

  for dist_ud, part_ud in zip(PT.get_nodes_from_name(dist_tree, '.Solver#*'), PT.get_nodes_from_name(part_tree, '.Solver#*')):
    assert PT.is_same_node(dist_ud, part_ud) # Nodes are matched in same order, so this comparison is OK
