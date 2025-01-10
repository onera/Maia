import pytest
import pytest_parallel
import numpy      as np

import maia.pytree      as PT

import maia
import maia.transfer.part_to_dist.tree_api as PTB


@pytest_parallel.mark.parallel(2)
class Test_IterativeData:

  dist_tree = PT.yaml.to_cgns_tree("""
  Base CGNSBase_t:
    Zone Zone_t:
  """)
  part_trees = [
    PT.yaml.to_cgns_tree("""
    Base CGNSBase_t:
      Zone.P0.N0 Zone_t:
        ZoneIterativeData ZoneIterativeData_t:
          FlowSolutionPointers DataArray_t ["FS0", "FS1"]:
      BaseIterativeData BaseIterativeData_t [2]:
        TimeValues DataArray_t [0., 1]:
    """), # Rank 0
    PT.yaml.to_cgns_tree("""
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
  dist_zone = PT.get_all_Zone_t(dist_tree)[0]
  # Change a BC into GC
  gc = PT.pop_node_from_path(dist_zone, 'ZoneBC/Zmin')
  PT.update_node(gc, label='GridConnectivity_t', value='zone')
  PT.new_child(dist_zone, 'ZoneGridConnectivity', 'ZoneGridConnectivity_t', children=[gc])

  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

  part_base = PT.get_all_CGNSBase_t(part_tree)[0]
  dist_base = PT.get_all_CGNSBase_t(dist_tree)[0]
  part_zone = PT.get_all_Zone_t(part_tree)[0]

  PT.new_Family('MyFamily', parent=part_base)
  PT.new_Family('MyOtherFamily', parent=part_base)
  for i in range(3):
    part_family_n = PT.new_Family(f'WALL_{i}', family_bc='BCWall', parent=part_base)
    PT.new_Family(f'WALL_{i}', family_bc='BCWall', parent=dist_base)
    PT.new_node('.Solver#BC', label='UserDefinedData_t', value=np.array([i,i+1,i+2]), children=[], parent=part_family_n)
    PT.new_node('.Solver#Property', label='UserDefinedData_t', value=np.array([i,i+1,i+2]), children=[], parent=part_family_n)
  for i, bc_n  in enumerate(PT.get_nodes_from_predicate(part_zone, lambda n : PT.get_label(n) in ['BC_t', 'GridConnectivity_t'])):
    PT.new_node('.Solver#BC', label='UserDefinedData_t', value=np.array([i,i+1,i+2]), children=[], parent=bc_n)
    PT.new_node('.Solver#Property', label='UserDefinedData_t', value=np.array([i,i+1,i+2]), children=[], parent=bc_n)

  # This one already exists on dist_tree, but it should be updated
  PT.new_UserDefinedData('.Solver#BC', value=np.array([42]), parent=PT.get_node_from_name(dist_tree, 'Zmax'))

  if missing_part_node:
    PT.rm_nodes_from_label(part_tree, 'ZoneBC_t')

  ud_predicates = [['CGNSBase_t', 'Family_t', lambda n : PT.get_name(n).startswith('.Solver#')],
                  'CGNSBase_t/Zone_t/ZoneBC_t/BC_t/.Solver#*',
                  'CGNSBase_t/Zone_t/*/Zmin/.Solver#Property',
                  'CGNSBase_t/MyFamily']
  for ud_predicate in ud_predicates:
    PTB.part_tree_to_dist_tree_copy(dist_tree, part_tree, ud_predicate, comm)

  if not missing_part_node:
    for dist_bc in PT.get_nodes_from_label(dist_tree, 'BC_t'):
      part_bc = PT.get_node_from_name_and_label(part_tree, PT.get_name(dist_bc), 'BC_t')
      for name in ['.Solver#Property', '.Solver#BC']:
        assert PT.is_same_node(PT.get_child_from_name(dist_bc, name), PT.get_child_from_name(part_bc, name))

  assert (PT.get_node_from_path(dist_tree, 'Base/zone/ZoneGridConnectivity/Zmin/.Solver#Property')[1] == [0,1,2]).all()

  assert PT.get_label(PT.get_child_from_name(dist_base, 'MyFamily')) == 'Family_t'
  assert PT.get_child_from_name(dist_base, 'MyOtherFamily') is None
  if not missing_part_node:
    assert PT.get_node_from_path(dist_base, 'zone/ZoneBC/Zmax/.Solver#BC')[1].size != 1

  uds = [PT.new_Descriptor('Descr1', 'Value1', parent=part_tree), PT.new_Descriptor('Descr2', 'Value2', parent=part_tree)]
  pred = [lambda n : PT.get_label(n) == 'Descriptor_t'] if missing_part_node else 'Descr*'
  #                                                     ^ just a way to test two different predicates
  PTB.part_tree_to_dist_tree_copy(dist_tree, part_tree, pred, comm)
  for dist_ud, part_ud in zip(PT.get_children_from_label(dist_tree, 'Descriptor_t'), uds):
    assert PT.is_same_node(dist_ud, part_ud)