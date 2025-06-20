import pytest
import pytest_parallel
import numpy as np

import maia
import maia.pytree as PT

import maia.transfer as transfer

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

  dist_base = PT.get_all_CGNSBase_t(dist_tree)[0]
  dist_zone = PT.get_all_Zone_t(dist_tree)[0]
  part_base = PT.get_all_CGNSBase_t(part_tree)[0]
  for i in range(3):
    dist_family_n = PT.new_Family(f'WALL_{i}', family_bc='BCWall', parent=dist_base)
    PT.new_Family(f'WALL_{i}', family_bc='BCWall', parent=part_base)
    PT.new_node('.Solver#BC', label='UserDefinedData_t', value=np.array([i,i+1,i+2]), children=[], parent=dist_family_n)
    PT.new_node('.Solver#Property', label='UserDefinedData_t', value=np.array([i,i+1,i+2]), children=[], parent=dist_family_n)
  for i, bc_n  in enumerate(PT.get_nodes_from_predicate(dist_zone, PT.pred.label_in(['BC_t', 'GridConnectivity_t']))):
    PT.new_node('.Solver#BC', label='UserDefinedData_t', value=np.array([i,i+1,i+2]), children=[], parent=bc_n)
    PT.new_node('.Solver#Property', label='UserDefinedData_t', value=np.array([i,i+1,i+2]), children=[], parent=bc_n)

  if missing_part_node:
    PT.rm_nodes_from_label(part_tree, 'ZoneBC_t')
  else:
  # This one already exists on part_tree, but it should be updated
    PT.new_UserDefinedData('.Solver#BC', value=np.array([42]), parent=PT.get_node_from_name(part_tree, 'Zmax'))

  ud_predicates = [['CGNSBase_t', 'Family_t', lambda n : PT.get_name(n).startswith('.Solver#')],
                  'CGNSBase_t/Zone_t/ZoneBC_t/BC_t/.Solver#*', 
                  'CGNSBase_t/Zone_t/*/GridConnectivity_t/.Solver#Property']
  for ud_predicate in ud_predicates:
    transfer.dist_tree_to_part_tree_copy(dist_tree, part_tree, ud_predicate, comm)
  
  if not missing_part_node:
    for dist_bc in PT.get_nodes_from_label(dist_tree, 'BC_t'):
      part_bc = PT.get_node_from_name_and_label(part_tree, PT.get_name(dist_bc), 'BC_t')
      for name in ['.Solver#Property', '.Solver#BC']:
        assert PT.is_same_node(PT.get_child_from_name(dist_bc, name), PT.get_child_from_name(part_bc, name))

  assert (PT.get_node_from_path(part_tree, 'Base/zone.P0.N0/ZoneGridConnectivity/Zmin.0/.Solver#Property')[1] == [5,6,7]).all()

  if not missing_part_node:
    assert PT.get_node_from_path(part_base, 'zone.P0.N0/ZoneBC/Zmax/.Solver#BC')[1].size != 1

  ud = PT.new_UserDefinedData('TopLevelNode', [1,2,3], parent=dist_tree)
  transfer.dist_tree_to_part_tree_copy(dist_tree, part_tree, 'TopLevelNode', comm)
  assert PT.is_same_tree(ud, PT.get_child_from_name(part_tree, 'TopLevelNode'))