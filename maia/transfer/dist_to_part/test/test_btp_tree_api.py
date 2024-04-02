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
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

  dist_base = PT.get_all_CGNSBase_t(dist_tree)[0]
  dist_zone = PT.get_all_Zone_t(dist_tree)[0]
  part_base = PT.get_all_CGNSBase_t(part_tree)[0]
  for i in range(3):
    dist_family_n = PT.new_Family(f'WALL_{i}', family_bc='BCWall', parent=dist_base)
    PT.new_Family(f'WALL_{i}', family_bc='BCWall', parent=part_base)
    PT.new_node('.Solver#BC', label='UserDefinedData_t', value=np.array([i,i+1,i+2]), children=[], parent=dist_family_n)
    PT.new_node('.Solver#Property', label='UserDefinedData_t', value=np.array([i,i+1,i+2]), children=[], parent=dist_family_n)
  for i, bc_n  in enumerate(PT.get_nodes_from_predicates(dist_zone, 'ZoneBC_t/BC_t')):
    PT.new_node('.Solver#BC', label='UserDefinedData_t', value=np.array([i,i+1,i+2]), children=[], parent=bc_n)
    PT.new_node('.Solver#Property', label='UserDefinedData_t', value=np.array([i,i+1,i+2]), children=[], parent=bc_n)

  if missing_part_node:
    PT.rm_nodes_from_label(part_tree, 'ZoneBC_t')

  ud_predicates = [['CGNSBase_t', 'Family_t', lambda n : PT.get_name(n).startswith('.Solver#')],
                  'CGNSBase_t/Zone_t/ZoneBC_t/BC_t/.Solver#*']
  for ud_predicate in ud_predicates:
    transfer.dist_tree_to_part_tree_copy(dist_tree, part_tree, ud_predicate, comm)
  
  for dist_ud, part_ud in zip(PT.get_nodes_from_name(dist_tree, '.Solver#*'), PT.get_nodes_from_name(part_tree, '.Solver#*')):
    assert PT.is_same_node(dist_ud, part_ud) # Nodes are matched in same order, so this comparison is OK
