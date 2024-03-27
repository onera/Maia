import numpy as np
# MAIA
import maia
import maia.pytree as PT
from maia.transfer.part_to_dist import tree_api as transfer
import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD
# Pytest
import pytest_parallel

@pytest_parallel.mark.parallel(1)
def test_recover_UDData(comm):
    dist_tree = maia.factory.generate_dist_block(3, 'Poly', comm)
    part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

    part_base = PT.get_all_CGNSBase_t(part_tree)[0]
    dist_base = PT.get_all_CGNSBase_t(dist_tree)[0]
    part_zone = PT.get_all_Zone_t(part_tree)[0]
    for i in [1,2,3]:
        part_family_n = PT.new_Family(f'WALL_{i}', family_bc='BCWall', parent=part_base)
        PT.new_Family(f'WALL_{i}', family_bc='BCWall', parent=dist_base)
        PT.new_node('.Solver#BC', label='UserDefinedData_t', value=np.array([i,i+1,i+2]), children=[], parent=part_family_n)
        PT.new_node('.Solver#Property', label='UserDefinedData_t', value=np.array([i,i+1,i+2]), children=[], parent=part_family_n)
    for i, bc_n  in enumerate(PT.get_nodes_from_predicates(part_zone, 'ZoneBC_t/BC_t')):
        PT.new_node('.Solver#BC', label='UserDefinedData_t', value=np.array([i,i+1,i+2]), children=[], parent=bc_n)
        PT.new_node('.Solver#Property', label='UserDefinedData_t', value=np.array([i,i+1,i+2]), children=[], parent=bc_n)
    ud_predicates = [['CGNSBase_t', 'Family_t', '.Solver#*'],
                    ['CGNSBase_t', 'Zone_t', 'ZoneBC_t', 'BC_t', '.Solver#*']]
    for ud_predicate in ud_predicates:
        transfer.recover_UDData_from_part_to_dist(dist_tree, part_tree, comm, ud_predicate=ud_predicate)

    for dist_ud, part_ud in zip(PT.get_nodes_from_name(dist_tree, '.Solver#*'), PT.get_nodes_from_name(part_tree, '.Solver#*')):
        assert PT.is_same_node(dist_ud, part_ud)
