import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np
import os
import copy

import maia.pytree        as PT

from maia.io          import file_to_dist_tree
from maia.utils       import test_utils as TU

from maia.algo.dist   import convert_elements_to_mixed

@mark_mpi_test([1,2,3,17])
def test_convert_mixed_to_elements(sub_comm):
    rank = sub_comm.Get_rank()
    size = sub_comm.Get_size()
    
    yaml_path = os.path.join(TU.sample_mesh_dir, 'hex_prism_pyra_tet.yaml')
    dist_tree = file_to_dist_tree(yaml_path, sub_comm)
    
    ref_dist_tree = copy.deepcopy(dist_tree)
    
    convert_elements_to_mixed(dist_tree, comm)
    
    ref_base = PT.get_node_from_label(ref_dist_tree,'CGNSBase_t')
    base = PT.get_node_from_label(dist_tree,'CGNSBase_t')  
    assert PT.get_name(base) == PT.get_name(ref_base)
    assert np.all(PT.get_value(base) == PT.get_value(ref_base))
    
    ref_zone = PT.get_node_from_label(ref_base, 'Zone_t')
    zone = PT.get_node_from_label(base, 'Zone_t') 
    assert PT.get_name(zone) == PT.get_name(ref_zone)
    assert np.all(PT.get_value(zone) == PT.get_value(ref_zone))
    
    mixed_nodes = PT.get_nodes_from_label(zone, 'Elements_t')
    assert len(mixed_nodes) == 1
    
    mixed_node = mixed_nodes[0]
    assert PT.get_value(mixed_node)[0] == 20
    
    mixed_er_node = PT.get_node_from_name(mixed_node, 'ElementRange')
    assert np.all(PT.get_value(mixed_er_node) == [1,16])
    
    mixed_eso_node = PT.get_node_from_name(mixed_node, 'ElementStartOffset')
    mixed_ec_node = PT.get_node_from_name(mixed_node, 'ElementConnectivity')
    assert mixed_eso_node is not None
    assert mixed_ec_node is not None
    
    if size == 1:
        expected_mixed_eso = [ 0,  5, 10, 15, 20, 25, 30, 34, 38, 42, 46, 50, 54, 63, 70, 76, 81]
        expected_mixed_ec  = [ 7,  1,  6,  9,  4,  7,  3,  5, 10,  8,  7,  1,  2,  7,  6,  7,  2,
                               3,  8,  7,  7,  4,  9, 10,  5,  7,  1,  4,  5,  2,  5,  6, 11,  9,
                               5,  8, 10, 11,  5,  6,  7, 11,  5,  7,  8, 11,  5,  2,  5,  3,  5,
                               9, 11, 10, 17,  1,  2,  5,  4,  6,  7, 10,  9, 14,  2,  3,  5,  7,
                               8, 10, 12,  6,  7, 10,  9, 11, 10,  7,  8, 10, 11]
    elif size == 2:
        if rank == 0:
            expected_mixed_eso = [ 0,  5, 10, 15, 20, 25, 30, 34, 38]
            expected_mixed_ec  = [ 7,  1,  6,  9,  4,  7,  3,  5, 10,  8,  7,  1,  2,  7,  6,  7,  2,
                                   3,  8,  7,  7,  4,  9, 10,  5,  7,  1,  4,  5,  2,  5,  6, 11,  9,
                                   5,  8, 10, 11]
        elif rank ==1:
            expected_mixed_eso = [38, 42, 46, 50, 54, 63, 70, 76, 81]
            expected_mixed_ec  = [ 5,  6,  7, 11,  5,  7,  8, 11,  5,  2,  5,  3,  5,  9, 11, 10, 17,
                                   1,  2,  5,  4,  6,  7, 10,  9, 14,  2,  3,  5,  7,  8, 10, 12,  6,
                                   7, 10,  9, 11, 10,  7,  8, 10, 11]
    elif size == 3:
        if rank == 0:
            expected_mixed_eso = [ 0,  5, 10, 15, 20, 25, 30]
            expected_mixed_ec  = [ 7,  1,  6,  9,  4,  7,  3,  5, 10,  8,  7,  1,  2,  7,  6,  7,  2,
                                   3,  8,  7,  7,  4,  9, 10,  5,  7,  1,  4,  5,  2]
        elif rank ==1:
            expected_mixed_eso = [30, 34, 38, 42, 46, 50]
            expected_mixed_ec  = [ 5,  6, 11,  9,  5,  8, 10, 11,  5,  6,  7, 11,  5,  7,  8, 11,  5,
                                   2,  5,  3]
        elif rank ==2:
            expected_mixed_eso = [50, 54, 63, 70, 76, 81]
            expected_mixed_ec  = [ 5,  9, 11, 10, 17,  1,  2,  5,  4,  6,  7, 10,  9, 14,  2,  3,  5,
                                   7,  8, 10, 12,  6,  7, 10,  9, 11, 10,  7,  8, 10, 11]
    elif size > 16:
        expected_mixed_eso_full = [ 0,  5, 10, 15, 20, 25, 30, 34, 38, 42, 46, 50, 54, 63, 70, 76, 81]
        expected_mixed_ec_full  = [ 7,  1,  6,  9,  4,  7,  3,  5, 10,  8,  7,  1,  2,  7,  6,  7,  2,
                                    3,  8,  7,  7,  4,  9, 10,  5,  7,  1,  4,  5,  2,  5,  6, 11,  9,
                                    5,  8, 10, 11,  5,  6,  7, 11,  5,  7,  8, 11,  5,  2,  5,  3,  5,
                                    9, 11, 10, 17,  1,  2,  5,  4,  6,  7, 10,  9, 14,  2,  3,  5,  7,
                                    8, 10, 12,  6,  7, 10,  9, 11, 10,  7,  8, 10, 11]
        if rank < 16:
            expected_mixed_eso = expected_mixed_eso_full[rank:rank+2]
            expected_mixed_ec =  expected_mixed_ec_full[expected_mixed_eso[0]:expected_mixed_eso[1]]
        else:
            expected_mixed_eso = [81]
            expected_mixed_ec =  []
    
    assert np.all(PT.get_value(mixed_eso_node) == expected_mixed_eso)
    assert np.all(PT.get_value(mixed_ec_node) == expected_mixed_ec)
    
    for ref_bc in PT.get_nodes_from_label(ref_zone, 'BC_t'):
        bc = PT.get_node_from_name(zone, PT.get_name(ref_bc))
        ref_pointlist = PT.get_value(PT.get_node_from_name(ref_bc,'PointList'))
        pointlist = PT.get_value(PT.get_node_from_name(bc,'PointList'))
        assert np.all(pointlist == ref_pointlist)    
