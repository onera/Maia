import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np
import os

import maia.pytree        as PT

from maia.io          import file_to_dist_tree
from maia.utils       import test_utils as TU

from maia.algo.dist   import convert_elements_to_mixed

from maia.algo.dist.mixed_to_std_elements   import convert_mixed_to_elements, \
                                                   collect_pl_nodes

@mark_mpi_test([1,2,3])
def test_collect_pl_nodes(sub_comm):
    yaml_path = os.path.join(TU.mesh_dir, 'cube_4.yaml')
    dist_tree = file_to_dist_tree(yaml_path, sub_comm)

    zone = PT.get_node_from_label(dist_tree, 'Zone_t')

    pointlist_nodes = collect_pl_nodes(zone)
    assert len(pointlist_nodes) == 7
    for pointlist_node in pointlist_nodes:
        assert PT.get_name(pointlist_node)  == 'PointList'
        assert PT.get_label(pointlist_node) == 'IndexArray_t'
    
    assert PT.get_nodes_from_label(zone, 'PointRange') == []

    pointlist_nodes = collect_pl_nodes(zone, filter_loc = 'Vertex')
    assert len(pointlist_nodes) == 1
    for pointlist_node in pointlist_nodes:
        assert PT.get_name(pointlist_node)  == 'PointList'
        assert PT.get_label(pointlist_node) == 'IndexArray_t'

    pointlist_nodes = collect_pl_nodes(zone, filter_loc = ['Vertex','FaceCenter'])
    assert len(pointlist_nodes) == 7
    for pointlist_node in pointlist_nodes:
       assert PT.get_name(pointlist_node)  == 'PointList'
       assert PT.get_label(pointlist_node) == 'IndexArray_t'

    pointlist_nodes = collect_pl_nodes(zone, filter_loc = ['FaceCenter'])
    assert len(pointlist_nodes) == 6
    for pointlist_node in pointlist_nodes:
        assert PT.get_name(pointlist_node)  == 'PointList'
        assert PT.get_label(pointlist_node) == 'IndexArray_t'

@mark_mpi_test([1,2,3,7])
def test_convert_mixed_to_elements(sub_comm):
    rank = sub_comm.Get_rank()
    size = sub_comm.Get_size()
    
    yaml_path = os.path.join(TU.mesh_dir, 'hex_prism_pyra_tet.yaml')
    dist_tree = file_to_dist_tree(yaml_path, sub_comm)
    
    # Assume this function is already tested
    convert_elements_to_mixed(dist_tree, sub_comm)
    
    convert_mixed_to_elements(dist_tree, sub_comm)
    
    base = PT.get_node_from_label(dist_tree,'CGNSBase_t')  
    assert PT.get_name(base) == 'Base'
    assert np.all(PT.get_value(base) == [3,3])
    
    zone = PT.get_node_from_label(base, 'Zone_t') 
    assert PT.get_name(zone) == 'Zone'
    assert np.all(PT.get_value(zone) == [[11,4,0]])
    
    elements_nodes = PT.get_nodes_from_label(zone, 'Elements_t')
    assert len(elements_nodes) == 6
    
    hexa_node = PT.get_node_from_name(zone, 'Hexa_8')
    hexa_er_node = PT.get_node_from_name(hexa_node, 'ElementRange')
    hexa_ec_node = PT.get_node_from_name(hexa_node, 'ElementConnectivity')
    assert PT.get_value(hexa_node)[0] == 17
    assert np.all(PT.get_value(hexa_er_node) == [1,1])
    
    penta_node = PT.get_node_from_name(zone, 'Penta_6')
    penta_er_node = PT.get_node_from_name(penta_node, 'ElementRange')
    penta_ec_node = PT.get_node_from_name(penta_node, 'ElementConnectivity')
    assert PT.get_value(penta_node)[0] == 14
    assert np.all(PT.get_value(penta_er_node) == [2,2])
    
    pyra_node = PT.get_node_from_name(zone, 'Pyra_5')
    pyra_er_node = PT.get_node_from_name(pyra_node, 'ElementRange')
    pyra_ec_node = PT.get_node_from_name(pyra_node, 'ElementConnectivity')
    assert PT.get_value(pyra_node)[0] == 12
    assert np.all(PT.get_value(pyra_er_node) == [3,3])
    
    tetra_node = PT.get_node_from_name(zone, 'Tetra_4')
    tetra_er_node = PT.get_node_from_name(tetra_node, 'ElementRange')
    tetra_ec_node = PT.get_node_from_name(tetra_node, 'ElementConnectivity')
    assert PT.get_value(tetra_node)[0] == 10
    assert np.all(PT.get_value(tetra_er_node) == [4,4])
    
    quad_node = PT.get_node_from_name(zone, 'Quad_4')
    quad_er_node = PT.get_node_from_name(quad_node, 'ElementRange')
    quad_ec_node = PT.get_node_from_name(quad_node, 'ElementConnectivity')
    assert PT.get_value(quad_node)[0] == 7
    assert np.all(PT.get_value(quad_er_node) == [5,10])
    
    tri_node = PT.get_node_from_name(zone, 'Tri_3')
    tri_er_node = PT.get_node_from_name(tri_node, 'ElementRange')
    tri_ec_node = PT.get_node_from_name(tri_node, 'ElementConnectivity')
    assert PT.get_value(tri_node)[0] == 5
    assert np.all(PT.get_value(tri_er_node) == [11,16])
    
    if rank == 0:
        expected_hexa_ec  = [ 1,  2,  5,  4,  6,  7, 10,  9]
        expected_penta_ec = [ 2,  3,  5,  7,  8, 10]
        expected_pyra_ec  = [ 6,  7, 10,  9, 11]
        expected_tetra_ec = [ 7,  8, 10, 11]
    else:
        expected_hexa_ec  = []
        expected_penta_ec = []
        expected_pyra_ec  = []
        expected_tetra_ec = []
        
    if size == 1:
        expected_quad_ec  = [ 1,  6,  9,  4,  3,  5, 10,  8,  1,  2,  7,  6,  2,  3,  8,  7,  4,
                              9, 10,  5,  1,  4,  5,  2]
        expected_tri_ec   = [ 6, 11,  9,  8, 10, 11,  6,  7, 11,  7,  8, 11,  2,  5,  3,  9, 11,
                             10]
    elif size == 2:
        if rank == 0:
            expected_quad_ec  = [ 1,  6,  9,  4,  3,  5, 10,  8,  1,  2,  7,  6]
            expected_tri_ec   = [ 6, 11,  9,  8, 10, 11,  6,  7, 11]
        elif rank == 1:
            expected_quad_ec  = [ 2,  3,  8,  7,  4,  9, 10,  5,  1,  4,  5,  2]
            expected_tri_ec   = [ 7,  8, 11,  2,  5,  3,  9, 11, 10]
    elif size == 3:
        if rank == 0:
            expected_quad_ec  = [ 1,  6,  9,  4,  3,  5, 10,  8]
            expected_tri_ec   = [ 6, 11,  9,  8, 10, 11]
        elif rank == 1:
            expected_quad_ec  = [1, 2, 7, 6, 2, 3, 8, 7]
            expected_tri_ec   = [ 6,  7, 11,  7,  8, 11]
        elif rank == 2:
            expected_quad_ec  = [ 4,  9, 10,  5,  1,  4,  5,  2]
            expected_tri_ec   = [ 2,  5,  3,  9, 11, 10]
    elif size > 6:
        expected_quad_ec_full  = [ 1,  6,  9,  4,  3,  5, 10,  8,  1,  2,  7,  6,  2,  3,  8,  7,  4,
                                   9, 10,  5,  1,  4,  5,  2]
        expected_tri_ec_full   = [ 6, 11,  9,  8, 10, 11,  6,  7, 11,  7,  8, 11,  2,  5,  3,  9, 11,
                                  10]
        if rank < 6:
            expected_quad_ec  = expected_quad_ec_full[4*rank:4*(rank+1)]
            expected_tri_ec   = expected_tri_ec_full[3*rank:3*(rank+1)]
        elif rank > 5:
            expected_quad_ec  = []
            expected_tri_ec   = []
    
    assert np.all(PT.get_value(hexa_ec_node) == expected_hexa_ec)
    assert np.all(PT.get_value(penta_ec_node) == expected_penta_ec)
    assert np.all(PT.get_value(pyra_ec_node) == expected_pyra_ec)
    assert np.all(PT.get_value(tetra_ec_node) == expected_tetra_ec)
    assert np.all(PT.get_value(quad_ec_node) == expected_quad_ec)
    assert np.all(PT.get_value(tri_ec_node) == expected_tri_ec)
    
    bc_xmin = PT.get_nodes_from_predicates(zone, 'ZoneBC_t/BC_t/Xmin')[0]
    pointlist_xmin = PT.get_value(PT.get_node_from_name(bc_xmin,'PointList'))
    
    bc_xmax = PT.get_nodes_from_predicates(zone, 'ZoneBC_t/BC_t/Xmax')[0]
    pointlist_xmax = PT.get_value(PT.get_node_from_name(bc_xmax,'PointList'))
    
    bc_ymin = PT.get_nodes_from_predicates(zone, 'ZoneBC_t/BC_t/Ymin')[0]
    pointlist_ymin = PT.get_value(PT.get_node_from_name(bc_ymin,'PointList'))
    
    bc_ymax = PT.get_nodes_from_predicates(zone, 'ZoneBC_t/BC_t/Ymax')[0]
    pointlist_ymax = PT.get_value(PT.get_node_from_name(bc_ymax,'PointList'))
    
    bc_zmin = PT.get_nodes_from_predicates(zone, 'ZoneBC_t/BC_t/Zmin')[0]
    pointlist_zmin = PT.get_value(PT.get_node_from_name(bc_zmin,'PointList'))
    
    bc_zmax = PT.get_nodes_from_predicates(zone, 'ZoneBC_t/BC_t/Zmax')[0]
    pointlist_zmax = PT.get_value(PT.get_node_from_name(bc_zmax,'PointList'))
    
    if size == 1:
        expected_bc_xmin = [[ 5, 11]]
        expected_bc_xmax = [[ 6, 12]]
        expected_bc_zmin = [[10, 15]]
    else:
        if rank == 0:
            expected_bc_xmin = [[ 5]]
            expected_bc_xmax = [[ 6]]
            expected_bc_zmin = [[10]]
        elif rank == 1:
            expected_bc_xmin = [[11]]
            expected_bc_xmax = [[12]]
            expected_bc_zmin = [[15]]
        else:
            expected_bc_xmin = []
            expected_bc_xmax = []
            expected_bc_zmin = []
    
    if size == 1:
        expected_bc_ymin = [[ 7,  8, 13, 14]]
    elif size == 2:
        if rank == 0:
            expected_bc_ymin = [[7, 8]]
        elif rank == 1:
            expected_bc_ymin = [[13, 14]]
    elif size == 3:
        if rank == 0:
            expected_bc_ymin = [[7, 8]]
        elif rank == 1:
            expected_bc_ymin = [[13]]
        elif rank == 2:
            expected_bc_ymin = [[14]]
    else:
        if rank == 0:
            expected_bc_ymin = [[7]]
        elif rank == 1:
            expected_bc_ymin = [[8]]
        elif rank == 2:
            expected_bc_ymin = [[13]]
        elif rank == 3:
            expected_bc_ymin = [[14]]
        else:
            expected_bc_ymin = []
    
    if rank == 0:
        expected_bc_ymax = [[9]]
        expected_bc_zmax = [[16]]
    else:
        expected_bc_ymax = []
        expected_bc_zmax = []
    
    assert np.all(pointlist_xmin == expected_bc_xmin)
    assert np.all(pointlist_xmax == expected_bc_xmax)
    assert np.all(pointlist_ymin == expected_bc_ymin)
    assert np.all(pointlist_ymax == expected_bc_ymax)
    assert np.all(pointlist_zmin == expected_bc_zmin)
    assert np.all(pointlist_zmax == expected_bc_zmax)    
