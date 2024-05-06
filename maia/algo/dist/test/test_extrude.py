import pytest
import pytest_parallel
import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia import npy_pdm_gnum_dtype as pdm_dtype


@pytest_parallel.mark.parallel([1,2])
def test_nodes_duplication(comm):

    # Prepare test
    extrusion_vector = [1., 2., 3.]
    dist_tree = maia.factory.generate_dist_block(11, 'TRI_3', comm)
    zone = PT.get_all_Zone_t(dist_tree)[0]
    coords_2d = PT.Zone.coordinates(zone)
    distrib_vtx_2d = MT.getDistribution(zone, 'Vertex')[1]

    # Run test
    maia.algo.dist.extrude._nodes_duplication(zone, extrusion_vector, comm)

    # Verification
    assert MT.getDistribution(zone, 'Vertex')[1][2] == 2*distrib_vtx_2d[2]
    coords = PT.Zone.coordinates(zone)
    if comm.size==1:
        for i in range(3):
            assert (coords[i][0:len(coords_2d[i])] == coords_2d[i]).all()
            assert (coords[i][len(coords_2d[i]):len(coords[i])] == coords_2d[i]+extrusion_vector[i]).all()
    elif comm.size==2:
        for i in range(3):
            if comm.rank == 0:
                assert (coords[i][0:len(coords_2d[i]//2)] == coords_2d[i][0:len(coords_2d[i]//2)]).all()
            elif comm.rank == 1:
                assert (coords[i][len(coords_2d[i]//2):len(coords_2d[i])] == coords_2d[i][len(coords_2d[i]//2):len(coords_2d[i])]+extrusion_vector[i]).all()

@pytest_parallel.mark.parallel([1,2])
def test_determine_mesh_orientation(comm):

    # Prepare test
    extrusion_vector = np.array([1., 2., 3.])
    dist_tree = maia.factory.generate_dist_block(11, 'TRI_3', comm)
    zone = PT.get_all_Zone_t(dist_tree)[0]

    # Run test and verification
    assert maia.algo.dist.extrude._determine_mesh_orientation(zone, extrusion_vector, comm)
    assert not(maia.algo.dist.extrude._determine_mesh_orientation(zone, -extrusion_vector, comm))

def test_reorder_ngon_ec():

    # Prepare test
    ngon_n = PT.new_NGonElements(erange=[1, 1], eso=[7, 11, 14], ec=[1,2,3,4, 5,6,7])
    MT.newDistribution({'ElementConnectivity': [7, 10, 25]}, parent=ngon_n)

    # Run test
    maia.algo.dist.extrude._reorder_ngon_ec(ngon_n)

    # Verification
    ngon_ec = PT.get_child_from_name(ngon_n, 'ElementConnectivity')[1]
    assert (ngon_ec == [1, 4, 3, 2, 5, 7, 6]).all()

@pytest_parallel.mark.parallel([1,2])
@pytest.mark.parametrize("align", [True, False])
def test_ngon_duplication(align, comm):

    # Prepare test
    dist_tree = maia.factory.generate_dist_block(11, 'TRI_3', comm)
    maia.algo.dist.convert_elements_to_ngon(dist_tree, comm)
    zone = PT.get_all_Zone_t(dist_tree)[0]
    n_cell_2d = PT.Zone.n_cell(zone)
    n_vtx_2d  = PT.Zone.n_vtx(zone)
    ngon_n = PT.deep_copy(PT.Zone.NGonNode(zone))
    ngon_ec = PT.get_child_from_name(ngon_n, "ElementConnectivity")

    # Run test
    maia.algo.dist.extrude._ngon_duplication(zone, comm, align)

    # Verification
    is_ngon = lambda n: PT.get_label(n) == "Elements_t" and PT.Element.CGNSName(n) == 'NGON_n'
    assert len(PT.get_children_from_predicate(zone, is_ngon)) == 2
    old_ngon_n = PT.get_child_from_name(zone, PT.get_name(ngon_n))
    old_ngon_er  = PT.get_child_from_name(old_ngon_n, "ElementRange")
    old_ngon_eso = PT.get_child_from_name(old_ngon_n, "ElementStartOffset")
    old_ngon_ec  = PT.get_child_from_name(old_ngon_n, "ElementConnectivity")
    old_ngon_pe  = PT.get_child_from_name(old_ngon_n, "ParentElements")
    new_ngon_n = PT.get_child_from_name(zone, f'{PT.get_name(ngon_n)}_bis')
    new_ngon_er  = PT.get_child_from_name(new_ngon_n, "ElementRange")
    new_ngon_eso = PT.get_child_from_name(new_ngon_n, "ElementStartOffset")
    new_ngon_ec  = PT.get_child_from_name(new_ngon_n, "ElementConnectivity")
    new_ngon_pe  = PT.get_child_from_name(new_ngon_n, "ParentElements")
    assert (new_ngon_er[1] == old_ngon_er[1]+n_cell_2d).all()
    assert (new_ngon_eso[1] == old_ngon_eso[1]).all()
    assert old_ngon_pe is not None
    assert new_ngon_pe is not None
    assert (new_ngon_pe[1] == old_ngon_pe[1]).all()
    if align:
        assert (new_ngon_ec[1] == ngon_ec[1]+n_vtx_2d).all()
    else:
        assert (old_ngon_ec[1] == ngon_ec[1]).all()

@pytest.mark.parametrize("align", [True, False])
def test_extrude_bar_to_ngon(align):

    # Prepare test
    n_vtx  = 25
    n_cell = 37
    bar = PT.new_Elements('EdgeElements', type='BAR_2', erange=[1,17], econn=[1,2, 2,3, 3,4, 4,1, 1,3], 
                          pe=[[6, 6, 7, 7, 6], [0, 0, 0, 0, 7]])
    MT.newDistribution({'Element': [5, 10, 17]}, parent=bar)

    # Run test
    maia.algo.dist.extrude._extrude_bar_to_ngon(bar, n_vtx, n_cell, align)

    # Verification
    assert (bar[1] == [22, 0]).all()
    assert (PT.get_child_from_name(bar, 'ElementRange')[1] == [1, 17]).all()
    bar_ec  = PT.get_child_from_name(bar, 'ElementConnectivity')
    assert len(bar_ec[1]) == 5*4
    if align:
        assert (bar_ec[1] == [2,1,1+n_vtx,2+n_vtx, 3,2,2+n_vtx,3+n_vtx,
                              4,3,3+n_vtx,4+n_vtx, 1,4,4+n_vtx,1+n_vtx,
                              3,1,1+n_vtx,3+n_vtx]).all()
    else:
        assert (bar_ec[1] == [1,2,2+n_vtx,1+n_vtx, 2,3,3+n_vtx,2+n_vtx,
                              3,4,4+n_vtx,3+n_vtx, 4,1,1+n_vtx,4+n_vtx,
                              1,3,3+n_vtx,1+n_vtx]).all()
    bar_eso = PT.get_child_from_name(bar, 'ElementStartOffset')
    assert bar_eso is not None
    assert len(bar_eso[1]) == 5+1
    assert (bar_eso[1] == [20, 24, 28, 32, 36, 40]).all()

@pytest_parallel.mark.parallel([1,2])
def test_merge_ngons(comm):

    # Prepare test
    dist_tree = maia.factory.generate_dist_block(11, 'HEXA_8', comm)
    maia.algo.dist.convert_elements_to_ngon(dist_tree, comm)
    zone = PT.get_all_Zone_t(dist_tree)[0]
    PT.rm_node_from_path(zone, 'NFaceElements')
    old_ngon_n = PT.Zone.NGonNode(zone)
    old_ngon_bis_n = PT.deep_copy(old_ngon_n)
    PT.set_name(old_ngon_bis_n, f'{old_ngon_n[0]}_bis')
    old_ngon_er     = PT.get_child_from_name(old_ngon_n,     'ElementRange')
    old_ngon_bis_er = PT.get_child_from_name(old_ngon_bis_n, 'ElementRange')
    PT.set_value(old_ngon_bis_er, old_ngon_er[1]+old_ngon_er[1][1])
    PT.add_child(zone, old_ngon_bis_n)

    # Run test
    maia.algo.dist.extrude._merge_ngons(zone, comm)

    # Verification
    is_ngon = lambda n: PT.get_label(n) == "Elements_t" and PT.Element.CGNSName(n) == 'NGON_n'
    assert len(PT.get_children_from_predicate(zone, is_ngon)) == 1
    new_ngon_n = PT.Zone.NGonNode(zone)
    assert PT.get_name(new_ngon_n) == 'NGonElements'
    new_ngon_er = PT.get_child_from_name(new_ngon_n, 'ElementRange')
    assert (new_ngon_er[1] == [old_ngon_er[1][0], old_ngon_bis_er[1][1]]).all()
    old_ngon_eso     = PT.get_child_from_name(old_ngon_n,     'ElementStartOffset')[1]
    old_ngon_bis_eso = PT.get_child_from_name(old_ngon_bis_n, 'ElementStartOffset')[1]
    new_ngon_eso     = PT.get_child_from_name(new_ngon_n,     'ElementStartOffset')[1]
    if comm.size==1:
        assert (new_ngon_eso[0:len(old_ngon_eso)]    == old_ngon_eso).all()
        assert (new_ngon_eso[len(old_ngon_eso)-2:-1] == old_ngon_bis_eso+old_ngon_eso[-2]).all()
    elif comm.size==2:
        if comm.rank == 0:
            assert (new_ngon_eso[0:len(old_ngon_eso)//2] == old_ngon_eso[0:len(old_ngon_eso)//2]).all()
        elif comm.rank == 1:
            assert (new_ngon_eso[len(new_ngon_eso//2):-1] == old_ngon_bis_eso[len(old_ngon_bis_eso//2):len(old_ngon_bis_eso)]+old_ngon_eso[-2]).all()
    old_ngon_ec     = PT.get_child_from_name(old_ngon_n,     'ElementConnectivity')[1]
    old_ngon_bis_ec = PT.get_child_from_name(old_ngon_bis_n, 'ElementConnectivity')[1]
    new_ngon_ec     = PT.get_child_from_name(new_ngon_n,     'ElementConnectivity')[1]
    if comm.size==1:
        assert (new_ngon_ec[0:len(old_ngon_ec)]  == old_ngon_ec).all()
        assert (new_ngon_ec[len(old_ngon_ec):len(new_ngon_ec)] == old_ngon_bis_ec).all()
    elif comm.size==2:
        if comm.rank == 0:
            assert (new_ngon_ec[0:len(old_ngon_ec)//2] == old_ngon_ec[0:len(old_ngon_ec)//2]).all()
        elif comm.rank == 1:
            assert (new_ngon_ec[len(new_ngon_ec//2)-1:-1] == old_ngon_bis_ec[len(old_ngon_bis_ec//2):len(old_ngon_bis_ec)]).all()

@pytest.mark.parametrize("align", [True, False])
def test_extrude_tri_to_prism_and_tris(align):

    # Prepare test
    n_vtx  = 25
    num    = 1
    er_max = 12
    tri = PT.new_Elements('TRI', type='TRI_3', erange=[1,17], econn=[1,2,3, 1,3,4])
    old_tri = PT.deep_copy(tri)

    # Run test
    new_tri1, new_tri2 = maia.algo.dist.extrude._extrude_tri_to_prism_and_tris(tri, num, n_vtx, er_max, align)

    # Verification
    assert PT.get_name(tri) == 'PENTA_6.1'
    assert PT.Element.CGNSName(tri) == 'PENTA_6'
    assert PT.get_name(new_tri1) == 'TRI_3.1a'
    assert PT.Element.CGNSName(new_tri1) == 'TRI_3'
    assert PT.get_name(new_tri2) == 'TRI_3.1b'
    assert PT.Element.CGNSName(new_tri2) == 'TRI_3'
    old_tri_ec  = PT.get_child_from_name(old_tri, 'ElementConnectivity')[1]
    new_tri1_ec = PT.get_child_from_name(new_tri1, 'ElementConnectivity')[1]
    new_tri2_ec = PT.get_child_from_name(new_tri2, 'ElementConnectivity')[1]
    penta_ec    = PT.get_child_from_name(tri, 'ElementConnectivity')[1]
    if align:
        assert (new_tri2_ec == old_tri_ec+n_vtx).all()
    else:
        assert (new_tri1_ec == old_tri_ec).all()
    assert (penta_ec == [1,2,3,26,27,28, 1,3,4,26,28,29]).all()
    old_tri_er  = PT.get_child_from_name(old_tri, 'ElementRange')[1]
    new_tri1_er = PT.get_child_from_name(new_tri1, 'ElementRange')[1]
    new_tri2_er = PT.get_child_from_name(new_tri2, 'ElementRange')[1]
    penta_er    = PT.get_child_from_name(tri, 'ElementRange')[1]
    assert (new_tri1_er == [er_max+1,    er_max+17]).all()
    assert (new_tri2_er == [er_max+1+17, er_max+2*17]).all()
    assert (penta_er == old_tri_er).all()

@pytest.mark.parametrize("align", [True, False])
def test_extrude_quad_to_hexa_and_quads(align):

    # Prepare test
    n_vtx  = 25
    num    = 1
    er_max = 12
    quad = PT.new_Elements('QUAD', type='QUAD_4', erange=[1,17], econn=[1,2,5,4, 2,3,6,5])
    old_quad = PT.deep_copy(quad)

    # Run test
    new_quad1, new_quad2 = maia.algo.dist.extrude._extrude_quad_to_hexa_and_quads(quad, num, n_vtx, er_max, align)

    # Verification
    assert PT.get_name(quad) == 'HEXA_8.1'
    assert PT.Element.CGNSName(quad) == 'HEXA_8'
    assert PT.get_name(new_quad1) == 'QUAD_4.1a'
    assert PT.Element.CGNSName(new_quad1) == 'QUAD_4'
    assert PT.get_name(new_quad2) == 'QUAD_4.1b'
    assert PT.Element.CGNSName(new_quad2) == 'QUAD_4'
    old_quad_ec  = PT.get_child_from_name(old_quad, 'ElementConnectivity')[1]
    new_quad1_ec = PT.get_child_from_name(new_quad1, 'ElementConnectivity')[1]
    new_quad2_ec = PT.get_child_from_name(new_quad2, 'ElementConnectivity')[1]
    hexa_ec      = PT.get_child_from_name(quad, 'ElementConnectivity')[1]
    if align:
        assert (new_quad2_ec == old_quad_ec+n_vtx).all()
    else:
        assert (new_quad1_ec == old_quad_ec).all()
    assert (hexa_ec == [1,2,5,4,26,27,30,29, 2,3,6,5,27,28,31,30]).all()
    old_quad_er  = PT.get_child_from_name(old_quad, 'ElementRange')[1]
    new_quad1_er = PT.get_child_from_name(new_quad1, 'ElementRange')[1]
    new_quad2_er = PT.get_child_from_name(new_quad2, 'ElementRange')[1]
    hexa_er      = PT.get_child_from_name(quad, 'ElementRange')[1]
    assert (new_quad1_er == [er_max+1,    er_max+17]).all()
    assert (new_quad2_er == [er_max+1+17, er_max+2*17]).all()
    assert (hexa_er == old_quad_er).all()

@pytest.mark.parametrize("align", [True, False])
def test_extrude_bar_to_quad(align):

    # Prepare test
    n_vtx  = 25
    num    = 1
    bar = PT.new_Elements('EdgeElements', type='BAR_2', erange=[1,17], econn=[1,2, 2,3])
    old_bar = PT.deep_copy(bar)

    # Run test
    maia.algo.dist.extrude._extrude_bar_to_quad(bar, num, n_vtx, align)

    # Verification
    assert PT.get_name(bar) == 'QUAD_4.1'
    assert PT.Element.CGNSName(bar) == 'QUAD_4'
    old_bar_ec = PT.get_child_from_name(old_bar, 'ElementConnectivity')[1]
    quad_ec    = PT.get_child_from_name(bar, 'ElementConnectivity')[1]
    if align:
        assert (quad_ec == [1,2,27,26, 2,3,28,27]).all()
    else:
        assert (quad_ec == [2,1,26,27, 3,2,27,28]).all()
    old_bar_er = PT.get_child_from_name(old_bar, 'ElementRange')[1]
    quad_er    = PT.get_child_from_name(bar, 'ElementRange')[1]
    assert (quad_er == old_bar_er).all()

@pytest_parallel.mark.parallel([1,2])
@pytest.mark.parametrize("pl", [None, [5,7,13]])
@pytest.mark.parametrize("data", [{}, {'data': [5.,7.,13.]}])
def test_pl_and_data_vtx_duplication(pl, data, comm):

    # Prepare test
    sub_data = data
    if comm.size == 1:
        sub_pl = None if pl is None else np.array([pl])
        distrib_idx = [0,3,3]
        if data is not {}: sub_data['data'] = np.array([5.,7.,13.])
    elif comm.size == 2:
        if comm.rank == 0:
            distrib_idx = [0,2,3]
            sub_pl = None if pl is None else np.array([[5,7]])
            if data is not {}: sub_data['data'] = np.array([5.,7.])
        elif comm.rank == 1:
            distrib_idx = [2,3,3]
            sub_pl = None if pl is None else np.array([[13]])
            if data is not {}: sub_data['data'] = np.array([13.])
    if (pl is None) and (data is {}):
        distrib_idx = [0,0,0]
    n_vtx_2d = 25

    # Run test
    new_distrib_idx, dist_data = maia.algo.dist.extrude._pl_and_data_vtx_duplication(sub_pl, distrib_idx, n_vtx_2d, sub_data, comm)

    # Verification
    if pl is None: assert not('PointList' in dist_data.keys())
    if data is {}: assert not('data' in dist_data.keys())
    if comm.size == 1:
        assert(new_distrib_idx == [0,6,6]).all()
        if pl is not None: assert (dist_data['PointList'] == [5,7,13,30,32,38]).all()
        if data is not {}: assert (dist_data['data'] == [5.,7.,13.,5.,7.,13.]).all()
    elif comm.size == 2:
        if comm.rank == 0:
            assert(new_distrib_idx == [0,3,6]).all()
            if pl is not None: assert (dist_data['PointList'] == [5,7,13]).all()
        elif comm.rank == 1:
            assert(new_distrib_idx == [3,6,6]).all()
            if pl is not None: assert (dist_data['PointList'] == [30,32,38]).all()
        if data is not {}: assert (dist_data['data'] == [5.,7.,13.]).all()

@pytest_parallel.mark.parallel([1,3])
@pytest.mark.parametrize("coords_dim", [2, 3])
@pytest.mark.parametrize("kplan_type", ['perio', 'fam_bc'])
def test_extrusion_2d_cart_ngon(coords_dim, kplan_type, comm):

    # Prepare 2D case
    dist_tree = maia.factory.generate_dist_block(11, 'TRI_3', comm)
    maia.algo.dist.convert_elements_to_ngon(dist_tree, comm)

    if coords_dim == 2:
        PT.rm_node_from_path(dist_tree, 'Base/zone/GridCoordinates/CoordinateZ')
        base = PT.get_child_from_name(dist_tree, 'Base')
        PT.set_value(base, [2,2])

    is_bar = lambda n: (PT.get_label(n) == 'Elements_t') and (PT.get_value(n)[0] == 3)
    is_edge_center = lambda n: (PT.get_label(n) in ['BC_t', 'GridConnectivity', 'GridConnectivity_1to1']) \
                                and (PT.Subset.GridLocation(n) == 'EdgeCenter')
    is_face_center = lambda n: (PT.get_label(n) in ['BC_t', 'GridConnectivity', 'GridConnectivity_1to1']) \
                                and (PT.Subset.GridLocation(n) == 'FaceCenter')

    base = PT.get_all_CGNSBase_t(dist_tree)[0]
    zone = PT.get_all_Zone_t(dist_tree)[0]

    # TO DO: create function in 'node_inspect.py' to obtain edges number of a zone ?
    # Remark: this is good only for 2D U-NGon meshes !
    n_edges = 0
    for elem_n in PT.get_nodes_from_predicate(zone, is_bar):
        er = PT.get_child_from_name(elem_n, 'ElementRange')[1]
        n_edges += er[1]-er[0]+1
    n_cell = PT.Zone.n_cell(zone)
    nb_edge_subsets_2d = len(PT.get_nodes_from_predicate(zone, is_edge_center))

    # Run test
    maia.algo.dist.extrusion_2d(dist_tree, [0., 0., 1.], comm, kplan_type=kplan_type)

    # Verification
    assert np.all(PT.get_value(base) == [3, 3])

    assert len(PT.get_nodes_from_predicate(zone, is_bar)) == 0

    ngon_n = PT.Zone.NGonNode(zone)
    assert PT.get_child_from_name(ngon_n, 'ParentElements') is not None

    assert len(PT.get_nodes_from_predicate(zone, is_edge_center)) == 0
    if kplan_type == 'perio':
        assert len(PT.get_nodes_from_predicate(zone, is_face_center)) == nb_edge_subsets_2d
        assert len(PT.get_nodes_from_predicates(zone, 'ZoneGridConnectivity_t/GridConnectivity_t')) == 2
        assert np.all(np.abs(PT.get_node_from_name(zone, 'Translation')[1]) == [0., 0., 1.])
    elif kplan_type == 'fam_bc':
        assert len(PT.get_nodes_from_predicate(zone, is_face_center)) == nb_edge_subsets_2d + 2
        assert len(PT.get_children_from_label(base, 'Family_t')) == 2

    assert PT.Zone.FaceSize(zone) == n_edges+2*n_cell

@pytest_parallel.mark.parallel([1])
@pytest.mark.parametrize("bc_loc", ['Vertex', 'EdgeCenter', 'CellCenter'])
@pytest.mark.parametrize("join", [True, False])
@pytest.mark.parametrize("dupl_vtx_info", [True, False])
def test_extrusion_2d_cart_ngon_loc(bc_loc, join, dupl_vtx_info, comm):

    # Prepare test
    dist_tree = maia.factory.generate_dist_block(2, 'TRI_3', comm)
    maia.algo.dist.convert_elements_to_ngon(dist_tree, comm)

    is_bar = lambda n: (PT.get_label(n) == 'Elements_t') and (PT.get_value(n)[0] == 3)
    is_edge_center = lambda n: (PT.get_label(n) in ['BC_t', 'GridConnectivity', 'GridConnectivity_1to1']) \
                                and (PT.Subset.GridLocation(n) == 'EdgeCenter')
    is_face_center = lambda n: (PT.get_label(n) in ['BC_t', 'GridConnectivity', 'GridConnectivity_1to1']) \
                                and (PT.Subset.GridLocation(n) == 'FaceCenter')

    base = PT.get_all_CGNSBase_t(dist_tree)[0]
    zone = PT.get_all_Zone_t(dist_tree)[0]

    n_cell = PT.Zone.n_cell(zone)
    n_vtx  = PT.Zone.n_vtx(zone)
    n_edges = 0
    for elem_n in PT.get_nodes_from_predicate(zone, is_bar):
        er = PT.get_child_from_name(elem_n, 'ElementRange')[1]
        n_edges += er[1]-er[0]+1

    # > Container CellCenter
    id_cc = np.arange(1,n_cell+1, dtype=pdm_dtype)+n_edges
    distrib_cell = np.array([0,n_cell,n_cell], dtype=pdm_dtype)
    fs_wopl_cc = PT.new_FlowSolution('FS_woPL#CellCenter', loc='CellCenter', fields={'Id': id_cc}, parent=zone)
    dd_wopl_cc = PT.new_DiscreteData('DD_woPL#CellCenter', loc='CellCenter', fields={'Id': id_cc}, parent=zone)
    fs_wpl_cc = PT.new_FlowSolution('FS_wPL#CellCenter', loc='CellCenter', fields={'Id': id_cc}, parent=zone)
    PT.new_IndexArray('PointList', value=[id_cc], parent=fs_wpl_cc)
    MT.newDistribution({'Index': distrib_cell}, parent=fs_wpl_cc)
    dd_wpl_cc = PT.new_DiscreteData('DD_wPL#CellCenter', loc='CellCenter', fields={'Id': id_cc}, parent=zone)
    PT.new_IndexArray('PointList', value=[id_cc], parent=dd_wpl_cc)
    MT.newDistribution({'Index': distrib_cell}, parent=dd_wpl_cc)
    zsr_wpl_cc = PT.new_ZoneSubRegion('ZSR_wPL#CellCenter', loc='CellCenter', fields={'Id': id_cc}, parent=zone)
    PT.new_IndexArray('PointList', value=[id_cc], parent=zsr_wpl_cc)
    MT.newDistribution({'Index': distrib_cell}, parent=zsr_wpl_cc)

    # > Container EdgeCenter
    id_ec = np.arange(1,n_edges+1, dtype=pdm_dtype)
    distrib_edge = np.array([0,n_edges,n_edges], dtype=pdm_dtype)
    # A confirmer : en 2D, un FS EdgeCenter sans PL n'a pas de sens, cf. FS FaceCenter en 3D, non ?
    # fs_wopl_ec = PT.new_FlowSolution('FS_woPL#EdgeCenter', loc='EdgeCenter', fields={'Id': id_ec}, parent=zone)
    # dd_wopl_ec = PT.new_DiscreteData('DD_woPL#EdgeCenter', loc='EdgeCenter', fields={'Id': id_ec}, parent=zone)
    fs_wpl_ec = PT.new_FlowSolution('FS_wPL#EdgeCenter', loc='EdgeCenter', fields={'Id': id_ec}, parent=zone)
    PT.new_IndexArray('PointList', value=[id_ec], parent=fs_wpl_ec)
    MT.newDistribution({'Index': distrib_edge}, parent=fs_wpl_ec)
    dd_wpl_ec = PT.new_DiscreteData('DD_wPL#EdgeCenter', loc='EdgeCenter', fields={'Id': id_ec}, parent=zone)
    PT.new_IndexArray('PointList', value=[id_ec], parent=dd_wpl_ec)
    MT.newDistribution({'Index': distrib_edge}, parent=dd_wpl_ec)
    zsr_wpl_ec = PT.new_ZoneSubRegion('ZSR_wPL#EdgeCenter', loc='EdgeCenter', fields={'Id': id_ec}, parent=zone)
    PT.new_IndexArray('PointList', value=[id_ec], parent=zsr_wpl_ec)
    MT.newDistribution({'Index': distrib_edge}, parent=zsr_wpl_ec)

    # > Container Vertex
    id_vtx = np.arange(1,n_vtx+1, dtype=pdm_dtype)
    distrib_vtx = np.array([0,n_vtx,n_vtx], dtype=pdm_dtype)
    fs_wopl_vtx = PT.new_FlowSolution('FS_woPL#Vertex', loc='Vertex', fields={'Id': id_vtx}, parent=zone)
    dd_wopl_vtx = PT.new_DiscreteData('DD_woPL#Vertex', loc='Vertex', fields={'Id': id_vtx}, parent=zone)
    fs_wpl_vtx = PT.new_FlowSolution('FS_wPL#Vertex', loc='Vertex', fields={'Id': id_vtx}, parent=zone)
    PT.new_IndexArray('PointList', value=[id_vtx], parent=fs_wpl_vtx)
    MT.newDistribution({'Index': distrib_vtx}, parent=fs_wpl_vtx)
    dd_wpl_vtx = PT.new_DiscreteData('DD_wPL#Vertex', loc='Vertex', fields={'Id': id_vtx}, parent=zone)
    PT.new_IndexArray('PointList', value=[id_vtx], parent=dd_wpl_vtx)
    MT.newDistribution({'Index': distrib_vtx}, parent=dd_wpl_vtx)
    zsr_wpl_vtx = PT.new_ZoneSubRegion('ZSR_wPL#Vertex', loc='Vertex', fields={'Id': id_vtx}, parent=zone)
    PT.new_IndexArray('PointList', value=[id_vtx], parent=zsr_wpl_vtx)
    MT.newDistribution({'Index': distrib_vtx}, parent=zsr_wpl_vtx)

    xmin = PT.get_node_from_predicates(zone,'ZoneBC_t/Xmin')
    xmax = PT.get_node_from_predicates(zone,'ZoneBC_t/Xmax')
    ymin = PT.get_node_from_predicates(zone,'ZoneBC_t/Ymin')
    ymax = PT.get_node_from_predicates(zone,'ZoneBC_t/Ymax')

    # > BCDataSet Xmin CellCenter
    bcds_wpl_cc = PT.new_BCDataSet(name='BCDS_wpl#CellCenter', loc='CellCenter', point_list=[[6]], parent=xmin)
    PT.new_BCData('NeumannData', fields={'Id': [6]}, parent=bcds_wpl_cc)
    MT.newDistribution({'Index': np.array([0,1,1],dtype=pdm_dtype)}, parent=bcds_wpl_cc)
    # > BCDataSet Xmin EdgeCenter
    bcds_wpl_ec = PT.new_BCDataSet(name='BCDS_wpl#EdgeCenter', loc='EdgeCenter', point_list=[[2]], parent=xmin)
    PT.new_BCData('NeumannData', fields={'Id': [2]}, parent=bcds_wpl_ec)
    MT.newDistribution({'Index': np.array([0,1,1],dtype=pdm_dtype)}, parent=bcds_wpl_ec)
    # > BCDataSet Xmin Vertex
    bcds_wpl_vtx = PT.new_BCDataSet(name='BCDS_wpl#Vertex', loc='Vertex', point_list=[[1,3]], parent=xmin)
    PT.new_BCData('NeumannData', fields={'Id': [1,3]}, parent=bcds_wpl_vtx)
    MT.newDistribution({'Index': np.array([0,2,2],dtype=pdm_dtype)}, parent=bcds_wpl_vtx)

    # > Manage BC location
    PT.set_value(PT.get_child_from_name(xmin,'GridLocation'), bc_loc)
    PT.set_value(PT.get_child_from_name(xmax,'GridLocation'), bc_loc)
    PT.set_value(PT.get_child_from_name(ymin,'GridLocation'), bc_loc)
    PT.set_value(PT.get_child_from_name(ymax,'GridLocation'), bc_loc)
    pl_xmin = PT.get_child_from_name(xmin,'PointList')
    pl_xmax = PT.get_child_from_name(xmax,'PointList')
    pl_ymin = PT.get_child_from_name(ymin,'PointList')
    pl_ymax = PT.get_child_from_name(ymax,'PointList')
    bcds_wopl = PT.new_BCDataSet(name=f'BCDS_wopl#{bc_loc}', loc=f'{bc_loc}', parent=xmin)
    if bc_loc == 'Vertex':
        PT.set_value(pl_xmin, [np.array([1,3],dtype=pdm_dtype)])
        PT.set_value(pl_xmax, [np.array([2,4],dtype=pdm_dtype)])
        PT.set_value(pl_ymin, [np.array([1,2],dtype=pdm_dtype)])
        PT.set_value(pl_ymax, [np.array([3,4],dtype=pdm_dtype)])
        PT.update_child(PT.get_child_from_name(xmin, ':CGNS#Distribution'), name='Index', value=np.array([0,2,2],dtype=pdm_dtype))
        PT.update_child(PT.get_child_from_name(xmax, ':CGNS#Distribution'), name='Index', value=np.array([0,2,2],dtype=pdm_dtype))
        PT.update_child(PT.get_child_from_name(ymin, ':CGNS#Distribution'), name='Index', value=np.array([0,2,2],dtype=pdm_dtype))
        PT.update_child(PT.get_child_from_name(ymax, ':CGNS#Distribution'), name='Index', value=np.array([0,2,2],dtype=pdm_dtype))
        PT.new_BCData('NeumannData', fields={'Id': [1,3]}, parent=bcds_wopl)
        zsr_xmin = PT.new_ZoneSubRegion(f'ZSR_link_Xmin#{bc_loc}', loc=bc_loc, bc_name='Xmin', fields={'Id': [1,3]}, parent=zone)
        zsr_xmin_copy = PT.new_ZoneSubRegion(f'ZSR_copied_from_Xmin#{bc_loc}', loc=bc_loc, point_list=[[1,3]], fields={'Id': [1,3]}, parent=zone)
    elif bc_loc == 'EdgeCenter':
        PT.new_BCData('NeumannData', fields={'Id': [2]}, parent=bcds_wopl)
        zsr_xmin = PT.new_ZoneSubRegion(f'ZSR_link_Xmin#{bc_loc}', loc=bc_loc, bc_name='Xmin', fields={'Id': [2]}, parent=zone)
        zsr_xmin_copy = PT.new_ZoneSubRegion(f'ZSR_copied_from_Xmin#{bc_loc}', loc=bc_loc, point_list=[[2]], fields={'Id': [2]}, parent=zone)
    elif bc_loc == 'CellCenter':
        PT.set_value(pl_xmin, [np.array([6],dtype=pdm_dtype)])
        PT.set_value(pl_xmax, [np.array([7],dtype=pdm_dtype)])
        PT.set_value(pl_ymin, [np.array([6],dtype=pdm_dtype)])
        PT.set_value(pl_ymax, [np.array([7],dtype=pdm_dtype)])
        PT.new_BCData('NeumannData', fields={'Id': [6]}, parent=bcds_wopl)
        zsr_xmin = PT.new_ZoneSubRegion(f'ZSR_link_Xmin#{bc_loc}', loc=bc_loc, bc_name='Xmin', fields={'Id': [6]}, parent=zone)
        zsr_xmin_copy = PT.new_ZoneSubRegion(f'ZSR_copied_from_Xmin#{bc_loc}', loc=bc_loc, point_list=[[6]], fields={'Id': [6]}, parent=zone)
    MT.newDistribution({'Index': np.array([0,PT.Subset.n_elem(xmin),PT.Subset.n_elem(xmin)],dtype=pdm_dtype)}, parent=zsr_xmin_copy)

    if join:
        # Remark: GC is not well defined but enough for test. For exemple, wrong value and not perio.
        zone_gc = PT.new_ZoneGridConnectivity(parent=zone)
        xmin_pld = PT.deep_copy(pl_xmax)
        xmax_pld = PT.deep_copy(pl_xmin)
        PT.set_name(xmin_pld, 'PointListDonor')
        PT.set_name(xmax_pld, 'PointListDonor')
        PT.add_child(xmin, xmin_pld)
        PT.add_child(xmax, xmax_pld)
        PT.update_node(xmin, value=f'{base[0]}/{zone[0]}', label='GridConnectivity_t')
        PT.update_node(xmax, value=f'{base[0]}/{zone[0]}', label='GridConnectivity_t')
        PT.rm_nodes_from_label(xmin, 'BCDataSet_t')
        PT.rm_nodes_from_label(xmax, 'BCDataSet_t')
        PT.add_child(zone_gc, xmin)
        PT.add_child(zone_gc, xmax)
        PT.rm_node_from_path(zone, 'ZoneBC/Xmin')
        PT.rm_node_from_path(zone, 'ZoneBC/Xmax')
        PT.rm_node_from_path(zsr_xmin, 'BCRegionName')
        PT.new_Descriptor('GridConnectivityRegionName', 'Xmin', parent=zsr_xmin)

    # Run test
    maia.algo.dist.extrusion_2d(dist_tree, [0., 0., 1.], comm, dupl_vtx_info=dupl_vtx_info)

    # Verification
    for container_name in ['FS_woPL','DD_woPL']:
        for loc in ['Vertex', 'CellCenter']:
            container = PT.get_node_from_name(zone, f'{container_name}#{loc}')
            assert PT.Subset.GridLocation(container) == loc
            if loc == 'CellCenter':
                assert (PT.get_child_from_name(container, 'Id')[1] == [6,7]).all()
                assert (PT.get_child_from_name(container, 'PointList') is None)
            elif (loc == 'Vertex') and dupl_vtx_info:
                assert (PT.get_child_from_name(container, 'Id')[1] == [1,2,3,4, 1,2,3,4]).all()
                assert (PT.get_child_from_name(container, 'PointList') is None)
            elif (loc == 'Vertex') and not dupl_vtx_info:
                assert (PT.get_child_from_name(container, 'Id')[1] == [1,2,3,4]).all()
                assert (PT.get_child_from_name(container, 'PointList')[1] == [1,2,3,4]).all()
    for container_name in ['FS_wPL','DD_wPL','ZSR_wPL']:
        for loc in ['Vertex', 'CellCenter']:
            container = PT.get_node_from_name(zone, f'{container_name}#{loc}')
            assert PT.Subset.GridLocation(container) == loc
            if loc == 'CellCenter':
                assert (PT.get_child_from_name(container, 'Id')[1] == [6,7]).all()
                assert (PT.get_child_from_name(container, 'PointList')[1] == [6,7]).all()
            elif (loc == 'Vertex') and dupl_vtx_info:
                assert (PT.get_child_from_name(container, 'Id')[1] == [1,2,3,4, 1,2,3,4]).all()
                assert (PT.get_child_from_name(container, 'PointList')[1] == [1,2,3,4, 1+n_vtx,2+n_vtx,3+n_vtx,4+n_vtx]).all()
            elif (loc == 'Vertex') and not dupl_vtx_info:
                assert (PT.get_child_from_name(container, 'Id')[1] == [1,2,3,4]).all()
                assert (PT.get_child_from_name(container, 'PointList')[1] == [1,2,3,4]).all()
    for container_name in ['FS_wPL','DD_wPL','ZSR_wPL']:
        container = PT.get_node_from_name(zone, f'{container_name}#EdgeCenter')
        assert PT.Subset.GridLocation(container) == 'FaceCenter'
        assert (PT.get_child_from_name(container, 'Id')[1] == [1,2,3,4,5]).all()
        assert (PT.get_child_from_name(container, 'PointList')[1] == [1,2,3,4,5]).all()

    if not join:
        container = PT.get_node_from_name(zone, 'BCDS_wpl#CellCenter')
        assert PT.Subset.GridLocation(container) == 'CellCenter'
        assert (PT.get_node_from_name(container, 'Id')[1] == [6]).all()
        assert (PT.get_child_from_name(container, 'PointList')[1] == [6]).all()

        container = PT.get_node_from_name(zone, 'BCDS_wpl#EdgeCenter')
        assert PT.Subset.GridLocation(container) == 'FaceCenter'
        assert (PT.get_node_from_name(container, 'Id')[1] == [2]).all()
        assert (PT.get_child_from_name(container, 'PointList')[1] == [2]).all()

        container = PT.get_node_from_name(zone, 'BCDS_wpl#Vertex')
        assert PT.Subset.GridLocation(container) == 'Vertex'
        if dupl_vtx_info:
            assert (PT.get_node_from_name(container, 'Id')[1] == [1,3, 1,3]).all()
            assert (PT.get_child_from_name(container, 'PointList')[1] == [1,3, 1+n_vtx,3+n_vtx]).all()
        else:
            assert (PT.get_node_from_name(container, 'Id')[1] == [1,3]).all()
            assert (PT.get_child_from_name(container, 'PointList')[1] == [1,3]).all()

        container = PT.get_node_from_name(zone, f'BCDS_wopl#{bc_loc}')
        if bc_loc == 'Vertex':
            assert PT.Subset.GridLocation(container) == 'Vertex'
            if dupl_vtx_info:
                assert (PT.get_node_from_name(container, 'Id')[1] == [1,3, 1,3]).all()
                assert (PT.get_child_from_name(container, 'PointList') is None)
            else:
                assert (PT.get_node_from_name(container, 'Id')[1] == [1,3]).all()
                assert (PT.get_child_from_name(container, 'PointList')[1] == [1,3]).all()
        elif bc_loc == 'EdgeCenter':
            assert PT.Subset.GridLocation(container) == 'FaceCenter'
            assert (PT.get_node_from_name(container, 'Id')[1] == [2]).all()
            assert (PT.get_child_from_name(container, 'PointList') is None)
        elif bc_loc == 'CellCenter':
            assert PT.Subset.GridLocation(container) == 'CellCenter'
            assert (PT.get_node_from_name(container, 'Id')[1] == [6]).all()
            assert (PT.get_child_from_name(container, 'PointList') is None)

    xmin = PT.get_node_from_name(zone,'Xmin')
    xmax = PT.get_node_from_name(zone,'Xmax')
    ymin = PT.get_node_from_name(zone,'Ymin')
    ymax = PT.get_node_from_name(zone,'Ymax')
    
    if bc_loc == 'CellCenter':
       assert PT.Subset.GridLocation(xmin) == 'CellCenter'
       assert (PT.get_child_from_name(xmin, 'PointList')[1] == [6]).all()
       if join: assert (PT.get_child_from_name(xmin, 'PointListDonor')[1] == [7]).all()
       assert PT.Subset.GridLocation(xmax) == 'CellCenter'
       assert (PT.get_child_from_name(xmax, 'PointList')[1] == [7]).all()
       if join: assert (PT.get_child_from_name(xmax, 'PointListDonor')[1] == [6]).all()
       assert PT.Subset.GridLocation(ymin) == 'CellCenter'
       assert (PT.get_child_from_name(ymin, 'PointList')[1] == [6]).all()
       assert PT.Subset.GridLocation(ymax) == 'CellCenter'
       assert (PT.get_child_from_name(ymax, 'PointList')[1] == [7]).all()
    elif bc_loc == 'EdgeCenter':
       assert PT.Subset.GridLocation(xmin) == 'FaceCenter'
       assert (PT.get_child_from_name(xmin, 'PointList')[1] == [2]).all()
       if join: assert (PT.get_child_from_name(xmin, 'PointListDonor')[1] == [4]).all()
       assert PT.Subset.GridLocation(xmax) == 'FaceCenter'
       assert (PT.get_child_from_name(xmax, 'PointList')[1] == [4]).all()
       if join: assert (PT.get_child_from_name(xmax, 'PointListDonor')[1] == [2]).all()
       assert PT.Subset.GridLocation(ymin) == 'FaceCenter'
       assert (PT.get_child_from_name(ymin, 'PointList')[1] == [1]).all()
       assert PT.Subset.GridLocation(ymax) == 'FaceCenter'
       assert (PT.get_child_from_name(ymax, 'PointList')[1] == [5]).all()
    elif bc_loc == 'Vertex':
       assert PT.Subset.GridLocation(xmin) == 'Vertex'
       assert (PT.get_child_from_name(xmin, 'PointList')[1] == [1,3, 1+n_vtx,3+n_vtx]).all()
       if join: assert (PT.get_child_from_name(xmin, 'PointListDonor')[1] == [2,4, 2+n_vtx,4+n_vtx]).all()
       assert PT.Subset.GridLocation(xmax) == 'Vertex'
       assert (PT.get_child_from_name(xmax, 'PointList')[1] == [2,4, 2+n_vtx,4+n_vtx]).all()
       if join: assert (PT.get_child_from_name(xmax, 'PointListDonor')[1] == [1,3, 1+n_vtx,3+n_vtx]).all()
       assert PT.Subset.GridLocation(ymin) == 'Vertex'
       assert (PT.get_child_from_name(ymin, 'PointList')[1] == [1,2, 1+n_vtx,2+n_vtx]).all()
       assert PT.Subset.GridLocation(ymax) == 'Vertex'
       assert (PT.get_child_from_name(ymax, 'PointList')[1] == [3,4, 3+n_vtx,4+n_vtx]).all()

@pytest_parallel.mark.parallel([1,3])
@pytest.mark.parametrize("element_type", ['TRI_3', 'QUAD_4'])
def test_extrusion_2d_cart_elem(element_type, comm):

    # Prepare 2D case
    dist_tree = maia.factory.generate_dist_block(11, element_type, comm)

    is_bar = lambda n: (PT.get_label(n) == 'Elements_t') and (PT.get_value(n)[0] == 3)
    is_edge_center = lambda n: (PT.get_label(n) in ['BC_t', 'GridConnectivity', 'GridConnectivity_1to1']) \
                                and (PT.Subset.GridLocation(n) == 'EdgeCenter')
    is_face_center = lambda n: (PT.get_label(n) in ['BC_t', 'GridConnectivity', 'GridConnectivity_1to1']) \
                                and (PT.Subset.GridLocation(n) == 'FaceCenter')

    base = PT.get_all_CGNSBase_t(dist_tree)[0]
    zone = PT.get_all_Zone_t(dist_tree)[0]

    n_cell = PT.Zone.n_cell(zone)
    nb_edge_subsets_2d = len(PT.get_nodes_from_predicate(zone, is_edge_center))

    # Run test
    maia.algo.dist.extrusion_2d(dist_tree, [0., 0., 1.], comm)

    # Verification
    assert np.all(PT.get_value(base) == [3, 3])

    assert len(PT.get_nodes_from_predicate(zone, is_bar)) == 0

    for elem in PT.Zone.get_ordered_elements(zone):
        if element_type == 'TRI_3':
	        assert PT.Element.CGNSName(elem) in ['TRI_3', 'QUAD_4', 'PENTA_6']
        elif element_type == 'QUAD_4':
	        assert PT.Element.CGNSName(elem) in ['QUAD_4', 'HEXA_8']

    assert [len(elts) for elts in PT.Zone.get_ordered_elements_per_dim(zone)] == [0, 0, 3, 1]

    assert len(PT.get_nodes_from_predicate(zone, is_edge_center)) == 0
    assert len(PT.get_nodes_from_predicate(zone, is_face_center)) == nb_edge_subsets_2d
    assert len(PT.get_nodes_from_predicates(zone, 'ZoneGridConnectivity_t/GridConnectivity_t')) == 2
    assert np.all(np.abs(PT.get_node_from_name(zone, 'Translation')[1]) == [0., 0., 1.])

@pytest_parallel.mark.parallel([1,3])
def test_extrusion_2d_cyl_ngon(comm):

    # Prepare 2D case
    dist_tree = maia.factory.generate_dist_block(11, 'TRI_3', comm)
    maia.algo.dist.convert_elements_to_ngon(dist_tree, comm)
    
    maia.algo.transform.cartesian_to_cylindrical(dist_tree, [1., 0., 0.])
    
    is_bar = lambda n: (PT.get_label(n) == 'Elements_t') and (PT.get_value(n)[0] == 3)
    is_edge_center = lambda n: (PT.get_label(n) in ['BC_t', 'GridConnectivity', 'GridConnectivity_1to1']) \
                                and (PT.Subset.GridLocation(n) == 'EdgeCenter')
    is_face_center = lambda n: (PT.get_label(n) in ['BC_t', 'GridConnectivity', 'GridConnectivity_1to1']) \
                                and (PT.Subset.GridLocation(n) == 'FaceCenter')
    
    base = PT.get_all_CGNSBase_t(dist_tree)[0]
    zone = PT.get_all_Zone_t(dist_tree)[0]
    
    # TO DO: create function in 'node_inspect.py' to obtain edges number of a mesh ?
    n_edges = 0
    for elem_n in PT.get_nodes_from_predicate(zone, is_bar):
        er = PT.get_child_from_name(elem_n, 'ElementRange')[1]
        n_edges += er[1]-er[0]+1
    n_cell = PT.Zone.n_cell(zone)
    nb_edge_subsets_2d = len(PT.get_nodes_from_predicate(zone, is_edge_center))
    
    # Run test
    maia.algo.dist.extrusion_2d(dist_tree, [0., 1., 0.], comm)
    
    # Verification
    assert np.all(PT.get_value(base) == [3, 3])
    
    assert len(PT.get_nodes_from_predicate(zone, is_bar)) == 0
    
    ngon_n = PT.Zone.NGonNode(zone)
    assert PT.get_child_from_name(ngon_n, 'ParentElements') is not None
    
    assert len(PT.get_nodes_from_predicate(zone, is_edge_center)) == 0
    assert len(PT.get_nodes_from_predicate(zone, is_face_center)) == nb_edge_subsets_2d
  
    assert PT.Zone.FaceSize(zone) == n_edges+2*n_cell
