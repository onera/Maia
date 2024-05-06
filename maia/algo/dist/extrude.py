import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia          import npy_pdm_gnum_dtype  as pdm_dtype
from maia.transfer import protocols           as EP
from maia.utils    import np_utils, par_utils

def _nodes_duplication(zone, extrusion_vector, comm):
    """
    Internal function used by _extrusion_2d_u_ngon and _extrusion_2d_u_elem to create 
    the duplicated nodes needed to generate the second plan
    """
    # > Define new distribution
    distrib_vtx_n    = MT.getDistribution(zone, 'Vertex')
    distrib_vtx      = PT.get_value(distrib_vtx_n)
    new_distrib_vtx  = par_utils.uniform_distribution(distrib_vtx[2]*2,  comm)
    # > Change value of Z or Theta coordinates in part_data
    coords = PT.Zone.coordinates(zone)
    part_data = {}
    for i, name in enumerate(coords._fields):
       part_data[name] = [coords[i], coords[i]+extrusion_vector[i]]
    # > Compute ln_to_gn
    ln_to_gn_l = []
    ln_to_gn_l.append(np.arange(distrib_vtx[0], distrib_vtx[1])+1)
    ln_to_gn_l.append(np.arange(distrib_vtx[0]+distrib_vtx[2], distrib_vtx[1]+distrib_vtx[2])+1)
    # > Part to block
    dist_data = EP.part_to_block(part_data, new_distrib_vtx, ln_to_gn_l, comm)
    # > Update coordinates values
    for name in coords._fields:
        coord_n = PT.get_node_from_predicates(zone, f'GridCoordinates/{name}')
        PT.set_value(coord_n, dist_data[name])
    # > Update vertex distribution
    PT.set_value(distrib_vtx_n, new_distrib_vtx)
    

def _determine_mesh_orientation(zone, extrusion_vector, comm):
    # Remark: to determine orientation of faces, we need to test the sign of
    #         scalar product between extrusion vector and the first face in
    #         the current proc
    # > Get the third first nodes of the first face in current proc
    if PT.Zone.has_ngon_elements(zone): #U-NGon zone
        first_elem_2d = PT.Zone.NGonNode(zone)
    else: #U-Elem zone
        first_elem_2d = PT.Zone.get_ordered_elements_per_dim(zone)[2][0]
    first_elem_2d_ec = PT.get_value(PT.get_child_from_name(first_elem_2d, 'ElementConnectivity'))
    if len(first_elem_2d_ec)>0:
        nodes_of_first_face = first_elem_2d_ec[0:3]
    else:
        nodes_of_first_face = []
    # > Get coordinates of nodes of the first face
    coords = PT.Zone.coordinates(zone)
    dist_coords_data = {}
    for i, name in enumerate(coords._fields):
        dist_coords_data[name] = coords[i]
    distrib_vtx = MT.getDistribution(zone, 'Vertex')[1]
    ln_to_gn_vtx_list = [nodes_of_first_face]
    part_coords_data = EP.block_to_part(dist_coords_data, distrib_vtx, ln_to_gn_vtx_list, comm)
    align = 0
    # > Scalar product
    if len(nodes_of_first_face) > 0:
        # vector a = n1_n2
        vector_a_1 = part_coords_data[coords._fields[0]][0][1] - part_coords_data[coords._fields[0]][0][0]
        vector_a_2 = part_coords_data[coords._fields[1]][0][1] - part_coords_data[coords._fields[1]][0][0]
        vector_a_3 = part_coords_data[coords._fields[2]][0][1] - part_coords_data[coords._fields[2]][0][0]
        # vector b = n2_n3
        vector_b_1 = part_coords_data[coords._fields[0]][0][2] - part_coords_data[coords._fields[0]][0][1]
        vector_b_2 = part_coords_data[coords._fields[1]][0][2] - part_coords_data[coords._fields[1]][0][1]
        vector_b_3 = part_coords_data[coords._fields[2]][0][2] - part_coords_data[coords._fields[2]][0][1]
        # vector c = a^b
        vector_c_1 = vector_a_2*vector_b_3 - vector_a_3*vector_b_2
        vector_c_2 = vector_a_3*vector_b_1 - vector_a_1*vector_b_3
        vector_c_3 = vector_a_1*vector_b_2 - vector_a_2*vector_b_1
        # ps = c.extrusion_vector
        ps = vector_c_1*extrusion_vector[0] + vector_c_2*extrusion_vector[1] + vector_c_3*extrusion_vector[2]
        # >>> d. change order of one of two
        if ps>0: # Need to change initial faces
            align = 1
        elif ps<0: # Need to change duplicated faces
            align = -1
    aligns = np.array(comm.allgather(align), dtype=np.int32)
    if np.all(aligns >= 0):
        return True
    elif np.all(aligns <= 0):
        return False
    else:
        raise ValueError('Faces of 2D mesh are not uniform oriented !')


def _reorder_ngon_ec(ngon_n):
    """
    Internal function used by _ngon_duplication to reorder the NGonNode connectivity
    to be exited normal
    
    TO DO
    Question: To be efficient, instead of reverse roll each face, we can reverse
              all the EC, but we loose the initial numbering !
    """
    ec_n  = PT.get_child_from_name(ngon_n, 'ElementConnectivity')
    eso_n = PT.get_child_from_name(ngon_n, 'ElementStartOffset')
    
    ec  = PT.get_value(ec_n)
    eso = PT.get_value(eso_n)
    distrib_ec = MT.getDistribution(ngon_n, 'ElementConnectivity')[1]
    
    for i in range(len(eso)-1):
        sub_ec = ec[eso[i]-distrib_ec[0]:eso[i+1]-distrib_ec[0]]
        ec[eso[i]-distrib_ec[0]:eso[i+1]-distrib_ec[0]] = np_utils.roll_from(sub_ec, start_idx=0, reverse=True)
    

def _ngon_duplication(zone, comm, align=True):
    """
    Internal function used by _extrusion_2d_u_ngon to create the duplicated NGonNode
    needed to generate the second plan
    
    Remark : no need to change the ElementStartOffset
    """
    # > Get number of 2D cells (face)
    n_cell_2d = PT.Zone.n_cell(zone)
    # > Get number of vertices
    n_vtx_2d = PT.Zone.n_vtx(zone)
    # > Add ParentElements to NGon node
    ngon_n = PT.Zone.NGonNode(zone)
    distrib_elem = MT.getDistribution(ngon_n, 'Element')[1]
    er = PT.get_value(PT.get_child_from_name(ngon_n, 'ElementRange'))
    pe = np.zeros((distrib_elem[1]-distrib_elem[0],2), dtype=pdm_dtype)
    pe[:,0] = np.arange(distrib_elem[0],distrib_elem[1],dtype=pdm_dtype)+er[0]+2*n_cell_2d
    PT.new_child(ngon_n, 'ParentElements', 'DataArray_t', pe)
    # > Duplicate NGon
    ngon_bis_n = PT.deep_copy(ngon_n)
    PT.set_name(ngon_bis_n, f'{ngon_bis_n[0]}_bis')
    PT.add_child(zone, ngon_bis_n)
    # > Update ElementRange
    # on suppose que l'on a deja tous les elements 1D et 2D de définis dans le CGNS
    # TO DO: creer le NGon si on a que Bar + PE avant d'appliquer cette fonction
    er_bis_n = PT.get_child_from_name(ngon_bis_n, 'ElementRange')
    PT.set_value(er_bis_n,PT.get_value(er_bis_n)+n_cell_2d)
    # > Update ElementConnectivity
    ec_bis_n = PT.get_child_from_name(ngon_bis_n, 'ElementConnectivity')
    PT.set_value(ec_bis_n, PT.get_value(ec_bis_n)+n_vtx_2d)
    if align: # Need to change Ngon
        _reorder_ngon_ec(ngon_n)
    else: # Need to change duplicated Ngon
        _reorder_ngon_ec(ngon_bis_n)


def _extrude_bar_to_ngon(bar, n_vtx, n_cell, align=True):
    """
    Internal function used by _extrusion_2d_u_ngon to create face by extrusion of BAR elements
    """
    # > Change value: 3 => 22
    PT.set_value(bar, [22, 0])
    # > Create ElementStartOffset
    distrib_elem = MT.getDistribution(bar, 'Element')[1]
    eso = np.arange(distrib_elem[0], distrib_elem[1]+1)*4
    PT.new_child(bar, 'ElementStartOffset', 'DataArray_t', eso)
    # > Update ElementConnectivity
    ec_n = PT.get_child_from_name(bar, 'ElementConnectivity')
    first_nodes  = ec_n[1][0::2]
    second_nodes = ec_n[1][1::2]
    third_nodes  = np.copy(ec_n[1][1::2]) + n_vtx
    fourth_nodes = np.copy(ec_n[1][0::2]) + n_vtx
    if align:
        new_ec = np_utils.interweave_arrays([second_nodes, first_nodes, fourth_nodes, third_nodes])
    else:
        new_ec = np_utils.interweave_arrays([first_nodes, second_nodes, third_nodes, fourth_nodes])
    PT.set_value(ec_n, new_ec)
    # > Create ElementConnectivity distribution
    MT.newDistribution({'ElementConnectivity' : distrib_elem*4}, parent=bar)
    # > Update PE
    # Remark: new cells are the former faces because we keep the order
    #         so we just have to shift there values of 2*n_cell
    pe_n = PT.get_child_from_name(bar, 'ParentElements')
    new_pe = PT.get_value(pe_n) + 2*n_cell*(PT.get_value(pe_n)>0)
    PT.set_value(pe_n, new_pe.astype(pdm_dtype))
    
    
def _merge_ngons(zone, comm):
    """
    Internal function used by _extrusion_2d_u_ngon to create the merged NGonNode
    
    TO DO
    > mutualier avec la fonction _merge_ngon de merge.py ?
    """
    n_faces = 0
    tot_nodes_in_new_ec = 0
    part_diff_eso = []
    part_ec = []
    part_pe0 = []
    part_pe1 = []
    ln_to_gn_elem_l = []
    ln_to_gn_ec_l = []
    ngon_names = []
    size_prev_eso = 0
    for ngon_n in PT.Zone.get_ordered_elements_per_dim(zone)[2]:
        assert PT.get_value(ngon_n)[0] == 22
        ngon_names.append(PT.get_name(ngon_n))
        er  = PT.get_child_from_name(ngon_n, 'ElementRange')[1]
        ec  = PT.get_child_from_name(ngon_n, 'ElementConnectivity')[1]
        eso = PT.get_child_from_name(ngon_n, 'ElementStartOffset')[1]
        pe  = PT.get_child_from_name(ngon_n, 'ParentElements')[1]
        distrib_elem = MT.getDistribution(ngon_n, 'Element')[1]
        distrib_ec   = MT.getDistribution(ngon_n, 'ElementConnectivity')[1]
        n_faces += er[1]-er[0]+1
        tot_nodes_in_new_ec += distrib_ec[2]
        part_diff_eso.append(np.diff(eso).astype(np.int32))
        part_ec.append(ec)
        part_pe0.append(pe[:,0])
        part_pe1.append(pe[:,1])
        gn_beg_elem = distrib_elem[0] + er[0] - 1
        gn_end_elem = distrib_elem[1] + er[0] - 1
        ln_to_gn_elem_l.append(np.arange(gn_beg_elem, gn_end_elem)+1)
        gn_beg_ec = distrib_ec[0] + size_prev_eso
        gn_end_ec = distrib_ec[1] + size_prev_eso
        ln_to_gn_ec_l.append(np.arange(gn_beg_ec, gn_end_ec)+1)
        size_prev_eso += distrib_ec[2]
    # > Define new ElementRange
    # Warning : ne fonctionne pas si il y a des 'NODE' dans l'arbre !
    #           si tous les NODE sont avant, il faut faire démarrer l'ER au total des éléments de NODE
    #           si tous les NODE sont après, rien à changer
    #           si les NODE ne sont pas spécialement ordonnés, faire une réorganisation en entrée de 'extrusion_2d' ?
    new_er = [1, n_faces]
    # > Define new Element distribution
    new_distrib_elem = par_utils.uniform_distribution(n_faces,  comm)
    # > Exchange to define new ElementStartOffset, ElementConnectivity and ParentElements
    ptb = EP.PartToBlock(new_distrib_elem, ln_to_gn_elem_l, comm)
    new_diff_eso, new_ec = ptb.exchange_field(part_ec, part_diff_eso)
    _, new_pe0 = ptb.exchange_field(part_pe0)
    _, new_pe1 = ptb.exchange_field(part_pe1)
    new_pe = np.concatenate([new_pe0, new_pe1]).reshape((len(new_pe0),2),order='F')
    nodes_in_loc_eso_by_proc = comm.allgather(np.sum(new_diff_eso))
    prev_nodes_number_in_eso = np.sum(nodes_in_loc_eso_by_proc[0:comm.rank],dtype=new_diff_eso.dtype)
    new_eso = np.concatenate([[0], np.cumsum(new_diff_eso)]) + prev_nodes_number_in_eso
    # > Define new ElementConnectivity distribution
    new_distrib_ec = np.array([prev_nodes_number_in_eso, np.sum(nodes_in_loc_eso_by_proc[0:comm.rank+1],dtype=new_diff_eso.dtype), np.sum(nodes_in_loc_eso_by_proc)])
    # > Delete old ngons
    for ngon_name in ngon_names:
        PT.rm_node_from_path(zone, ngon_name)
    # > Create new NGon node
    new_ngon_n = PT.new_NGonElements(erange=new_er, eso=new_eso, ec=new_ec, pe=new_pe, parent=zone)
    MT.newDistribution({'Element' : new_distrib_elem}, parent=new_ngon_n)
    MT.newDistribution({'ElementConnectivity' : new_distrib_ec}, parent=new_ngon_n)
    

def _extrusion_2d_u_ngon(zone, base_name, extrusion_vector, comm, kplan_type='perio'):
    """
    Internal function used by extrusion_2d to extrude a 2D unstructured mesh describe by edges
    in the direction of the extrusion vector in cartesian and cylindrical coordinates.
    """
    
    # 0/ Global information
    n_vtx  = PT.Zone.n_vtx(zone)
    n_cell = PT.Zone.n_cell(zone)
    # TO DO: create function in 'node_inspect.py' to obtain edges number of a mesh ?
    is_bar = lambda n: (PT.get_label(n) == 'Elements_t') and (PT.get_value(n)[0] == 3)
    n_edges = 0
    for elem_n in PT.get_nodes_from_predicate(zone, is_bar):
        er = PT.get_child_from_name(elem_n, 'ElementRange')[1]
        n_edges += er[1]-er[0]+1
    
    # 1/ Duplication of nodes to generate the second plan
    _nodes_duplication(zone, extrusion_vector, comm)
    
    # 1bis/ Determine the mesh orientation
    align = _determine_mesh_orientation(zone, extrusion_vector, comm)
    
    # 2/ Create faces of the second plan
    align = _ngon_duplication(zone, comm, align=align)
    
    # 3/ Extrude Bar to NGon
    is_bar = lambda n: (PT.get_label(n) == 'Elements_t') and (PT.get_value(n)[0] == 3)
    for bar in PT.get_nodes_from_predicate(zone, is_bar):
        _extrude_bar_to_ngon(bar, n_vtx, n_cell, align=align)
    
    # 4/ Merge all NGon nodes
    _merge_ngons(zone, comm)
    
    # 5/ Manage K-plans
    if kplan_type=='perio':
        # > Generate GridConnectivity between the two planes
        # Remark: former NGon is the first GridConnectivity and the duplicated one the second one
        zgc = PT.update_child(zone, 'ZoneGridConnectivity', 'ZoneGridConnectivity_t')
        distrib_idx = par_utils.uniform_distribution(n_cell, comm)
        pl1 = np.arange(distrib_idx[0], distrib_idx[1])+n_edges+1
        pl2 = np.arange(distrib_idx[0], distrib_idx[1])+n_edges+1+n_cell
        # TO DO: GC names ok ???
        #        PR or PL ???
        gc1_name = '__maia_former_plan'
        gc2_name = '__maia_extruded_plan'
        gc1 = PT.new_GridConnectivity(name=gc1_name, donor_name=f'{base_name}/{PT.get_name(zone)}',
                                      type='Abutting1to1', loc='FaceCenter',
                                      # point_range=[n_edges+1, n_edges+1+n_cell], 
                                      # point_range_donor=[n_edges+1+n_cell, n_edges+1+2*n_cell],
                                      point_list=[pl1],
                                      point_list_donor=[pl2],
                                      parent=zgc)
        PT.new_GridConnectivityProperty({"translation": np.array(extrusion_vector, dtype=np.float64)}, parent=gc1)
        MT.newDistribution({'Index' : distrib_idx}, parent=gc1)
        gc2 = PT.new_GridConnectivity(name=gc2_name, donor_name=f'{base_name}/{PT.get_name(zone)}',
                                      type='Abutting1to1', loc='FaceCenter',
                                      # point_range=[n_edges+1+n_cell, n_edges+1+2*n_cell],
                                      # point_range_donor=[n_edges+1, n_edges+1+n_cell], 
                                      point_list=[pl2],
                                      point_list_donor=[pl1],
                                      parent=zgc)
        PT.new_GridConnectivityProperty({"translation": -np.array(extrusion_vector, dtype=np.float64)}, parent=gc2)
        MT.newDistribution({'Index' : distrib_idx}, parent=gc2)
        # TO DO: to keep ???
        PT.new_child(gc1, "GridConnectivityDonorName", "Descriptor_t", gc2_name)
        PT.new_child(gc2, "GridConnectivityDonorName", "Descriptor_t", gc1_name)
    elif kplan_type=='fam_bc':
        zbc = PT.update_child(zone, 'ZoneBC', 'ZoneBC_t')
        distrib_idx = par_utils.uniform_distribution(n_cell, comm)
        pl_former   = np.arange(distrib_idx[0], distrib_idx[1])+n_edges+1
        pl_extruded = np.arange(distrib_idx[0], distrib_idx[1])+n_edges+1+n_cell
        bc1 = PT.new_BC(name='BC__maia_former_plan', type='FamilySpecified', point_list=[pl_former],
                        loc='FaceCenter', family='__maia_former_plan', parent=zbc)
        bc2 = PT.new_BC(name='BC__maia_extruded_plan', type='FamilySpecified', point_list=[pl_extruded],
                        loc='FaceCenter', family='__maia_extruded_plan', parent=zbc)
        MT.newDistribution({'Index' : distrib_idx}, parent=bc1)
        MT.newDistribution({'Index' : distrib_idx}, parent=bc2)
    else:
        raise RuntimeError(f"'kplan_type' is {kplan_type} but only 'perio' and 'fam_bc' are allowed !")


def _extrude_tri_to_prism_and_tris(tri, num, n_vtx, er_max, align=True):
    """
    Internal function used by _extrusion_2d_u_elem to create face by extrusion of TRI elements
    
    Remark : no need to change the element distribution
    """
    # > Copy of former quad node to generate new quad nodes
    new_tri1 = PT.deep_copy(tri)
    new_tri2 = PT.deep_copy(tri)
    # > Change value: 5 => 14
    PT.set_value(tri, [14, 0])
    # > Change name
    PT.set_name(tri, f'PENTA_6.{num}')
    # > Update ElementConnectivity
    ec_n = PT.get_child_from_name(tri, 'ElementConnectivity')
    first_nodes  = ec_n[1][0::3]
    second_nodes = ec_n[1][1::3]
    third_nodes  = ec_n[1][2::3]
    fourth_nodes = np.copy(ec_n[1][0::3]) + n_vtx
    fifth_nodes  = np.copy(ec_n[1][1::3]) + n_vtx
    sixth_nodes  = np.copy(ec_n[1][2::3]) + n_vtx
    ec_penta = np_utils.interweave_arrays([first_nodes, second_nodes, third_nodes, fourth_nodes, fifth_nodes, sixth_nodes])
    PT.set_value(ec_n, ec_penta)
    # > Treat new tri nodes
    # >>> Change ec of new TRIs with preservation of good orientation
    if align:
        new_tri1_ec = np_utils.interweave_arrays([first_nodes, third_nodes, second_nodes])
        new_tri2_ec = np_utils.interweave_arrays([fourth_nodes, fifth_nodes, sixth_nodes])
    else:
        new_tri1_ec = np_utils.interweave_arrays([first_nodes, second_nodes, third_nodes])
        new_tri2_ec = np_utils.interweave_arrays([fourth_nodes, sixth_nodes, fifth_nodes])
    new_tri1_ec_n = PT.get_child_from_name(new_tri1, 'ElementConnectivity')
    new_tri2_ec_n = PT.get_child_from_name(new_tri2, 'ElementConnectivity')
    PT.set_value(new_tri1_ec_n, new_tri1_ec)
    PT.set_value(new_tri2_ec_n, new_tri2_ec)
    # >>> Change name
    PT.set_name(new_tri1, f'TRI_3.{num}a')
    PT.set_name(new_tri2, f'TRI_3.{num}b')
    # >>> Change ER
    new_tri1_er_n = PT.get_child_from_name(new_tri1,"ElementRange")
    new_tri2_er_n = PT.get_child_from_name(new_tri2,"ElementRange")
    PT.set_value(new_tri1_er_n, np.array([er_max+1,                      er_max+PT.Element.Size(tri)],   dtype=pdm_dtype))
    PT.set_value(new_tri2_er_n, np.array([er_max+1+PT.Element.Size(tri), er_max+2*PT.Element.Size(tri)], dtype=pdm_dtype))
    return (new_tri1, new_tri2)


def _extrude_quad_to_hexa_and_quads(quad, num, n_vtx, er_max, align=True):
    """
    Internal function used by _extrusion_2d_u_elem to create face by extrusion of QUAD elements
    
    Remark : no need to change the element distribution
    """
    # > Copy of former quad node to generate new quad nodes
    new_quad1 = PT.deep_copy(quad)
    new_quad2 = PT.deep_copy(quad)
    # > Change value: 7 => 17
    PT.set_value(quad, [17, 0])
    # > Change name
    PT.set_name(quad, f'HEXA_8.{num}') #ou on s'appuie sur le nom initial de l'élément ?
    # > Update ElementConnectivity
    ec_n = PT.get_child_from_name(quad, 'ElementConnectivity')
    first_nodes   = ec_n[1][0::4]
    second_nodes  = ec_n[1][1::4]
    third_nodes   = ec_n[1][2::4]
    fourth_nodes  = ec_n[1][3::4]
    fifth_nodes   = np.copy(ec_n[1][0::4]) + n_vtx
    sixth_nodes   = np.copy(ec_n[1][1::4]) + n_vtx
    seventh_nodes = np.copy(ec_n[1][2::4]) + n_vtx
    eighth_nodes  = np.copy(ec_n[1][3::4]) + n_vtx
    ec_hexa = np_utils.interweave_arrays([first_nodes, second_nodes, third_nodes, fourth_nodes, fifth_nodes, sixth_nodes, seventh_nodes, eighth_nodes])
    PT.set_value(ec_n, ec_hexa)
    # > Treat new quad nodes
    # >>> Change ec of new QUADs with preservation of good orientation
    if align:
        new_quad1_ec = np_utils.interweave_arrays([first_nodes, fourth_nodes, third_nodes, second_nodes])
        new_quad2_ec = np_utils.interweave_arrays([fifth_nodes, sixth_nodes, seventh_nodes, eighth_nodes])
    else:
        new_quad1_ec = np_utils.interweave_arrays([first_nodes, second_nodes, third_nodes, fourth_nodes])
        new_quad2_ec = np_utils.interweave_arrays([fifth_nodes, eighth_nodes, seventh_nodes, sixth_nodes])
    new_quad1_ec_n = PT.get_child_from_name(new_quad1, 'ElementConnectivity')
    new_quad2_ec_n = PT.get_child_from_name(new_quad2, 'ElementConnectivity')
    PT.set_value(new_quad1_ec_n, new_quad1_ec)
    PT.set_value(new_quad2_ec_n, new_quad2_ec)
    # >>> Change name
    PT.set_name(new_quad1, f'QUAD_4.{num}a')
    PT.set_name(new_quad2, f'QUAD_4.{num}b')
    # >>> Change ER
    new_quad1_er_n = PT.get_child_from_name(new_quad1,"ElementRange")
    new_quad2_er_n = PT.get_child_from_name(new_quad2,"ElementRange")
    PT.set_value(new_quad1_er_n, np.array([er_max+1,                       er_max+PT.Element.Size(quad)],   dtype=pdm_dtype))
    PT.set_value(new_quad2_er_n, np.array([er_max+1+PT.Element.Size(quad), er_max+2*PT.Element.Size(quad)], dtype=pdm_dtype))
    return (new_quad1, new_quad2)


def _extrude_bar_to_quad(bar, num, n_vtx, align=True):
    """
    Internal function used by _extrusion_2d_u_elem to create face by extrusion of BAR elements
    
    Remark : no need to change the element distribution
    """
    # > Change value: 3 => 7
    PT.set_value(bar, [7, 0])
    # > Change name
    PT.set_name(bar, f'QUAD_4.{num}') #ou on s'appuie sur le nom initial de l'élément ?
    # > Update ElementConnectivity
    ec_n = PT.get_child_from_name(bar, 'ElementConnectivity')
    first_nodes  = ec_n[1][0::2]
    second_nodes = ec_n[1][1::2]
    third_nodes  = np.copy(ec_n[1][1::2]) + n_vtx
    fourth_nodes = np.copy(ec_n[1][0::2]) + n_vtx
    if align:
        new_ec = np_utils.interweave_arrays([first_nodes, second_nodes, third_nodes, fourth_nodes])
    else:
        new_ec = np_utils.interweave_arrays([second_nodes, first_nodes, fourth_nodes, third_nodes])
    PT.set_value(ec_n, new_ec)


def _extrusion_2d_u_elem(zone, base_name, extrusion_vector, comm, kplan_type='perio'):
    """
    Internal function used by extrusion_2d to extrude a 2D unstructured mesh describe by elements
    in the direction of the extrusion vector in cartesian and cylindrical coordinates.
    """
    
    # 0/ Global information
    n_vtx  = PT.Zone.n_vtx(zone)
    er_max = PT.Zone.get_max_elt_range(zone)
    
    # 1/ Duplication of nodes to generate the second plan
    _nodes_duplication(zone, extrusion_vector, comm)
    
    # 1bis/ Determine the mesh orientation
    align = _determine_mesh_orientation(zone, extrusion_vector, comm)
    
    # 2/ Extrude Tri to Prism
    is_tri = lambda n: (PT.get_label(n) == 'Elements_t') and (PT.get_value(n)[0] == 5) #On pourrait aussi utiliser le CGNSName pour plus de lisibilité
    new_tris_l = []
    for num, tri in enumerate(PT.get_nodes_from_predicate(zone, is_tri)):
        new_tri1, new_tri2 = _extrude_tri_to_prism_and_tris(tri, num, n_vtx, er_max, align=align)
        new_tris_l.append(new_tri1)
        new_tris_l.append(new_tri2)
        er_max += 2*PT.Element.Size(tri)
    for new_tri in new_tris_l:
        PT.add_child(zone, new_tri)
    
    # 3/ Extrude Quad to Hexa
    is_quad = lambda n: (PT.get_label(n) == 'Elements_t') and (PT.get_value(n)[0] == 7) #On pourrait aussi utiliser le CGNSName pour plus de lisibilité
    new_quads_l = []
    for num, quad in enumerate(PT.get_nodes_from_predicate(zone, is_quad)):
        new_quad1, new_quad2 = _extrude_quad_to_hexa_and_quads(quad, num, n_vtx, er_max, align=align)
        new_quads_l.append(new_quad1)
        new_quads_l.append(new_quad2)
        er_max += 2*PT.Element.Size(quad)
    for new_quad in new_quads_l:
        PT.add_child(zone, new_quad)
    
    # 4/ Extrude Bar to Quad
    is_bar = lambda n: (PT.get_label(n) == 'Elements_t') and (PT.get_value(n)[0] == 3) #On pourrait aussi utiliser le CGNSName pour plus de lisibilité
    for num, bar in enumerate(PT.get_nodes_from_predicate(zone, is_bar)):
        _extrude_bar_to_quad(bar, num, n_vtx, align=align)
    
    # 5/ Manage K-plans
    if kplan_type=='perio':
        # > Generate GridConnectivity between the two planes
        # Remark: former NGon is the first GridConnectivity and the duplicated one the second one
        # TO DO: GC names ok ???
        #        PR or PL ???
        zgc = PT.update_child(zone, 'ZoneGridConnectivity', 'ZoneGridConnectivity_t')
        for n, new_elem2d in enumerate(new_tris_l+new_quads_l):
            new_elem2d_range = PT.Element.Range(new_elem2d)
            if n%2==0:
                gc1_name = f'__maia_former_plan_{PT.get_name(new_elem2d)}'
                distrib1 = MT.getDistribution(new_elem2d, 'Element')[1]
                pl1 = np.arange(new_elem2d_range[0], new_elem2d_range[1]+1, dtype=pdm_dtype)[distrib1[0]:distrib1[1]]
                # pr1=[new_elem2d_range[0], new_elem2d_range[1]+1, dtype=pdm_dtype],
            else:
                gc2_name = f'__maia_extruded_plan_{PT.get_name(new_elem2d)}'
                distrib2 = MT.getDistribution(new_elem2d, 'Element')[1]
                pl2 = np.arange(new_elem2d_range[0], new_elem2d_range[1]+1, dtype=pdm_dtype)[distrib2[0]:distrib2[1]]
                # pr2=[new_elem2d_range[0], new_elem2d_range[1]+1, dtype=pdm_dtype],
                gc1 = PT.new_GridConnectivity(name=gc1_name, donor_name=f'{base_name}/{PT.get_name(zone)}',
                                              type='Abutting1to1', loc='FaceCenter',
                                              # point_range      =pr1,
                                              # point_range_donor=pr2,
                                              point_list=[pl1],
                                              point_list_donor=[pl2],
                                              parent=zgc)
                PT.new_GridConnectivityProperty({"translation": np.array(extrusion_vector, dtype=np.float64)}, parent=gc1)
                MT.newDistribution({'Index' : distrib1}, parent=gc1)
                gc2 = PT.new_GridConnectivity(name=gc2_name, donor_name=f'{base_name}/{PT.get_name(zone)}',
                                              type='Abutting1to1', loc='FaceCenter',
                                              # point_range      =pr2,
                                              # point_range_donor=pr1,
                                              point_list=[pl2],
                                              point_list_donor=[pl1],
                                              parent=zgc)
                PT.new_GridConnectivityProperty({"translation": -np.array(extrusion_vector, dtype=np.float64)}, parent=gc2)
                MT.newDistribution({'Index' : distrib2}, parent=gc2)
                # TO DO: to keep ???
                PT.new_child(gc1, "GridConnectivityDonorName", "Descriptor_t", gc2_name)
                PT.new_child(gc2, "GridConnectivityDonorName", "Descriptor_t", gc1_name)
    elif kplan_type=='fam_bc':
        zbc = PT.update_child(zone, 'ZoneBC', 'ZoneBC_t')
        for n, new_elem2d in enumerate(new_tris_l+new_quads_l):
            if n%2==0:
                bc_name = f'BC__maia_former_plan_{PT.get_name(new_elem2d)}'
                fam_name = '__maia_former_plan'
            else:
                bc_name = f'BC__maia_extruded_plan{PT.get_name(new_elem2d)}'
                fam_name = '__maia_extruded_plan'
            distrib = MT.getDistribution(new_elem2d, 'Element')[1]
            new_elem2d_range = PT.Element.Range(new_elem2d)
            pl = np.arange(new_elem2d_range[0], new_elem2d_range[1]+1, dtype=pdm_dtype)[distrib[0]:distrib[1]]
            bc = PT.new_BC(name=bc_name, type='FamilySpecified', point_list=[pl],
                           loc='FaceCenter', family=fam_name, parent=zbc)
            MT.newDistribution({'Index' : distrib}, parent=bc)
    else:
        raise RuntimeError(f"'kplan_type' is {kplan_type} but only 'perio' and 'fam_bc' are allowed !")

def _pl_and_data_vtx_duplication(pl, distrib_idx, n_vtx_2d, data, comm):
    """
    Internal function used by _extrusion_2d to create the duplicated PointList 
    and associated datas needed to extented it to the second plan
    """
    new_distrib_idx  = par_utils.uniform_distribution(distrib_idx[2]*2,  comm)
    # > Duplicate data in part_data
    if pl is None:
        part_data = {}
    else:
        part_data = {'PointList': [pl[0], pl[0]+n_vtx_2d]}
    for name, value in data.items():
       part_data[name] = [value, value]
    # > Compute ln_to_gn
    ln_to_gn_l = []
    ln_to_gn_l.append(np.arange(distrib_idx[0], distrib_idx[1])+1)
    ln_to_gn_l.append(np.arange(distrib_idx[0]+distrib_idx[2], distrib_idx[1]+distrib_idx[2])+1)
    # > Part to block
    dist_data = EP.part_to_block(part_data, new_distrib_idx, ln_to_gn_l, comm)
    # > Return
    return(new_distrib_idx, dist_data)
    

def extrusion_2d(dist_tree, extrusion_vector, comm, kplan_type='perio', dupl_vtx_info=False):
    """
    Extrude a 2D mesh in the direction of the extrusion vector in cartesian and cylindrical coordinates.

    Input tree is modified inplace.
  
    Args:
      dist_tree (CGNSTree): Input distributed tree
      comm      (MPIComm) : MPI communicator
      extrusion_vector (array of 3 floats): List of the value of the extrusion in each direction
      kplan_type (str): Option to define K plans as periodic GridConnectivity_t ('perio')
                           or has FamilySpecified BC_t ('fam_bc'). Default value is 'perio'
      dupl_vtx_info (str): Option to define how to manage 'Vertex' information when extruded. Keep
                           information on initial vertices only (False) or duplicate it on extuded
                           plan (True). Default value is 'False'
          
    Warning: For now, meshes with only BAR elements with ParentElements are not managed
    
    TO DO
    > ajouter exemple/snippet dans la doc ?
    """
    
    for base in PT.get_all_CGNSBase_t(dist_tree):
        # Test if the mesh is 2D
        assert PT.get_value(base)[0] == 2
        
        # Save data
        zone_to_distrib_vtx={}
        for zone in PT.get_children_from_label(base, 'Zone_t'):
            distrib_vtx_2d_n = PT.deep_copy(MT.getDistribution(zone, 'Vertex'))
            zone_to_distrib_vtx[f'{base[0]}/{zone[0]}'] = PT.get_value(distrib_vtx_2d_n)

    for base in PT.get_all_CGNSBase_t(dist_tree):
        
        for zone in PT.get_children_from_label(base, 'Zone_t'):
            
            distrib_vtx_2d = zone_to_distrib_vtx[f'{base[0]}/{zone[0]}']

            # Add third coordinate if needed
            if PT.get_value(base)[1] == 2:
                coord_n = PT.get_child_from_label(zone, 'GridCoordinates_t')
                coords = PT.Zone.coordinates(zone)
                distrib_vtx = MT.getDistribution(zone, 'Vertex')[1]
                n_vtx_loc = distrib_vtx[1] - distrib_vtx[0]
                for cn, coord_name in enumerate(coords._fields):
                    if coords[cn] is None:
                        PT.new_child(coord_n, coord_name, 'DataArray_t', np.zeros((n_vtx_loc), dtype=np.float64))
            
            # Generate new vertices and Elements
            if PT.Zone.Type(zone) == 'Structured':
                raise NotImplementedError('Extrusion of 2D structured meshes is not yet implemented !')
            elif PT.Zone.Type(zone) == 'Unstructured':
                all_element_types = np.unique([PT.get_value(elem_n)[0] for elem_n in PT.get_children_from_label(zone, 'Elements_t')])
                if np.all(np.isin(all_element_types, [2, 3, 22])):
                    _extrusion_2d_u_ngon(zone, base[0], extrusion_vector, comm, kplan_type=kplan_type)
                elif np.all(np.isin(all_element_types, [2, 3, 5, 7])):
                    _extrusion_2d_u_elem(zone, base[0], extrusion_vector, comm, kplan_type=kplan_type)
                else:
                    raise ValueError(f'Zone "{PT.get_name(zone)}" is neither full Ngon or composed only of TRI and QUAD !')
            else:
                raise ValueError(f'Zone "{PT.get_name(zone)}" is neither structured nor unstructured !')
    
            # Update zone dims
            # Remark: in extrusion, no need to change nb_cell because the new 3D cells are the 
            #         former 2D ones extruded
            zone_dims = PT.get_value(zone)
            zone_dims[0][0] = 2*PT.Zone.n_vtx(zone)
            PT.set_value(zone, zone_dims)
            
            # Update containers
            # > CellCenter
            # Remark: nothing to do
            # > FaceCenter
            # Remark: in 2D, 'FaceCenter' must not exist in 'GridLocation' => will be deleted !
            # Reflexion : ou on les convertit en CellCenter ou on lève une erreur ?
            is_container_face_center = lambda n: (PT.get_label(n) in ['FlowSolution_t', 'DiscreteData_t',
                                                                      'ZoneSubRegion_t', 'BCDataSet_t']) \
                                       and (PT.Subset.GridLocation(n) == 'FaceCenter')
            container_face_l = [PT.get_name(container_face) for container_face in PT.get_children_from_predicate(zone, is_container_face_center)]
            if len(container_face_l)>0:
                print(f"Warning: The 2D mesh have 'FaceCenter' containers, that is not allowed by CGNS norm and so {container_face_l} will be deleted ! ")
                for container_name in container_face_l:
                    PT.rm_node_from_path(zone, container_name)
            # > EdgeCenter
            # Remark: EdgeCenter in 2D become FaceCenter in 3D
            is_container_edge_center = lambda n: (PT.get_label(n) in ['FlowSolution_t', 'DiscreteData_t',
                                                                      'ZoneSubRegion_t', 'BCDataSet_t']) \
                                             and (PT.Subset.GridLocation(n) == 'EdgeCenter')
            for container in PT.get_nodes_from_predicate(zone, is_container_edge_center):
                gl_n = PT.get_child_from_name(container, 'GridLocation')
                PT.set_value(gl_n, 'FaceCenter')
            # > Vertex
            # Remark: no conversion to EdgeCenter (cf. 'dupl_vtx_info')
            is_container_vertex_with_pl = lambda n: (PT.get_label(n) in ['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t']) \
                                                and (PT.Subset.GridLocation(n) == 'Vertex') \
                                                and (PT.get_node_from_name(n, 'PointList') is not None)
            is_fs_and_dd_vertex_wo_pl = lambda n: (PT.get_label(n) in ['FlowSolution_t', 'DiscreteData_t']) \
                                              and (PT.Subset.GridLocation(n) == 'Vertex') \
                                              and (PT.get_node_from_name(n, 'PointList') is None)
            is_zsr_vertex_wo_pl = lambda n: (PT.get_label(n) == 'ZoneSubRegion_t') \
                                        and (PT.Subset.GridLocation(n) == 'Vertex') \
                                        and (PT.get_node_from_name(n, 'PointList') is None)
            if dupl_vtx_info:
                for container in PT.get_children_from_predicate(zone, is_container_vertex_with_pl):
                    pl = PT.get_child_from_name(container, 'PointList')
                    distrib_idx = MT.getDistribution(container, 'Index')
                    data = {}
                    for da in PT.get_children_from_label(container, 'DataArray_t'):
                        data[da[0]] = da[1]
                    new_distrib_idx, new_data = _pl_and_data_vtx_duplication(pl[1], distrib_idx[1], distrib_vtx_2d[2], data, comm)
                    PT.set_value(pl, [new_data['PointList']])
                    PT.set_value(distrib_idx, new_distrib_idx)
                    for data_name in data.keys():
                        PT.set_value(PT.get_child_from_name(container, data_name), new_data[data_name])
                for container in PT.get_children_from_predicate(zone, is_fs_and_dd_vertex_wo_pl):
                    data = {}
                    for da in PT.get_children_from_label(container, 'DataArray_t'):
                        data[da[0]] = da[1]
                    _, new_data = _pl_and_data_vtx_duplication(None, distrib_vtx_2d, distrib_vtx_2d[2], data, comm)
                    for data_name in data.keys():
                        PT.set_value(PT.get_child_from_name(container, data_name), new_data[data_name])
                for container in PT.get_children_from_predicate(zone, is_zsr_vertex_wo_pl):
                    zsr_extent = PT.Subset.ZSRExtent(container, zone)
                    extent_node = PT.get_node_from_path(zone, zsr_extent)
                    distrib_idx = MT.getDistribution(extent_node, 'Index')
                    data = {}
                    for da in PT.get_children_from_label(container, 'DataArray_t'):
                        data[da[0]] = da[1]
                    _, new_data = _pl_and_data_vtx_duplication(None, distrib_idx[1], distrib_vtx_2d[2], data, comm)
                    for data_name in data.keys():
                        PT.set_value(PT.get_child_from_name(container, data_name), new_data[data_name])
                # Reflexion: doit-on faire de même pour pour des BCDataSet sous une GC même si ce n'est pas la norme ?
                for _, bc, bcds in PT.get_nodes_from_predicates(zone, 'ZoneBC_t/BC_t/BCDataSet_t', ancestors=True):
                    if PT.Subset.GridLocation(bcds) == 'Vertex':
                        data = {}
                        for bcd, da in PT.get_nodes_from_predicates(bcds, 'BCData_t/DataArray_t', ancestors=True):
                            data[f'{bcds[0]}/{bcd[0]}/{da[0]}'] = da[1]
                        bcds_pl = PT.get_child_from_name(bcds, 'PointList')
                        if bcds_pl is None:
                            pl = PT.get_child_from_name(bc, 'PointList')
                            distrib_idx = MT.getDistribution(bc, 'Index')
                        else:
                            pl = bcds_pl
                            distrib_idx = MT.getDistribution(bcds, 'Index')
                        new_distrib_idx, new_data = _pl_and_data_vtx_duplication(pl[1], distrib_idx[1], distrib_vtx_2d[2], data, comm)
                        if bcds_pl is not None:
                            PT.set_value(pl, [new_data['PointList']])
                            PT.set_value(distrib_idx, new_distrib_idx)
                        for bcd, da in PT.get_nodes_from_predicates(bcds, 'BCData_t/DataArray_t', ancestors=True):
                            PT.set_value(da, new_data[f'{bcds[0]}/{bcd[0]}/{da[0]}'])
            else:
                for container in PT.get_children_from_predicate(zone, is_fs_and_dd_vertex_wo_pl):
                    PT.new_IndexArray('PointList', value=[np.arange(distrib_vtx_2d[0], distrib_vtx_2d[1], dtype=pdm_dtype)+1], parent=container)
                    MT.newDistribution({'Index': distrib_vtx_2d}, parent=container)
                for container in PT.get_children_from_predicate(zone, is_zsr_vertex_wo_pl):
                    zsr_extent = PT.Subset.ZSRExtent(container, zone)
                    extent_node = PT.get_node_from_path(zone, zsr_extent)
                    PT.add_child(container, PT.deep_copy(PT.get_child_from_name(extent_node, 'PointList')))
                    PT.add_child(container, PT.deep_copy(PT.get_child_from_name(extent_node, ':CGNS#Distribution')))
                # Reflexion: doit-on faire de même pour pour des BCDataSet sous une GC même si ce n'est pas la norme ?
                for _, bc, bcds in PT.get_nodes_from_predicates(zone, 'ZoneBC_t/BC_t/BCDataSet_t', ancestors=True):
                    if (PT.Subset.GridLocation(bcds) == 'Vertex') and (PT.get_node_from_name(bcds, 'PointList') is None):
                        PT.add_child(bcds, PT.deep_copy(PT.get_child_from_name(bc, 'PointList')))
                        PT.add_child(bcds, PT.deep_copy(PT.get_child_from_name(bc, ':CGNS#Distribution')))
            
            # Update subsets
            # > FaceCenter
            # Remark: in 2D, 'FaceCenter' must not exist in 'GridLocation' => will be deleted !
            is_face_center = lambda n: (PT.get_label(n) in ['BC_t', 'GridConnectivity_t', 'GridConnectivity_1to1_t']) \
                                   and (PT.Subset.GridLocation(n) == 'FaceCenter')
            subset_face_l = [PT.get_name(subset_face) for subset_face in PT.get_children_from_predicate(zone, is_face_center)]
            if len(subset_face_l)>0:
                raise RuntimeError(f"Error: The 2D mesh have 'FaceCenter' BC or GC, that is not allowed by CGNS norm !")
            # > EdgeCenter
            # Remark: no need to change their PointList
            is_edge_center = lambda n: (PT.get_label(n) in ['BC_t', 'GridConnectivity_t', 'GridConnectivity_1to1_t']) \
                                   and (PT.Subset.GridLocation(n) == 'EdgeCenter')
            for subset_edge in PT.get_nodes_from_predicate(zone, is_edge_center):
                gl_n = PT.get_child_from_name(subset_edge, 'GridLocation')
                PT.set_value(gl_n, 'FaceCenter')
            # > Vertex
            # Remark : we have to add the duplicated nodes from PL vertices to PL
            is_vertex_center = lambda n: (PT.get_label(n) in ['BC_t', 'GridConnectivity_t', 'GridConnectivity_1to1_t']) \
                                   and (PT.Subset.GridLocation(n) == 'Vertex')
            for subset_vertex in PT.get_nodes_from_predicate(zone, is_vertex_center):
                pl = PT.get_child_from_name(subset_vertex, 'PointList')
                distrib_idx = MT.getDistribution(subset_vertex, 'Index')
                new_distrib_idx, new_data = _pl_and_data_vtx_duplication(pl[1], distrib_idx[1], distrib_vtx_2d[2], {}, comm)
                # Manage PointListDonor
                pld = PT.get_child_from_name(subset_vertex, 'PointListDonor')
                if pld is not None:
                    pld_value_split = PT.get_value(subset_vertex).split('/')
                    if len(pld_value_split) == 1: #only zone name
                        opp_zone_path = f'{base[0]}/{PT.get_value(subset_vertex)}'
                    elif len(pld_value_split) == 2: #base name and zone name
                        opp_zone_path = PT.get_value(subset_vertex)
                    else:
                        raise RuntimeError(f'The value of GC {subset_vertex[0]} is not CGNS compliant !')
                    distrib_vtx_2d_opp = zone_to_distrib_vtx[opp_zone_path]
                    _, new_data_pld = _pl_and_data_vtx_duplication(pld[1], distrib_idx[1], distrib_vtx_2d_opp[2], {}, comm)
                    PT.set_value(pld, [new_data_pld['PointList']])
                # Update PointList and Distribution
                PT.set_value(pl, [new_data['PointList']])
                PT.set_value(distrib_idx, new_distrib_idx)
    
    # Update base dimension
    for base in PT.get_all_CGNSBase_t(dist_tree):
        PT.set_value(base, [3, 3])
        if kplan_type=='fam_bc':
            PT.new_Family('__maia_former_plan',   family_bc='UserDefined', parent=base)
            PT.new_Family('__maia_extruded_plan', family_bc='UserDefined', parent=base)
