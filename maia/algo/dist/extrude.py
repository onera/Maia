from mpi4py import MPI
import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia          import npy_pdm_gnum_dtype  as pdm_dtype
from maia.transfer import protocols           as EP
from maia.utils    import np_utils, par_utils
from maia.utils    import logging as mlog

is_bar = lambda n: PT.get_label(n) == 'Elements_t' and PT.Element.CGNSName(n) == 'BAR_2'

def _nodes_duplication(zone, extrusion_vector, comm):
    """
    Internal function used by _extrusion_2d_u_ngon and _extrusion_2d_u_elem to create 
    the duplicated nodes needed to generate the second plan
    """
    # > Define new distribution
    distrib_vtx_n    = MT.getDistribution(zone, 'Vertex')
    distrib_vtx      = PT.get_value(distrib_vtx_n)
    new_distrib_vtx  = par_utils.uniform_distribution(2*distrib_vtx[2],  comm)
    # > Change value of Z or Theta coordinates in part_data
    coords = PT.Zone.coordinates(zone)
    part_data = {name : [coord, coord+extru] for extru, (name, coord) in zip(extrusion_vector, coords._asdict().items())}
    # > Compute ln_to_gn
    ln_to_gn_l = [np.arange(distrib_vtx[0]+1, distrib_vtx[1]+1, dtype=pdm_dtype),
                  np.arange(distrib_vtx[0]+1, distrib_vtx[1]+1, dtype=pdm_dtype) + distrib_vtx[2]]
    # > Part to block
    dist_data = EP.part_to_block(part_data, new_distrib_vtx, ln_to_gn_l, comm)
    # > Update coordinates values
    for name, new_val in dist_data.items():
        coord_n = PT.get_child_from_predicates(zone, f'GridCoordinates_t/{name}')
        PT.set_value(coord_n, new_val)
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
    if len(first_elem_2d_ec) > 0:
        nodes_of_first_face = first_elem_2d_ec[0:3]
    else:
        nodes_of_first_face = np.empty(0, int)
    # > Get coordinates of nodes of the first face
    coords = PT.Zone.coordinates(zone)
    dist_coords_data = coords._asdict()
    distrib_vtx = MT.getDistribution(zone, 'Vertex')[1]
    part_coords_data = EP.block_to_part(dist_coords_data, distrib_vtx, [nodes_of_first_face], comm)
    align = 0
    # > Compute scalar product
    # **NB** Here we test only one face per rank. Maybe we should do it on all faces and check that is same everywhere ?
    #        We need compute_normals to be implemented for elts to do this
    if len(nodes_of_first_face) > 0:
        n1 = np.array([part_coords_data[name][0][0] for name in coords._fields])
        n2 = np.array([part_coords_data[name][0][1] for name in coords._fields])
        n3 = np.array([part_coords_data[name][0][2] for name in coords._fields])
        # normal vector = a^b with a = n1_n2 and b = n2_n3
        normal_vec = np.cross(n2-n1, n3-n2)
        # ps = c.extrusion_vector
        ps = np.dot(normal_vec, extrusion_vector)
        # >>> d. change order of one of two
        if ps>0: # Need to change initial faces
            align = 1
        elif ps<0: # Need to change duplicated faces
            align = -1
    if comm.allreduce(align >= 0, MPI.LAND):
        return True
    elif comm.allreduce(align <= 0, MPI.LAND):
        return False
    else:
        raise ValueError('Faces of 2D mesh are not uniformly oriented !')


def _reorder_ngon_ec(ngon_n):
    """
    Internal function used by _ngon_duplication to reorder the NGonNode connectivity
    to be exited normal
    """
    ec  = PT.get_child_from_name(ngon_n, 'ElementConnectivity')[1]
    eso = PT.get_child_from_name(ngon_n, 'ElementStartOffset')[1]
    
    np_utils.reverse_by_stride(eso-eso[0], ec, inplace=True)
    

def _ngon_duplication(zone, comm, align=True):
    """
    Internal function used by _extrusion_2d_u_ngon to create the duplicated NGonNode
    needed to generate the second plan
    
    Remark : no need to change the ElementStartOffset
    """
    # > Get number of 2D cells (face) and vertices
    n_cell_2d = PT.Zone.n_cell(zone)
    n_vtx_2d  = PT.Zone.n_vtx(zone)
    # > Add ParentElements to NGon node
    ngon_n = PT.Zone.NGonNode(zone)
    distrib_elem = MT.getDistribution(ngon_n, 'Element')[1]
    er = PT.Element.Range(ngon_n)
    pe = np.zeros((distrib_elem[1]-distrib_elem[0],2), dtype=er.dtype)
    pe[:,0] = np.arange(distrib_elem[0], distrib_elem[1]) + er[0] + 2*n_cell_2d
    PT.new_child(ngon_n, 'ParentElements', 'DataArray_t', pe)
    # > Duplicate NGon
    ngon_bis_n = PT.deep_copy(ngon_n)
    PT.set_name(ngon_bis_n, f'{PT.get_name(ngon_n)}_bis')
    # > Update ElementRange and ElementConnectivity
    # on suppose que l'on a deja tous les elements 1D et 2D de définis dans le CGNS
    # TO DO: creer le NGon si on a que Bar + PE avant d'appliquer cette fonction
    er = PT.get_child_from_name(ngon_bis_n, 'ElementRange')[1]
    ec = PT.get_child_from_name(ngon_bis_n, 'ElementConnectivity')[1]
    er += n_cell_2d
    ec += n_vtx_2d
    if align: # Need to change Ngon
        _reorder_ngon_ec(ngon_n)
    else: # Need to change duplicated Ngon
        _reorder_ngon_ec(ngon_bis_n)

    PT.add_child(zone, ngon_bis_n)


def _extrude_bar_to_ngon(bar, n_vtx, n_cell, align=True):
    """
    Internal function used by _extrusion_2d_u_ngon to create face by extrusion of BAR elements
    """
    distrib_elem = MT.getDistribution(bar, 'Element')[1]
    ec_n = PT.get_child_from_name(bar, 'ElementConnectivity')

    # > Change value: 3 => 22
    PT.get_value(bar)[0] = 22
    # > Create ElementStartOffset
    eso = 4*np.arange(distrib_elem[0], distrib_elem[1]+1, dtype=ec_n[1].dtype)
    PT.new_child(bar, 'ElementStartOffset', 'DataArray_t', eso)
    # > Update ElementConnectivity
    first_nodes  = ec_n[1][0::2]
    second_nodes = ec_n[1][1::2]
    third_nodes  = second_nodes + n_vtx
    fourth_nodes = first_nodes  + n_vtx
    if align:
        new_ec = np_utils.interweave_arrays([second_nodes, first_nodes, fourth_nodes, third_nodes])
    else:
        new_ec = np_utils.interweave_arrays([first_nodes, second_nodes, third_nodes, fourth_nodes])
    PT.set_value(ec_n, new_ec)
    # > Create ElementConnectivity distribution
    MT.newDistribution({'ElementConnectivity' : 4*distrib_elem}, parent=bar)
    # > Update PE
    # Remark: new cells are the former faces because we keep the order
    #         so we just have to shift there values of 2*n_cell
    pe = PT.get_child_from_name(bar, 'ParentElements')[1]
    pe += 2*n_cell * (pe > 0)
    
    
def _merge_ngons(zone, comm):
    """ Internal function used by _extrusion_2d_u_ngon to create the merged NGonNode """
    part_diff_eso = []
    part_ec = []
    part_pe0 = []
    part_pe1 = []
    ln_to_gn_elem_l = []
    ngon_nodes = PT.Zone.get_ordered_elements_per_dim(zone)[2]
    for ngon_n in ngon_nodes:
        assert PT.Element.CGNSName(ngon_n) == 'NGON_n'
        er  = PT.get_child_from_name(ngon_n, 'ElementRange')[1]
        ec  = PT.get_child_from_name(ngon_n, 'ElementConnectivity')[1]
        eso = PT.get_child_from_name(ngon_n, 'ElementStartOffset')[1]
        pe  = PT.get_child_from_name(ngon_n, 'ParentElements')[1]
        distrib_elem = MT.getDistribution(ngon_n, 'Element')[1]
        part_diff_eso.append(np.diff(eso).astype(np.int32))
        part_ec.append(ec)
        part_pe0.append(pe[:,0])
        part_pe1.append(pe[:,1])
        ln_to_gn_elem_l.append(np.arange(distrib_elem[0]+er[0], distrib_elem[1]+er[0], dtype=pdm_dtype))
    # > Define new ElementRange
    # Warning : ne fonctionne pas si il y a des 'NODE' dans l'arbre !
    #           si tous les NODE sont avant, il faut faire démarrer l'ER au total des éléments de NODE
    #           si tous les NODE sont après, rien à changer
    #           si les NODE ne sont pas spécialement ordonnés, faire une réorganisation en entrée de 'extrusion_2d' ?
    n_faces = sum(PT.Element.Size(ng) for ng in ngon_nodes)
    new_er = np.array([1, n_faces], zone[1].dtype)
    # > Define new Element distribution
    new_distrib_elem = par_utils.uniform_distribution(n_faces,  comm)
    # > Exchange to define new ElementStartOffset, ElementConnectivity and ParentElements
    ptb = EP.PartToBlock(new_distrib_elem, ln_to_gn_elem_l, comm)
    new_diff_eso, new_ec = ptb.exchange_field(part_ec, part_diff_eso)
    _, new_pe0 = ptb.exchange_field(part_pe0)
    _, new_pe1 = ptb.exchange_field(part_pe1)
    new_pe = np.empty((len(new_pe0), 2), order='F', dtype=zone[1].dtype)
    new_pe[:,0]  = new_pe0
    new_pe[:,1]  = new_pe1
    # > Define new ElementConnectivity distribution
    new_distrib_ec = par_utils.dn_to_distribution(new_diff_eso.sum(), comm)
    new_eso = np_utils.sizes_to_indices(new_diff_eso) + new_distrib_ec[0]
    # > Delete old ngons
    PT.rm_children_from_predicate(zone, lambda n: PT.get_label(n) == 'Elements_t' and PT.Element.CGNSName(n) == 'NGON_n')
    # > Create new NGon node
    new_ngon_n = PT.new_NGonElements(erange=new_er, eso=new_eso, ec=new_ec, pe=new_pe, parent=zone)
    MT.newDistribution({'Element' : new_distrib_elem, 'ElementConnectivity' : new_distrib_ec}, parent=new_ngon_n)
    

def _extrusion_2d_u_ngon(zone, extrusion_vector, comm, kplan_type='perio'):
    """
    Internal function used by extrusion_2d to extrude a 2D unstructured mesh describe by edges
    in the direction of the extrusion vector in cartesian and cylindrical coordinates.
    """
    
    # 0/ Global information
    n_vtx  = PT.Zone.n_vtx(zone)
    n_cell = PT.Zone.n_cell(zone)
    # TO DO: create function in 'node_inspect.py' to obtain edges number of a mesh ?
    n_edges = sum(PT.Element.Size(e) for e in PT.get_children_from_predicate(zone, is_bar))
    
    # 1/ Duplication of nodes to generate the second plan
    _nodes_duplication(zone, extrusion_vector, comm)
    
    # 1bis/ Determine the mesh orientation
    align = _determine_mesh_orientation(zone, extrusion_vector, comm)
    
    # 2/ Create faces of the second plan
    align = _ngon_duplication(zone, comm, align)
    
    # 3/ Extrude Bar to NGon
    for bar in PT.get_nodes_from_predicate(zone, is_bar):
        _extrude_bar_to_ngon(bar, n_vtx, n_cell, align)
    
    # 4/ Merge all NGon nodes
    _merge_ngons(zone, comm)
    
    # 5/ Manage K-plans
    # For now use PointList even if data is contiguous
    # Note : Subset are created as EdgeCenter right now, because they calling function convert it to FaceCenter after
    distrib_idx = par_utils.uniform_distribution(n_cell, comm)
    pl_former   = np.arange(distrib_idx[0], distrib_idx[1], dtype=zone[1].dtype).reshape((1,-1), order='F') + n_edges + 1
    pl_extruded = np.arange(distrib_idx[0], distrib_idx[1], dtype=zone[1].dtype).reshape((1,-1), order='F') + n_edges + 1 + n_cell

    if kplan_type=='perio':
        # > Generate GridConnectivity between the two planes
        # Remark: former NGon is the first GridConnectivity and the duplicated one the second one
        zgc = PT.update_child(zone, 'ZoneGridConnectivity', 'ZoneGridConnectivity_t')
        # TO DO: GC names ok ???
        gc1_name = '__maia_former_plan'
        gc2_name = '__maia_extruded_plan'
        gc1 = PT.new_GridConnectivity(name=gc1_name, donor_name=PT.get_name(zone),
                                      type='Abutting1to1', loc='EdgeCenter',
                                      # point_range=[n_edges+1, n_edges+1+n_cell], 
                                      # point_range_donor=[n_edges+1+n_cell, n_edges+1+2*n_cell],
                                      point_list=pl_former,
                                      point_list_donor=pl_extruded,
                                      parent=zgc)
        PT.new_GridConnectivityProperty({"translation": np.array(extrusion_vector, dtype=np.float64)}, parent=gc1)
        MT.newDistribution({'Index' : distrib_idx}, parent=gc1)

        gc2 = PT.new_GridConnectivity(name=gc2_name, donor_name=PT.get_name(zone),
                                      type='Abutting1to1', loc='EdgeCenter',
                                      # point_range=[n_edges+1+n_cell, n_edges+1+2*n_cell],
                                      # point_range_donor=[n_edges+1, n_edges+1+n_cell], 
                                      point_list=pl_extruded,
                                      point_list_donor=pl_former,
                                      parent=zgc)
        PT.new_GridConnectivityProperty({"translation": -np.array(extrusion_vector, dtype=np.float64)}, parent=gc2)
        MT.newDistribution({'Index' : distrib_idx}, parent=gc2)

        PT.new_Descriptor("GridConnectivityDonorName", gc2_name, parent=gc1)
        PT.new_Descriptor("GridConnectivityDonorName", gc1_name, parent=gc2)

    elif kplan_type=='fam_bc':
        zbc = PT.update_child(zone, 'ZoneBC', 'ZoneBC_t')
        bc1 = PT.new_BC(name='BC__maia_former_plan', type='FamilySpecified', point_list=pl_former,
                        loc='EdgeCenter', family='__maia_former_plan', parent=zbc)
        bc2 = PT.new_BC(name='BC__maia_extruded_plan', type='FamilySpecified', point_list=pl_extruded,
                        loc='EdgeCenter', family='__maia_extruded_plan', parent=zbc)
        MT.newDistribution({'Index' : distrib_idx}, parent=bc1)
        MT.newDistribution({'Index' : distrib_idx}, parent=bc2)
    else:
        raise RuntimeError(f"'kplan_type' is {kplan_type} but only 'perio' and 'fam_bc' are allowed !")


def _extrude_tri_to_prism_and_tris(tri, num, n_vtx, er_max, align=True):
    """
    Internal function used by _extrusion_2d_u_elem to create face by extrusion of TRI elements
    
    Remark : no need to change the element distribution
    """
    # > Change value: 5 => 14
    PT.get_value(tri)[0] = 14
    # > Change name
    PT.set_name(tri, f'PENTA_6.{num}')
    # > Update ElementConnectivity
    ec_n = PT.get_child_from_name(tri, 'ElementConnectivity')
    first_nodes  = ec_n[1][0::3]
    second_nodes = ec_n[1][1::3]
    third_nodes  = ec_n[1][2::3]
    fourth_nodes = first_nodes  + n_vtx
    fifth_nodes  = second_nodes + n_vtx
    sixth_nodes  = third_nodes  + n_vtx
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
    
    new_tri1_er = np.array([er_max+1, er_max+PT.Element.Size(tri)], new_tri1_ec.dtype)
    new_tri2_er = new_tri1_er + PT.Element.Size(tri)

    new_tri1 = PT.new_Elements(f'TRI_3.{num}a', 'TRI_3', erange=new_tri1_er, econn=new_tri1_ec)
    new_tri2 = PT.new_Elements(f'TRI_3.{num}b', 'TRI_3', erange=new_tri2_er, econn=new_tri2_ec)
    MT.newDistribution({'Element' : MT.getDistribution(tri, 'Element')[1].copy()}, new_tri1)
    MT.newDistribution({'Element' : MT.getDistribution(tri, 'Element')[1].copy()}, new_tri2)
    return (new_tri1, new_tri2)


def _extrude_quad_to_hexa_and_quads(quad, num, n_vtx, er_max, align=True):
    """
    Internal function used by _extrusion_2d_u_elem to create face by extrusion of QUAD elements
    
    Remark : no need to change the element distribution
    """
    # > Change value: 7 => 17
    PT.get_value(quad)[0] = 17
    # > Change name
    PT.set_name(quad, f'HEXA_8.{num}') #ou on s'appuie sur le nom initial de l'élément ?
    # > Update ElementConnectivity
    ec_n = PT.get_child_from_name(quad, 'ElementConnectivity')
    first_nodes   = ec_n[1][0::4]
    second_nodes  = ec_n[1][1::4]
    third_nodes   = ec_n[1][2::4]
    fourth_nodes  = ec_n[1][3::4]
    fifth_nodes   = first_nodes  + n_vtx
    sixth_nodes   = second_nodes + n_vtx
    seventh_nodes = third_nodes  + n_vtx
    eighth_nodes  = fourth_nodes + n_vtx
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

    new_quad1_er = np.array([er_max+1, er_max+PT.Element.Size(quad)], new_quad1_ec.dtype)
    new_quad2_er = new_quad1_er + PT.Element.Size(quad)

    new_quad1 = PT.new_Elements(f'QUAD_4.{num}a', 'QUAD_4', erange=new_quad1_er, econn=new_quad1_ec)
    new_quad2 = PT.new_Elements(f'QUAD_4.{num}b', 'QUAD_4', erange=new_quad2_er, econn=new_quad2_ec)
    MT.newDistribution({'Element' : MT.getDistribution(quad, 'Element')[1].copy()}, new_quad1)
    MT.newDistribution({'Element' : MT.getDistribution(quad, 'Element')[1].copy()}, new_quad2)
    return (new_quad1, new_quad2)

def _extrude_bar_to_quad(bar, num, n_vtx, align=True):
    """
    Internal function used by _extrusion_2d_u_elem to create face by extrusion of BAR elements
    
    Remark : no need to change the element distribution
    """
    # > Change value: 3 => 7
    PT.get_value(bar)[0] = 7
    # > Change name
    PT.set_name(bar, f'QUAD_4.{num}')
    # > Update ElementConnectivity
    ec_n = PT.get_child_from_name(bar, 'ElementConnectivity')
    first_nodes  = ec_n[1][0::2]
    second_nodes = ec_n[1][1::2]
    third_nodes  = second_nodes + n_vtx
    fourth_nodes = first_nodes  + n_vtx
    if align:
        new_ec = np_utils.interweave_arrays([first_nodes, second_nodes, third_nodes, fourth_nodes])
    else:
        new_ec = np_utils.interweave_arrays([second_nodes, first_nodes, fourth_nodes, third_nodes])
    PT.set_value(ec_n, new_ec)


def _extrusion_2d_u_elem(zone, extrusion_vector, comm, kplan_type='perio'):
    """
    Internal function used by extrusion_2d to extrude a 2D unstructured mesh describe by elements
    in the direction of the extrusion vector in cartesian and cylindrical coordinates.
    """
    
    # 0/ Global information
    n_vtx  = PT.Zone.n_vtx(zone)
    er_max = max(PT.Element.Range(e)[1] for e in PT.get_children_from_label(zone, 'Elements_t'))

    # 1/ Duplication of nodes to generate the second plan
    _nodes_duplication(zone, extrusion_vector, comm)
    
    # 1bis/ Determine the mesh orientation
    align = _determine_mesh_orientation(zone, extrusion_vector, comm)
    
    is_bar  = lambda n: PT.get_label(n) == 'Elements_t' and PT.Element.CGNSName(n) == 'BAR_2'
    is_tri  = lambda n: PT.get_label(n) == 'Elements_t' and PT.Element.CGNSName(n) == 'TRI_3'
    is_quad = lambda n: PT.get_label(n) == 'Elements_t' and PT.Element.CGNSName(n) == 'QUAD_4'

    # 2/ Extrude Tri to Prism
    new_tris_l = []
    for num, tri in enumerate(PT.get_children_from_predicate(zone, is_tri)):
        new_tri1, new_tri2 = _extrude_tri_to_prism_and_tris(tri, num, n_vtx, er_max, align=align)
        new_tris_l.extend([new_tri1, new_tri2])
        er_max += 2*PT.Element.Size(tri)
    for new_tri in new_tris_l:
        PT.add_child(zone, new_tri)
    
    # 3/ Extrude Quad to Hexa
    new_quads_l = []
    for num, quad in enumerate(PT.get_children_from_predicate(zone, is_quad)):
        new_quad1, new_quad2 = _extrude_quad_to_hexa_and_quads(quad, num, n_vtx, er_max, align=align)
        new_quads_l.extend([new_quad1, new_quad2])
        er_max += 2*PT.Element.Size(quad)
    for new_quad in new_quads_l:
        PT.add_child(zone, new_quad)
    
    # 4/ Extrude Bar to Quad
    for num, bar in enumerate(PT.get_nodes_from_predicate(zone, is_bar)):
        _extrude_bar_to_quad(bar, num, n_vtx, align=align)
    
    # 5/ Manage K-plans
    # Note : Subset are created as EdgeCenter right now, because they calling function convert it to FaceCenter after
    # TODO : fusionner pour avoir un seul couple de plan, regroupant tt les elts 2D (comme en NGON)
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
                gc1 = PT.new_GridConnectivity(name=gc1_name, donor_name=PT.get_name(zone),
                                              type='Abutting1to1', loc='EdgeCenter',
                                              # point_range      =pr1,
                                              # point_range_donor=pr2,
                                              point_list=[pl1],
                                              point_list_donor=[pl2],
                                              parent=zgc)
                PT.new_GridConnectivityProperty({"translation": np.array(extrusion_vector, dtype=np.float64)}, parent=gc1)
                MT.newDistribution({'Index' : distrib1}, parent=gc1)
                gc2 = PT.new_GridConnectivity(name=gc2_name, donor_name=PT.get_name(zone),
                                              type='Abutting1to1', loc='EdgeCenter',
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
                           loc='EdgeCenter', family=fam_name, parent=zbc)
            MT.newDistribution({'Index' : distrib}, parent=bc)
    else:
        raise RuntimeError(f"'kplan_type' is {kplan_type} but only 'perio' and 'fam_bc' are allowed !")

def _pl_and_data_vtx_duplication(pl, distrib_idx, n_vtx_2d, data, comm):
    """
    Internal function used by _extrusion_2d to create the duplicated PointList 
    and associated datas needed to extented it to the second plan
    """
    new_distrib_idx  = par_utils.uniform_distribution(2*distrib_idx[2],  comm)
    # > Duplicate data in part_data
    if pl is None:
        part_data = {}
    else:
        part_data = {'PointList': [pl[0], pl[0]+n_vtx_2d]}
    for name, value in data.items():
       part_data[name] = [value, value]
    # > Compute ln_to_gn
    ln_to_gn_l = [np.arange(distrib_idx[0]+1, distrib_idx[1]+1, dtype=pdm_dtype),
                  np.arange(distrib_idx[0]+1, distrib_idx[1]+1, dtype=pdm_dtype)+distrib_idx[2]]
    # > Part to block
    dist_data = EP.part_to_block(part_data, new_distrib_idx, ln_to_gn_l, comm)
    
    if pl is None:
        dist_pl = None
    else:
        dist_pl = dist_data.pop('PointList').reshape((1,-1), order='F')
    # > Return
    return new_distrib_idx, dist_pl, dist_data
    

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
    
    zone_to_distrib_vtx = dict()
    for zone_path in PT.predicates_to_paths(dist_tree, 'CGNSBase_t/Zone_t'):
        zone = PT.get_node_from_path(dist_tree, zone_path)
        if not PT.Zone.CellDimension(zone) == 2:
            raise ValueError("Only 2D zones are supported in this function")

        distrib_vtx_2d_n = MT.getDistribution(zone, 'Vertex')[1].copy()
        zone_to_distrib_vtx[zone_path] = distrib_vtx_2d_n

    for base, zone in PT.get_children_from_labels(dist_tree, ['CGNSBase_t', 'Zone_t'], ancestors=True):
        
            
        distrib_vtx_2d = zone_to_distrib_vtx[f'{base[0]}/{zone[0]}']
        cell_offset_2d = PT.Zone.get_elt_range_per_dim(zone)[2][0]
        n_vtx_2d = distrib_vtx_2d[2]

        coord_n = PT.get_child_from_label(zone, 'GridCoordinates_t')
        coords = PT.Zone.coordinates(zone)
        for coord_name, coord_val in coords._asdict().items():
            if coord_val is None:
                PT.new_DataArray(coord_name, np.zeros_like(coords[0]), parent=coord_n)
        
        # Generate new vertices and Elements
        if PT.Zone.Type(zone) == 'Structured':
            raise NotImplementedError('Extrusion of 2D structured meshes is not yet implemented !')
        elif PT.Zone.Type(zone) == 'Unstructured':
            all_element_types = set([PT.Element.CGNSName(e) for e in PT.get_children_from_label(zone, 'Elements_t')])
            if all_element_types <= {'NODE', 'BAR_2', 'NGON_n'}:
                _extrusion_2d_u_ngon(zone, extrusion_vector, comm, kplan_type=kplan_type)
            elif all_element_types <= {'NODE', 'BAR_2', 'TRI_3', 'QUAD_4'}:
                _extrusion_2d_u_elem(zone, extrusion_vector, comm, kplan_type=kplan_type)
            else:
                raise ValueError(f'Zone {PT.get_name(zone)} is neither full NGON or composed only of TRI and QUAD elements !')
        else:
            raise ValueError(f'Zone {PT.get_name(zone)} is neither structured nor unstructured !')

        # Update zone dims
        # Remark: in extrusion, no need to change nb_cell because the new 3D cells are the 
        #         former 2D ones extruded
        zone_dims = PT.get_value(zone)
        zone_dims[0][0] *= 2
        if PT.Zone.has_ngon_elements(zone):
            cell_offset_3d = PT.Element.Range(PT.Zone.NGonNode(zone))[1] + 1
        else:
            cell_offset_3d = PT.Zone.get_elt_range_per_dim(zone)[3][0]
        
        # Update containers
        is_container = lambda n : PT.get_label(n) in ['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t', 'BCDataSet_t']
        has_pl = lambda n : PT.get_child_from_name(n, 'PointList') is not None
        
        # > CellCenter -> Shift to refer cells ids
        is_container_cell = lambda n: is_container(n) and PT.Subset.GridLocation(n) == 'CellCenter'
        for container in PT.get_nodes_from_predicate(zone, lambda n : is_container_cell(n) and has_pl(n)):
            PT.get_child_from_name(container, 'PointList')[1] += cell_offset_3d - cell_offset_2d
        # > FaceCenter -> should not exist on 2d mesh, remove it
        is_container_face = lambda n: is_container(n) and PT.Subset.GridLocation(n) == 'FaceCenter'
        container_face_l = PT.get_nodes_from_predicate(zone, is_container_face, depth=3)
        if len(container_face_l) > 0:
            cnt_names = [PT.get_name(n) for n in container_face_l]
            msg = f"The following containers have been removed from the input 2D mesh, because GridLocation == FaceCenter" \
                  f" is not allowed by the CGNS norm on 2d meshes : {cnt_names}"
            mlog.error(msg)
            PT.rm_nodes_from_predicate(zone, is_container_face, depth=3)

        # > EdgeCenter -> becomes FaceCenter
        is_container_edge = lambda n: is_container(n) and PT.Subset.GridLocation(n) == 'EdgeCenter'
        for container in PT.get_nodes_from_predicate(zone, is_container_edge):
            PT.update_child(container, 'GridLocation', value='FaceCenter')

        # > Vertex
        is_vertex = lambda n : PT.Subset.GridLocation(n) == 'Vertex'
        
        if dupl_vtx_info:
            # dupl_vtx_info is True => we need to duplicate data in in Vertex containers
            for container in PT.get_children_from_predicate(zone, lambda n : is_container(n) and is_vertex(n)):
                # BCDS are skipped because of get_children
                if has_pl(container):
                    pl = PT.get_child_from_name(container, 'PointList')[1]
                    distrib_idx = MT.getDistribution(container, 'Index')[1]
                elif PT.get_label(container) == 'ZoneSubRegion_t':
                    pl = None
                    zsr_extent = PT.Subset.ZSRExtent(container, zone)
                    extent_node = PT.get_node_from_path(zone, zsr_extent)
                    distrib_idx = MT.getDistribution(extent_node, 'Index')[1]
                else:
                    pl = None
                    distrib_idx = distrib_vtx_2d
                data = {PT.get_name(n) : PT.get_value(n) for n in PT.get_children_from_label(container, 'DataArray_t')}
                new_distrib_idx, new_pl, new_data = _pl_and_data_vtx_duplication(pl, distrib_idx, n_vtx_2d, data, comm)
                if has_pl(container): # Update PointList + Distribution
                    PT.update_child(container, 'PointList', value=new_pl)
                    MT.newDistribution({'Index' : new_distrib_idx}, container)
                for name, value in new_data.items():
                    PT.set_value(PT.get_child_from_name(container, name), value)
            
            for _, bc, bcds in PT.get_children_from_predicates(zone, 'ZoneBC_t/BC_t/BCDataSet_t', ancestors=True):
                if PT.Subset.GridLocation(bcds) == 'Vertex':
                    pl_ower = bcds if has_pl(bcds) else bc
                    pl = PT.get_child_from_name(pl_ower, 'PointList')
                    distrib_idx = MT.getDistribution(pl_ower, 'Index')

                    data = {path : PT.get_node_from_path(bcds, path)[1] for path in PT.predicates_to_paths(bcds, 'BCData_t/DataArray_t')}
                    new_distrib_idx, new_pl, new_data = _pl_and_data_vtx_duplication(pl[1], distrib_idx[1], n_vtx_2d, data, comm)
                    if has_pl(bcds):
                        PT.set_value(pl, new_pl)
                        PT.set_value(distrib_idx, new_distrib_idx)
                    for path, value in new_data.items():
                        PT.set_value(PT.get_node_from_path(bcds, path), value)
        else:
            # dupl_vtx_info is False => do not add vertices in Vertex containers; consequently, we need to add a PointList if not already existing
            for container in PT.get_children_from_predicate(zone, lambda n : is_container(n) and is_vertex(n) and not has_pl(n)):
                # BCDS are skipped because of get_children
                if PT.get_label(container) == 'ZoneSubRegion_t':
                    zsr_extent = PT.Subset.ZSRExtent(container, zone)
                    extent_node = PT.get_node_from_path(zone, zsr_extent)
                    PT.add_child(container, PT.deep_copy(PT.get_child_from_name(extent_node, 'PointList')))
                    PT.add_child(container, PT.deep_copy(PT.get_child_from_name(extent_node, ':CGNS#Distribution')))
                    PT.rm_children_from_name(container, '*RegionName')
                else:
                    pl = np.arange(distrib_vtx_2d[0]+1, distrib_vtx_2d[1]+1, dtype=zone[1].dtype).reshape((1,-1), order='F')
                    PT.new_IndexArray('PointList', value=pl, parent=container)
                    MT.newDistribution({'Index': distrib_vtx_2d}, parent=container)
            for _, bc, bcds in PT.get_children_from_predicates(zone, 'ZoneBC_t/BC_t/BCDataSet_t', ancestors=True):
                if is_vertex(bcds) and not has_pl(bcds):
                    assert is_vertex(bc)
                    PT.add_child(bcds, PT.deep_copy(PT.get_child_from_name(bc, 'PointList')))
                    PT.add_child(bcds, PT.deep_copy(PT.get_child_from_name(bc, ':CGNS#Distribution')))
        
        # Update subsets
        is_subset = lambda n : PT.get_label(n) in ['BC_t', 'GridConnectivity_t', 'GridConnectivity_1to1_t']
        is_subset_cell = lambda n: is_subset(n) and PT.Subset.GridLocation(n) == 'CellCenter'
        is_subset_face = lambda n: is_subset(n) and PT.Subset.GridLocation(n) == 'FaceCenter'
        is_subset_edge = lambda n: is_subset(n) and PT.Subset.GridLocation(n) == 'EdgeCenter'
        is_subset_vtx  = lambda n: is_subset(n) and PT.Subset.GridLocation(n) == 'Vertex'
        # > CellCenter -> Shift to refer cells ids
        for container in PT.get_nodes_from_predicate(zone, is_subset_cell):
            PT.get_child_from_name(container, 'PointList')[1] += cell_offset_3d - cell_offset_2d
        # > FaceCenter -> should not exist
        subset_face_l = PT.get_nodes_from_predicate(zone, is_subset_face, depth=2)
        if len(subset_face_l) > 0:
            cnt_names = [PT.get_name(n) for n in subset_face_l]
            msg = f"The following subsets have been removed from the input 2D mesh, because GridLocation == FaceCenter" \
                  f" is not allowed by the CGNS norm on 2d meshes : {cnt_names}"
            mlog.error(msg)
            PT.rm_nodes_from_predicate(zone, is_subset_face, depth=2)

        # > EdgeCenter -> becomes FaceCenter (no need to change their PointList)
        for subset in PT.get_nodes_from_predicate(zone, is_subset_edge, depth=2):
            PT.update_child(subset, 'GridLocation', value='FaceCenter')

        # > Vertex : we have to add the duplicated nodes from PL vertices to PL
        for subset_vertex in PT.get_nodes_from_predicate(zone, is_subset_vtx):
            pl = PT.get_child_from_name(subset_vertex, 'PointList')
            distrib_idx = MT.getDistribution(subset_vertex, 'Index')
            new_distrib_idx, new_pl, _ = _pl_and_data_vtx_duplication(pl[1], distrib_idx[1], n_vtx_2d, {}, comm)
            # Manage PointListDonor
            pld = PT.get_child_from_name(subset_vertex, 'PointListDonor')
            if pld is not None:
                opp_zone_path = PT.GridConnectivity.ZoneDonorPath(subset_vertex, base[0])
                distrib_vtx_2d_opp = zone_to_distrib_vtx[opp_zone_path]
                _, new_pld, _ = _pl_and_data_vtx_duplication(pld[1], distrib_idx[1], distrib_vtx_2d_opp[2], {}, comm)
                PT.set_value(pld, new_pld)
            # Update PointList and Distribution
            PT.set_value(pl, new_pl)
            PT.set_value(distrib_idx, new_distrib_idx)
    
    # Update base dimension
    for base in PT.get_all_CGNSBase_t(dist_tree):
        PT.set_value(base, [3, 3])
        if kplan_type=='fam_bc':
            PT.new_Family('__maia_former_plan',   family_bc='UserDefined', parent=base)
            PT.new_Family('__maia_extruded_plan', family_bc='UserDefined', parent=base)
