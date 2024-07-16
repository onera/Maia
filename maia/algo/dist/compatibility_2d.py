import numpy as np

import Pypdm.Pypdm as PDM

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia                      import npy_pdm_gnum_dtype         as pdm_dtype
from maia.algo.dist.ngon_tools import PDM_dfacecell_to_dcellface
from maia.transfer             import protocols                  as EP
from maia.utils                import np_utils, par_utils


def convert_ngon2d_to_bar(zone):
    """
    Function to convert, in 2D, NGon node that wrongly describe edge_vtx 
    connectivity to BAR node
    """
    # > Get NGon node
    ngon_n = PT.Zone.NGonNode(zone)
    # > Test if only edges in ngon_n
    eso = PT.get_child_from_name(ngon_n, 'ElementStartOffset')[1]
    nb_nodes_per_faces = np.diff(eso)
    types_of_faces = list(set(nb_nodes_per_faces))
    assert len(types_of_faces) <= 1
    if len(types_of_faces) == 1:
        assert types_of_faces[0] == 2
    # > Del ESO
    PT.rm_node_from_path(ngon_n, 'ElementStartOffset')
    PT.rm_node_from_path(ngon_n, ':CGNS#Distribution/ElementConnectivity')
    # > Change name end value (22 => 3)
    #TO DO : choix du nom : BAR_2 ou EdgeElements
    PT.set_name(ngon_n, 'EdgeElements')
    PT.set_value(ngon_n, [3, 0])


# TODO : a mutualiser ?
def compute_face_vtx_from_face_edge_and_edge_vtx(face_edge_idx, face_edge, edge_vtx, face_distrib, edge_distrib, comm):
    """
    Compute face_vtx connectivity from face_edge and edge_vtx connectivities in
    a distributed context

    Args:
      face_edge_idx (array)  : Face to edge connectivity index
      face_edge     (array)  : Face to edge connectivity
      edge_vtx      (array)  : Edge to vertex connectivity
      face_distrib  (array)  : Face distribution
      edge_distrib  (array)  : Edge distribution
      comm          (MPIComm): MPI communicator
    """
    face_edge_idx = np_utils.safe_int_cast(face_edge_idx - face_edge_idx[0], np.int32)
    
    dist_data = {'connectivity' : edge_vtx}
    dist_stride = np.ones(edge_distrib[1]-edge_distrib[0], dtype=np.int32) * 2
    part_stride, part_data = EP.block_to_part_strided(dist_stride, dist_data, edge_distrib, [face_edge], comm)
    global_edge_vtx = part_data["connectivity"][0]
    
    # Convert in local numbering to be allowed to use part algo in dist context
    uniq, idx, inv = np.unique(global_edge_vtx, return_index=True, return_inverse=True)
    local_edge_vtx = np_utils.safe_int_cast(inv+1, np.int32)
    
    local_face_edge = (np.arange(len(face_edge), dtype=np.int32)+1)*np.sign(face_edge)
    
    local_face_vtx = PDM.compute_face_vtx_from_face_and_edge(face_edge_idx,
                                                             local_face_edge,
                                                             local_edge_vtx)
    
    # Return in the global numbering
    global_face_vtx = global_edge_vtx[idx][local_face_vtx-1]
    
    return(global_face_vtx)


def convert_nface2d_to_ngon(zone, comm):
    """
    Convert, in 2D, NFace node that wrongly describe face_edge 
    connectivity to NGOn node describing face_vtx connectivity
    """
    # > Get Bar node information
    is_bar = lambda n: (PT.get_label(n) == 'Elements_t') and (PT.get_value(n)[0] == 3)
    bar_n = PT.get_node_from_predicate(zone, is_bar)
    
    bar_distrib = MT.getDistribution(bar_n, 'Element')[1]
    
    # > Get NFace node information
    nface_n = PT.Zone.NFaceNode(zone)
    
    nface_distrib = MT.getDistribution(nface_n, 'Element')[1]
    nb_poly_loc = nface_distrib[1]-nface_distrib[0]
    
    # > Get vertex ids of faces from bar ids    
    face_bar_n = PT.get_child_from_name(nface_n, 'ElementConnectivity')
    face_bar_idx = PT.get_child_from_name(nface_n, 'ElementStartOffset')[1]
    bar_vtx = PT.get_child_from_name(bar_n, 'ElementConnectivity')[1]
    face_vtx = compute_face_vtx_from_face_edge_and_edge_vtx(face_bar_idx, face_bar_n[1], bar_vtx,
                                                            nface_distrib, bar_distrib, comm)
    PT.set_value(face_bar_n, face_vtx)
    
    # > Change name and value (23 => 22)
    #TO DO : choix du nom : NGon_n ou NGonElements
    PT.set_name(nface_n, 'NGonElements')
    PT.set_value(nface_n, [22, 0])


# TODO : a mutualiser ?
def update_gridlocation_subset(zone, gl_in, gl_out):
    """
    Change all GriLocation_t nodes value in a zone
    """
    is_gl_in_subset = lambda n: (PT.get_label(n) in ['BC_t', 'GridConnectivity', 'GridConnectivity_1to1', 'ZoneSubRegion']) \
                                 and (PT.Subset.GridLocation(n) == gl_in)
    for subset_face in PT.get_nodes_from_predicate(zone, is_gl_in_subset):
        PT.update_child(subset_face, 'GridLocation', 'GridLocation_t', gl_out)


def bar_pe_to_nface2d(zone, comm):
    """
    Convert, in 2D, BAR node with ParentElements node to a NFace node
    that wrongly describe face_edge connectivity
    """
    PT.rm_node_from_path(zone,'NGonElements')
    # > Get Bar node information
    is_bar = lambda n: (PT.get_label(n) == 'Elements_t') and (PT.get_value(n)[0] == 3)
    bar_n = PT.get_node_from_predicate(zone, is_bar)
    
    PT.set_value(bar_n, [22, 0])
    
    bar_distrib = MT.getDistribution(bar_n, 'Element')[1]
    
    edge_face = maia.algo.indexing.get_ngon_pe_local(bar_n).reshape(-1,order='C')
    edge_face_idx = np.arange(bar_distrib[0], bar_distrib[1]+1, dtype=np.int32)*2
    PT.new_DataArray('ElementStartOffset', value=edge_face_idx, parent=bar_n)
    MT.newDistribution({'ElementConnectivity': bar_distrib*2},
                        parent = bar_n)
    
    face_distrib = MT.getDistribution(zone, 'Cell')[1]
    full_bar_distrib = par_utils.partial_to_full_distribution(bar_distrib,comm)
    full_face_distrib = par_utils.partial_to_full_distribution(face_distrib,comm)
    face_edge_idx, face_edge = PDM_dfacecell_to_dcellface(comm,
                                                          full_bar_distrib,
                                                          full_face_distrib,
                                                          edge_face)
    faces_per_proc = comm.allgather(len(face_edge_idx)-1)
    nb_faces_tot = np.sum(faces_per_proc, dtype=pdm_dtype)
    nb_faces_prev = np.sum(faces_per_proc[0:comm.rank], dtype=pdm_dtype)
    nb_faces_cur = np.sum(faces_per_proc[0:comm.rank+1], dtype=pdm_dtype)
    size_face_edge_per_proc = comm.allgather(face_edge_idx[-1])
    size_face_edge_tot = np.sum(size_face_edge_per_proc, dtype=pdm_dtype)
    size_face_edge_prev = np.sum(size_face_edge_per_proc[0:comm.rank], dtype=pdm_dtype)
    size_face_edge_cur = np.sum(size_face_edge_per_proc[0:comm.rank+1], dtype=pdm_dtype)
    nface_er = np.array([1, nb_faces_tot], dtype=pdm_dtype)+PT.get_child_from_name(bar_n, 'ElementRange')[1][1]
    nface_n = PT.new_NFaceElements(erange=nface_er, ec=face_edge,eso=face_edge_idx+size_face_edge_prev,parent=zone)
    MT.newDistribution({'Element': [nb_faces_prev, nb_faces_cur, nb_faces_tot],
                        'ElementConnectivity': [size_face_edge_prev, size_face_edge_cur, size_face_edge_tot]},
                        parent = nface_n)


def convert_cass_to_std_2d_u(dist_tree, comm):
    """
    Convert tree with Cassiopee standard to be CGNS 4 compliant

    Args:
      dist_tree (CGNSTree): Distributed tree
      comm      (MPIComm) : MPI communicator
    """
    for zone in PT.get_all_Zone_t(dist_tree):
        convert_ngon2d_to_bar(zone)
        convert_nface2d_to_ngon(zone, comm)
        update_gridlocation_subset(zone, 'FaceCenter', 'EdgeCenter')


def convert_std_to_cass_2d_u(dist_tree, comm):
    """
    Convert CGNS 4 compliant tree at Cassiopee standard

    Args:
      dist_tree (CGNSTree): Distributed tree
      comm      (MPIComm) : MPI communicator
    """
    for zone in PT.get_all_Zone_t(dist_tree):
        bar_pe_to_nface2d(zone, comm)
        update_gridlocation_subset(zone, 'EdgeCenter', 'FaceCenter')
