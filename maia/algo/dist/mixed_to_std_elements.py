import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

import maia
from maia.typing import *
from maia.transfer      import protocols     as EP
from maia.utils         import par_utils, np_utils
import maia.pytree.sids.elements_utils    as MPSEU

HAS_SUBSET = PT.pred.has_child('PointList') or PT.pred.has_child('PointRange')

def collect_pl_nodes(root: CGNSTree, filter_loc: Optional[List[str]] = None) -> List[CGNSTree]:
  """
  Search and collect all the pointList nodes found in subsets found
  under root
  If filter_loc list is not None, select only the PointList nodes of given
  GridLocation.
  Remark : if the subset is defined by a PointRange, we replace the PointRange node
           to the equivalent PointList node and collect the new PointList node
  """
  pointlist_nodes = []
  for node in PT.get_all_subsets(root,filter_loc):
    pl_n = PT.get_child_from_name(node, 'PointList')
    pr_n = PT.get_child_from_name(node, 'PointRange')
    if pl_n is not None:
      pointlist_nodes.append(pl_n)
    elif pr_n is not None and (pr:=PT.get_np_value(pr_n)).shape[0] == 1:
      distrib = MT.distribution_value(node, 'Index')
      pl = np_utils.single_dim_pr_to_pl(pr, distrib)
      new_pl_n = PT.new_node(name='PointList', value=pl, label='IndexArray_t', parent=node)
      PT.rm_nodes_from_label(node,'IndexRange_t')
      pointlist_nodes.append(new_pl_n)
  return pointlist_nodes

def convert_mixed_to_elements(dist_tree: CGNSDistTree, comm: MPIComm) -> None:
    """
    Transform a mixed connectivity into an element based connectivity.
    
    Tree is modified in place : mixed elements are removed from the zones
    and the PointList are updated.
  
    Args:
      dist_tree  (CGNSDistTree): Tree with connectivity described by mixed elements
      comm       (`MPIComm`)   : MPI communicator
  
    Example:
        .. literalinclude:: snippets/test_algo.py
          :start-after: #convert_mixed_to_elements@start
          :end-before: #convert_mixed_to_elements@end
          :dedent: 2
    """
    MT.check_cgns_dist_tree(dist_tree)
    rank = comm.Get_rank()
    size = comm.Get_size()

    for zone in PT.get_all_Zone_t(dist_tree):
        elem_types:Dict[int, Dict[int, int]] = {} # For each element type: dict id of mixed node -> number of elts
        ec_per_elem_type_loc:Dict[int, List[NDArray]] = {}
        
        # 1/ Create local element connectivity for each element type found in each mixed node
        #    and deduce the local number of each element type
        for elem_pos,element in enumerate(PT.Zone.get_ordered_elements(zone)):
            assert PT.Element.CGNSName(element) not in ['NGON_n', 'NFACE_n']  
            if PT.Element.CGNSName(element) != 'MIXED':                       
                elem_ec  = PT.get_np_value(PT.find_child_from_name(element,'ElementConnectivity'))
                elem_distri = MT.distribution_value(element, 'Element')
                elem_type = PT.Element.Type(element)
                elem_size = elem_distri[1] - elem_distri[0]
                if elem_type not in elem_types.keys():
                    elem_types[elem_type] = {}
                elem_types[elem_type][elem_pos] = elem_size
                try:
                    ec_per_elem_type_loc[elem_type].append(elem_ec)
                except KeyError:
                    ec_per_elem_type_loc[elem_type] = [elem_ec]

            else:
                elem_cnt = MT.Element.connectivity(element)
                elem_eso_loc = elem_cnt.displs[:-1]
                elem_types_tab = elem_cnt.values[elem_eso_loc]
                elem_types_loc, nb_elems_per_types_loc = np.unique(elem_types_tab,return_counts=True)
                for e, elem_type in enumerate(elem_types_loc):
                    if elem_type not in elem_types.keys():
                        elem_types[elem_type] = {}
                    elem_types[elem_type][elem_pos] = nb_elems_per_types_loc[e]
                    nb_nodes_per_elem = MPSEU.element_number_of_nodes(elem_type)
                    assert nb_nodes_per_elem is not None
                    ec_per_type = np.empty(nb_nodes_per_elem*nb_elems_per_types_loc[e],dtype=elem_cnt.dtype)
                    # Retrive start idx of mixed elements having this type
                    indices = np.intersect1d(np.where(elem_cnt.values==elem_type),elem_eso_loc, assume_unique=True)
                    for n in range(nb_nodes_per_elem):
                        ec_per_type[n::nb_nodes_per_elem] = elem_cnt.values[indices+n+1]
                    try:
                        ec_per_elem_type_loc[elem_type].append(ec_per_type)
                    except KeyError:
                        ec_per_elem_type_loc[elem_type] = [ec_per_type]
        
        # 2/ Find all element types described in the mesh and the number of each
        elem_types_all = comm.allgather(elem_types)
        all_types = {} # For each type : total number of elts appearing in zone
        for proc_dict in elem_types_all:
            for kd,vd in proc_dict.items():
                if kd not in all_types:
                    all_types[kd] = 0
                all_types[kd] += sum(vd.values())
        all_types = dict(sorted(all_types.items()))
                    
        
        # 3/ To take into account Paraview limitation, elements are sorted by
        #    decreased dimensions : 3D->2D->1D->0D
        #    Without this limitation, replace the following lines by:
        #        `key_types = np.array(list(all_types.keys()),dtype=np.int32)`
        key_types = sorted(all_types.keys(), key=MPSEU.element_dim, reverse=True) #type:ignore #(element_dim does not return None on std elements)
        
        
        # 4/ Create old to new element numbering (to update PointList/PointRange)
        #    and old to new cell numbering (to update CellCenter FlowSolution)
        cell_dim = max([MPSEU.element_dim(k) for k in key_types]) #type:ignore #(element_dim does not return None on std elements)
        ln_to_gn_element_list = []
        ln_to_gn_cell_list = []
        old_to_new_element_numbering_list = []
        old_to_new_cell_numbering_list = []
        nb_elem_prev_element_t_nodes = 0
        for elem_pos,element in enumerate(PT.Zone.get_ordered_elements(zone)):
            is_std_elt = PT.Element.CGNSName(element) != 'MIXED'
            elem_distrib = MT.distribution_value(element, 'Element')
            nb_elem_loc = elem_distrib[1]-elem_distrib[0]
            nb_cell_loc = 0
            for et in elem_types:
                if MPSEU.element_dim(et) == cell_dim:
                    try:
                        nb_cell_loc += elem_types[et][elem_pos]
                    except KeyError:
                        pass
            elem_ec  = PT.get_np_value(PT.find_child_from_name(element, 'ElementConnectivity'))
            elem_eso = PT.find_child_from_name(element, 'ElementStartOffset')
            old_to_new_element_numbering = np.zeros(nb_elem_loc,dtype=elem_ec.dtype)
            old_to_new_cell_numbering    = np.zeros(nb_cell_loc,dtype=maia.npy_pdm_gnum_dtype)
            ln_to_gn_element = np.arange(nb_elem_loc,dtype=maia.npy_pdm_gnum_dtype) + 1\
                             + elem_distrib[0] + nb_elem_prev_element_t_nodes
            ln_to_gn_cell    = np.arange(nb_cell_loc,dtype=maia.npy_pdm_gnum_dtype)
            nb_elem_prev_element_t_nodes += elem_distrib[2]
            all_elem_previous_types = 0
            all_cell_previous_types = 0
            
            if is_std_elt:
                all_elem_pos = {elt_type : np.empty(0, int) for elt_type in key_types}
                all_elem_pos[PT.Element.Type(element)] = np.arange(nb_elem_loc)
                all_cell_pos = {}
                for elem_type in key_types:
                    if MPSEU.element_dim(elem_type) == cell_dim:
                        all_cell_pos[elem_type] = np.arange(nb_elem_loc) if PT.Element.Type(element) == elem_type else np.empty(0, int)
                        
            else:
                assert elem_eso[1] is not None
                elem_ec_type_pos = elem_ec[elem_eso[1][:-1]-elem_eso[1][0]] # Type of each element
                all_elem_pos = {}
                all_non_cell_pos = []
                for elem_type in key_types:
                    all_elem_pos[elem_type] = np.where(elem_ec_type_pos==elem_type)[0]
                    if MPSEU.element_dim(elem_type) != cell_dim:
                        all_non_cell_pos += list(all_elem_pos[elem_type])
                all_non_cell_pos = sorted(all_non_cell_pos)
                all_cell_pos = {}
                for elem_type in key_types:
                    if MPSEU.element_dim(elem_type) == cell_dim:
                        all_cell_pos[elem_type] = all_elem_pos[elem_type] - np.searchsorted(all_non_cell_pos,all_elem_pos[elem_type])
            for elem_type in key_types:
                nb_elems_per_type = all_types[elem_type]
                indices_elem = all_elem_pos[elem_type]
                old_to_new_element_numbering[indices_elem] = np.arange(len(indices_elem),dtype=elem_ec.dtype) + 1
                is_cell = MPSEU.element_dim(elem_type) == cell_dim
                if is_cell:
                    indices_cell = all_cell_pos[elem_type]
                    old_to_new_cell_numbering[indices_cell] = np.arange(len(indices_cell),dtype=elem_ec.dtype)
                    old_to_new_cell_numbering[indices_cell] += all_cell_previous_types
                # Add total elements of others (previous) type
                old_to_new_element_numbering[indices_elem] += all_elem_previous_types
                # Add number of elements on previous mixed nodes for this rank
                # Add number of cells on previous mixed nodes for this rank
                for p in range(elem_pos):
                    try:
                        old_to_new_element_numbering[indices_elem] += elem_types[elem_type][p]
                    except KeyError:
                        continue
                    if is_cell:
                        for r in range(size):
                            try:
                                ln_to_gn_cell += elem_types_all[r][elem_type][p]
                                old_to_new_cell_numbering[indices_cell] += elem_types_all[r][elem_type][p]
                            except KeyError:
                                continue
                # Add number of element on previous procs for this mixed node
                # Add number of cells on previous procs for this mixed node
                for r in range(rank):
                    try:
                        nb_elem_per_type_per_pos_per_rank = elem_types_all[r][elem_type][elem_pos]
                    except KeyError:
                        continue
                    old_to_new_element_numbering[indices_elem] += nb_elem_per_type_per_pos_per_rank
                    if is_cell:
                        ln_to_gn_cell += nb_elem_per_type_per_pos_per_rank
                        old_to_new_cell_numbering[indices_cell] += nb_elem_per_type_per_pos_per_rank
                all_elem_previous_types += all_types[elem_type]
                if is_cell:
                    all_cell_previous_types += all_types[elem_type]
            
            old_to_new_element_numbering_list.append(old_to_new_element_numbering)
            old_to_new_cell_numbering_list.append(old_to_new_cell_numbering)
            ln_to_gn_element_list.append(ln_to_gn_element)
            ln_to_gn_cell_list.append(ln_to_gn_cell)
        
        
        # 5/ Delete mixed nodes and add standard elements nodes
        PT.rm_nodes_from_label(zone,'Elements_t')
        beg_erange = 1
        for elem_type in key_types:
            nb_elems_per_type = all_types[elem_type]
            part_data_ec = []
            ln_to_gn_list = []
            nb_nodes_per_elem = MPSEU.element_number_of_nodes(elem_type)
            erange = [beg_erange, beg_erange+nb_elems_per_type-1]
            
            if elem_type in ec_per_elem_type_loc:
                for p,pos in enumerate(elem_types[elem_type]):
                    nb_nodes_loc = elem_types[elem_type][pos]
                    part_data_ec.append(ec_per_elem_type_loc[elem_type][p])
                    ln_to_gn = np.arange(nb_nodes_loc,dtype=elem_distrib.dtype)

                    offset = 0
                    for r,elem_types_rank in enumerate(elem_types_all):
                        if elem_type in elem_types_rank:
                            # Add number of elts on previous mixed nodes for all ranks
                            offset += sum([nb for other_pos, nb in elem_types_rank[elem_type].items() if other_pos < pos])
                            # Add number of elts on this mixed node, but only for previous ranks
                            if r < rank and pos in elem_types_rank[elem_type]:
                                offset += elem_types_rank[elem_type][pos]
                    ln_to_gn_list.append(ln_to_gn + offset)
            
            elem_distrib = par_utils.uniform_distribution(nb_elems_per_type,comm)

            GI_elem  = EP.GlobalIndexer(elem_distrib, ln_to_gn_list, comm)
            econn = GI_elem.Put(part_data_ec, count=nb_nodes_per_elem)
            
            beg_erange += nb_elems_per_type
            label = MPSEU.element_name(elem_type)
            elem_n = PT.new_Elements(label.capitalize(),label,erange=erange,econn=econn,parent=zone)
            MT.new_Distribution({'Element' : elem_distrib}, parent=elem_n)


        # 6/ Update all PointList with GridLocation != Vertex
        filter_loc = ['EdgeCenter','FaceCenter','CellCenter']
        pl_list = collect_pl_nodes(zone,filter_loc)
        
        ln_to_gn_pl_list = [maia.utils.as_pdm_gnum(PT.get_np_value(pl)[0]) for pl in pl_list]
            
        old_to_new_pl_list = EP.part_to_part(old_to_new_element_numbering_list, ln_to_gn_element_list, ln_to_gn_pl_list, comm)

        for pl_node, new_pl in zip(pl_list, old_to_new_pl_list):
            PT.set_value(pl_node, np.array(new_pl).reshape((1,-1), order='F'))


        # 7/ Update all FlowSolution with GridLocation == CellCenter
        
        # 7a. Redistribute old_to_new_cell_numbering to be coherent with
        #     cells distribution
        cells_distrib = MT.distribution_value(zone, 'Cell')

        GI_cell = EP.GlobalIndexer(cells_distrib, ln_to_gn_cell_list, comm)
        dist_old_to_new_cell_numbering = GI_cell.Put(old_to_new_cell_numbering_list)
        
        # 7b. Reorder FlowSolution DataArray
        GI_fs = EP.GlobalIndexer(cells_distrib, dist_old_to_new_cell_numbering, comm)

        is_fs_cc = PT.pred.label_in(['FlowSolution_t', 'DiscreteData_t']) & PT.pred.has_location('CellCenter') & ~HAS_SUBSET

        for node in PT.get_children_from_predicates(zone, [is_fs_cc, 'DataArray_t']):
            data = PT.get_np_value(node)
            GI_fs.Put(data, data) # Inplace

