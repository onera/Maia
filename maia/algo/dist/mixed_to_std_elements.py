import mpi4py.MPI as mpi

import os
import numpy as np

import maia

from maia            import pytree        as PT
from maia.io         import cgns_io_tree  as IOT
from maia.algo       import dist          as MAD
from maia.algo.dist  import s_to_u        as CSU
from maia.algo.dist  import merge         as ME
from maia.transfer   import protocols     as MTP
from maia.utils      import par_utils     as MUPar

import maia.pytree.sids.elements_utils    as MPSEU

import Pypdm.Pypdm as PDM

def collect_pl_nodes(root, filter_loc=None):
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
    elif pr_n is not None and PT.get_value(pr_n).shape[0] == 1:
      pr = PT.get_value(pr_n)
      distrib = PT.get_value(MT.getDistribution(node, 'Index'))
      pl = np_utils.single_dim_pr_to_pl(pr, distrib)
      new_pl_n = PT.new_node(name='PointList', value=pl, label='IndexArray_t', parent=node)
      PT.rm_nodes_from_label(zone,'Elements_t')
      pointlist_nodes.append(new_pl_n)
  return pointlist_nodes

def convert_mixed_to_elements(dist_tree, comm):
    """
    Transform a mixed connectivity into an element based connectivity.
    
    Tree is modified in place : mixed elements are removed from the zones
    and the PointList are updated.
  
    Args:
      dist_tree  (CGNSTree): Tree with connectivity described by mixed elements
      comm       (`MPIComm`) : MPI communicator
  
    Example:
        .. literalinclude:: snippets/test_algo.py
          :start-after: #convert_mixed_to_elements@start
          :end-before: #convert_mixed_to_elements@end
          :dedent: 2
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    for zone in PT.get_all_Zone_t(dist_tree):
        elem_types = {}
        ec_per_elem_type_loc = {}
        ln_to_gn_loc = {}
        nb_elem_prev = 0
        
        
        # 1/ Create local element connectivity for each element type found in each mixed node
        #    and deduce the local number of each element type
        for elem_pos,element in enumerate(PT.Zone.get_ordered_elements(zone)):
            assert PT.get_value(element)[0] == 20
            elem_er  = PT.get_node_from_name(element,'ElementRange',depth=1)[1]
            elem_ec  = PT.get_node_from_name(element,'ElementConnectivity',depth=1)[1]
            elem_eso = PT.get_node_from_name(element,'ElementStartOffset',depth=1)[1]
            elem_types_tab = elem_ec[elem_eso[:-1]-elem_eso[0]]
            elem_types_loc, nb_elems_per_types_loc = np.unique(elem_types_tab,return_counts=True)
            for e, elem_type in enumerate(elem_types_loc):
                if elem_type not in elem_types.keys():
                    elem_types[elem_type] = {}
                elem_types[elem_type][elem_pos] = nb_elems_per_types_loc[e]
                nb_nodes_per_elem = MPSEU.element_number_of_nodes(elem_type)
                ec_per_type = np.empty(nb_nodes_per_elem*nb_elems_per_types_loc[e],dtype=elem_ec.dtype)
                indices = np.intersect1d(np.where(elem_ec==elem_type),elem_eso[:-1]-elem_eso[0], assume_unique=True)
                for n in range(nb_nodes_per_elem):
                    ec_per_type[n::nb_nodes_per_elem] = elem_ec[indices+n+1]
                try:
                    ec_per_elem_type_loc[elem_type].append(ec_per_type)
                except:
                    ec_per_elem_type_loc[elem_type] = [ec_per_type]
        
        
        # 2/ Find all element types described in the mesh and the number of each
        elem_types_all = comm.allgather(elem_types)
        all_types = {}
        for d in elem_types_all:
            for kd,vd in d.items():
                if kd not in all_types.keys():
                    all_types[kd] = 0
                for pos, nb in vd.items():
                    all_types[kd] += nb
                    
        
        # 3/ To take into a count Paraview limitation, elements are sorted by
        #    decreased dimensions : 3D->2D->1D->0D
        #    Without this limitations, replace the following lines by:
        #        `all_types = dict(sorted(all_types.items()))`
        #        `key_types = np.array(list(all_types.keys()),dtype=np.int32)`
        dim_types = -1*np.ones(len(all_types.keys()),dtype=np.int32)
        key_types = np.array(list(all_types.keys()),dtype=np.int32)
        for k,key in enumerate(all_types.keys()):
            dim_types[k] = MPSEU.elements_properties[key][1]
        sort_indices = np.argsort(dim_types)
        key_types = key_types[sort_indices[::-1]] 
        
        
        # 4/ Create old to new element numbering for PointList update      
        ln_to_gn_element_list = []
        old_to_new_element_numbering_list = []
        nb_elem_prev_element_t_nodes = 0
        for elem_pos,element in enumerate(PT.Zone.get_ordered_elements(zone)):
            elem_ec  = PT.get_node_from_name(element,'ElementConnectivity',depth=1)[1]
            elem_eso = PT.get_node_from_name(element,'ElementStartOffset',depth=1)[1]
            elem_distrib = PT.get_node_from_path(element,':CGNS#Distribution/Element')[1]
            nb_elem_loc = elem_distrib[1]-elem_distrib[0]
            old_to_new_element_numbering = np.zeros(nb_elem_loc,dtype=elem_eso.dtype)
            ln_to_gn_element = np.arange(nb_elem_loc,dtype=np.int32) + 1 \
                           + elem_distrib[0] + nb_elem_prev_element_t_nodes
            nb_elem_prev_element_t_nodes += elem_distrib[2]
            all_elem_previous_types = 0
            
            for elem_type in key_types:
                nb_elems_per_type = all_types[elem_type]
                indices = np.where(elem_ec[elem_eso[:-1]-elem_eso[0]]==elem_type)[0]
                old_to_new_element_numbering[indices] = np.arange(len(indices),dtype=elem_eso.dtype) \
                                                      + 1 + all_elem_previous_types
                for p in range(elem_pos):
                    try:
                        old_to_new_element_numbering[indices] += elem_types[elem_type][p]
                    except:
                        old_to_new_element_numbering[indices] += 0
                for r in range(rank):
                    try:
                        old_to_new_element_numbering[indices] += elem_types_all[r][elem_type][elem_pos]
                    except:
                        old_to_new_element_numbering[indices] += 0
                all_elem_previous_types += all_types[elem_type]
            
            old_to_new_element_numbering_list.append(old_to_new_element_numbering)
            ln_to_gn_element_list.append(ln_to_gn_element)
        
        
        # 5/ Delete mixed nodes and add standard elements nodes
        PT.rm_nodes_from_label(zone,'Elements_t')
        beg_erange = 1
        for elem_type in key_types:
            nb_elems_per_type = all_types[elem_type]
            part_data_ec = []
            part_stride_ec = []
            ln_to_gn_list = []
            label = MPSEU.element_name(elem_type)
            name = label[0].upper() + label[1:].lower()
            nb_nodes_per_elem = MPSEU.element_number_of_nodes(elem_type)
            erange = [beg_erange, beg_erange+nb_elems_per_type-1]
            
            if elem_type in ec_per_elem_type_loc.keys():
                for p,pos in enumerate(elem_types[elem_type].keys()):
                    nb_nodes_loc = elem_types[elem_type][pos]
                    part_data_ec.append(ec_per_elem_type_loc[elem_type][p])
                    stride_ec = (nb_nodes_per_elem)*np.ones(nb_nodes_loc,dtype = np.int32)
                    part_stride_ec.append(stride_ec)
                    ln_to_gn = np.arange(nb_nodes_loc,dtype=elem_distrib.dtype) + 1
                    for r in range(size):
                        elem_types_all_next = elem_types_all[r]
                        if elem_type in elem_types_all_next.keys():
                            for elem_pos in elem_types_all_next[elem_type]:
                                if elem_pos < pos:
                                    ln_to_gn += elem_types_all_next[elem_type][elem_pos]
                    for r in range(rank):
                        elem_types_all_prev = elem_types_all[r]
                        if elem_type in elem_types_all_prev.keys():
                            if pos in elem_types_all_prev[elem_type].keys():
                                ln_to_gn += elem_types_all_prev[elem_type][pos]
                    ln_to_gn_list.append(ln_to_gn)
            
            elem_distrib = MUPar.uniform_distribution(nb_elems_per_type,comm)
            ptb = MTP.PartToBlock(elem_distrib,ln_to_gn_list,comm)
            __, econn = ptb.exchange_field(part_data_ec,part_stride_ec)
            
            beg_erange += nb_elems_per_type
            elem_n = PT.new_Elements(name,label,erange=erange,econn=econn,parent=zone)
            elem_n_distrib = PT.new_node(name=':CGNS#Distribution', label='UserDefined_t', parent=elem_n)
            PT.new_DataArray('Element', elem_distrib, parent=elem_n_distrib)
        
        # 6/ Update all PointList with GridLocation != Vertex
        filter_loc = ['EdgeCenter','FaceCenter','CellCenter']
        pl_list = collect_pl_nodes(zone,filter_loc)
        
        ln_to_gn_pl_list = []
        for p,pl in enumerate(pl_list):
            ln_to_gn_pl_list.append(PT.get_value(pl)[0].astype(np.int32))
        part1_to_part2_idx_list = []
        for l in ln_to_gn_element_list:
            part1_to_part2_idx = np.arange(l.size+1, dtype=np.int32)
            part1_to_part2_idx_list.append(part1_to_part2_idx)
            
        PTP = PDM.PartToPart(comm, ln_to_gn_element_list, ln_to_gn_pl_list, \
                             part1_to_part2_idx_list, ln_to_gn_element_list)
    
        request = PTP.iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                            PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART1_TO_PART2,
                            old_to_new_element_numbering_list)
        __, old_to_new_pl_list = PTP.wait(request)
    
        for p, pl in enumerate(pl_list):
            PT.set_value(pl,np.array(old_to_new_pl_list[p]).reshape((1,len(old_to_new_pl_list[p]))))
    
