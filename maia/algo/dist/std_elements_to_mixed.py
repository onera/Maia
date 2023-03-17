import mpi4py.MPI as mpi

import numpy as np

from maia            import pytree        as PT
from maia.transfer   import protocols     as MTP
from maia.utils      import par_utils     as MUPar


def convert_elements_to_mixed(dist_tree, comm):
    """
    Transform an element based connectivity into a mixed connectivity.
    
    Tree is modified in place : standard elements are removed from the zones.
    Note that the original ordering of elements is preserved.
  
    Args:
      dist_tree  (CGNSTree): Tree with connectivity described by standard elements
      comm       (`MPIComm`) : MPI communicator
  
    Example:
        .. literalinclude:: snippets/test_algo.py
          :start-after: #convert_elements_to_mixed@start
          :end-before: #convert_elements_to_mixed@end
          :dedent: 2
    """
    rank = comm.Get_rank()
    
    for zone in PT.get_all_Zone_t(dist_tree):
        part_data_ec = []
        part_data_eso = []
        part_stride_ec = []
        ln_to_gn_list = []
        nb_nodes_prev = 0
        nb_elem_prev = 0
        
        
        # 1/ Create local mixed connectivity and element start offeset tab for each element node
        #    and deduce the local number of each element type        
        for element in PT.Zone.get_ordered_elements(zone):
            assert PT.Element.CGNSName(element) not in ['MIXED', 'NGON_n', 'NFACE_n']
            elem_type = PT.get_value(element)[0]
            elem_er = PT.Element.Range(element)
            elem_ec = PT.get_node_from_name(element,'ElementConnectivity',depth=1)[1]
            elem_distrib = PT.maia.getDistribution(element, 'Element')[1]
            
            nb_nodes_per_elem = PT.Element.NVtx(element)
            nb_elem_loc = elem_distrib[1]-elem_distrib[0]
            
            mixed_partial_ec = np.zeros(nb_elem_loc*(nb_nodes_per_elem+1),dtype = elem_ec.dtype)
            mixed_partial_ec[::nb_nodes_per_elem+1] = elem_type
            for i in range(nb_nodes_per_elem):
                mixed_partial_ec[i+1::nb_nodes_per_elem+1] = elem_ec[i::nb_nodes_per_elem]
            part_data_ec.append(mixed_partial_ec)
            stride_ec = (nb_nodes_per_elem+1)*np.ones(nb_elem_loc,dtype = np.int32)
            part_stride_ec.append(stride_ec)
            
            mixed_partial_eso = (nb_nodes_per_elem+1)*np.arange(nb_elem_loc,dtype = elem_ec.dtype) + \
                                nb_nodes_prev + (nb_nodes_per_elem+1)*elem_distrib[0]
            part_data_eso.append(mixed_partial_eso)
            stride_eso = np.ones(nb_elem_loc,dtype = np.int32)
    
            ln_to_gn = np.array(range(nb_elem_loc),dtype=elem_distrib.dtype) + \
                       nb_elem_prev + elem_distrib[0] + 1
            ln_to_gn_list.append(ln_to_gn)
            
            nb_nodes_prev += (nb_nodes_per_elem+1)*PT.Element.Size(element)
            nb_elem_prev += (elem_er[1]-elem_er[0]+1)
        
        
        # 2/ Delete standard nodes and add mixed nodes
        PT.rm_nodes_from_label(zone,'Elements_t')
        elem_distrib = MUPar.uniform_distribution(nb_elem_prev,comm)
        ptb = MTP.PartToBlock(elem_distrib,ln_to_gn_list,comm)
    
        __, dist_data_eso_wo_last = ptb.exchange_field(part_data_eso)    
        dist_stride_ec, dist_data_ec = ptb.exchange_field(part_data_ec,part_stride_ec)
        
        dist_data_eso = np.empty(len(dist_data_eso_wo_last)+1,dtype=dist_data_eso_wo_last.dtype)
        dist_data_eso[:-1] = dist_data_eso_wo_last
        # If rank obtain data, the last value of local distributed ElementStartOffset is equal to 
        # the previous value of local ElementStartOffset plus the size of the last local element
        # else, all the element have been already distributed and local distributed 
        # ElementStartOffset distribution is equal to the total number of elements
        if len(dist_stride_ec) > 0:
            dist_data_eso[-1] = dist_data_eso_wo_last[-1] + dist_stride_ec[-1]
        else:
            assert rank >= nb_elem_prev
            dist_data_eso[-1] = nb_nodes_prev
        
        mixed = PT.new_Elements('Mixed','MIXED',erange=[1,nb_elem_prev],econn=dist_data_ec,parent=zone)
        eso = PT.new_DataArray('ElementStartOffset',dist_data_eso,parent=mixed)
        distri_ec = np.array((dist_data_eso[0],dist_data_eso[-1],nb_nodes_prev),dtype=elem_distrib.dtype)
        PT.maia.newDistribution({'Element' : elem_distrib, 'ElementConnectivity' : distri_ec}, parent=mixed)
