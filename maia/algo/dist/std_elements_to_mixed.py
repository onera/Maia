import mpi4py.MPI as mpi

import os
import numpy as np

import maia

from maia            import pytree        as MP
from maia.io         import cgns_io_tree  as IOT
from maia.algo       import dist          as MAD
from maia.algo.dist  import s_to_u        as CSU
from maia.algo.dist  import merge         as ME
from maia.transfer   import protocols     as MTP
from maia.utils      import par_utils     as MUPar

import maia.pytree.sids.elements_utils    as MPSEU


def convert_elements_to_mixed(dist_tree, comm):
    """
    Transform an element based connectivity into a mixed connectivity.
    
    Tree is modified in place : standard elements are removed from the zones.
    Remark : no need to update PointList nodes because the elements order is
    preserved.
  
    Args:
      dist_tree  (CGNSTree): Tree with connectivity described by standard elements
      comm       (`MPIComm`) : MPI communicator
  
    Example:
        .. literalinclude:: snippets/test_algo.py
          :start-after: #convert_elements_to_mixed@start
          :end-before: #convert_elements_to_mixed@end
          :dedent: 2
    """
    
    for zone in MP.get_all_Zone_t(dist_tree):
        part_data_ec = []
        part_data_eso = []
        part_stride_ec = []
        ln_to_gn_list = []
        nb_nodes_prev = 0
        nb_elem_prev = 0
        
        
        # 1/ Create local mixed connectivity and element start offeset tab for each element node
        #    and deduce the local number of each element type        
        for element in MP.Zone.get_ordered_elements(zone):
            assert MP.get_value(element)[0] not in [20,22,23]
            elem_type = MP.get_value(element)[0]
            elem_er = MP.get_node_from_name(element,'ElementRange',depth=1)[1]
            elem_ec = MP.get_node_from_name(element,'ElementConnectivity',depth=1)[1]
            elem_distrib = MP.get_node_from_path(element,':CGNS#Distribution/Element')[1]
            
            nb_nodes_per_elem = MPSEU.element_number_of_nodes(elem_type)
            nb_elem_loc = elem_distrib[1]-elem_distrib[0]
            
            mixed_partial_ec = np.zeros(nb_elem_loc*(nb_nodes_per_elem+1),dtype = elem_ec.dtype)
            mixed_partial_ec[::nb_nodes_per_elem+1] = elem_type
            for i in range(nb_nodes_per_elem):
                mixed_partial_ec[i+1::nb_nodes_per_elem+1] = elem_ec[i::nb_nodes_per_elem]
            part_data_ec.append(mixed_partial_ec)
            stride_ec = (nb_nodes_per_elem+1)*np.ones(elem_distrib[1]-elem_distrib[0],dtype = np.int32)
            part_stride_ec.append(stride_ec)
            
            mixed_partial_eso = (nb_nodes_per_elem+1)*np.arange(nb_elem_loc,dtype = elem_ec.dtype) + \
                                nb_nodes_prev + (nb_nodes_per_elem+1)*elem_distrib[0]
            part_data_eso.append(mixed_partial_eso)
            stride_eso = np.ones(elem_distrib[1]-elem_distrib[0],dtype = np.int32)
    
            ln_to_gn = np.array(range(elem_distrib[1]-elem_distrib[0]),dtype=elem_distrib.dtype) + \
                       nb_elem_prev + elem_distrib[0] + 1
            ln_to_gn_list.append(ln_to_gn)
            
            nb_nodes_prev += (nb_nodes_per_elem+1)*(elem_er[1]-elem_er[0]+1)
            nb_elem_prev += (elem_er[1]-elem_er[0]+1)
        
        
        # 2/ Delete standard nodes and add mixed nodes
        MP.rm_nodes_from_label(zone,'Elements_t')
        elem_distrib = MUPar.uniform_distribution(nb_elem_prev,comm)
        ptb = MTP.PartToBlock(elem_distrib,ln_to_gn_list,comm)
    
        __, dist_data_eso_wo_last = ptb.exchange_field(part_data_eso)    
        dist_stride_ec, dist_data_ec = ptb.exchange_field(part_data_ec,part_stride_ec)
        
        dist_data_eso = np.empty(len(dist_data_eso_wo_last)+1,dtype=dist_data_eso_wo_last.dtype)
        dist_data_eso[:-1] = dist_data_eso_wo_last
        dist_data_eso[-1] = dist_data_eso_wo_last[-1] + dist_stride_ec[-1]
        
        mixed = MP.new_Elements('Mixed','MIXED',erange=[1,nb_elem_prev],econn=dist_data_ec,parent=zone)
        eso = MP.new_DataArray('ElementStartOffset',dist_data_eso,parent=mixed)
        mixed_distrib = MP.new_node(name=':CGNS#Distribution', label='UserDefined_t', parent=mixed)
        MP.new_DataArray('Element', elem_distrib, parent=mixed_distrib)
        distri_ec = np.array((dist_data_eso[0],dist_data_eso[-1],nb_nodes_prev),dtype=elem_distrib.dtype)
        MP.new_DataArray('ElementConnectivity', distri_ec, parent=mixed_distrib)
