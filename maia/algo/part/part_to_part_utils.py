import maia.pytree as PT
from   maia.utils    import np_utils

import Pypdm.Pypdm as PDM

import numpy as np

LOC_TO_DIM   = {'Vertex':0, 'EdgeCenter':1, 'FaceCenter':2, 'CellCenter':3}

def local_pl_offset(part_zone, dim):
  # Works only for ngon / nface 3D meshes and elements meshes with one type of element per dim
  if   dim == 3:
    nface = PT.Zone.NFaceNode(part_zone)
    return PT.Element.Range(nface)[0] - 1
  elif dim == 2:
    if PT.Zone.has_ngon_elements(part_zone):
      ngon = PT.Zone.NGonNode(part_zone)
      return PT.Element.Range(ngon)[0] - 1
    else:
      tri_or_quad_elts = lambda n: PT.get_label(n)=='Elements_t' and PT.get_name(n) in ['TRI_3', 'QUAD_4']
      elt_n     = PT.get_child_from_predicate(part_zone, tri_or_quad_elts)
      return PT.Element.Range(elt_n)[0] - 1
  elif dim == 1:
    bar_elts  = lambda n: PT.get_label(n)=='Elements_t' and PT.get_name(n) in ['BAR_2']
    elt_n     = PT.get_child_from_predicate(part_zone, bar_elts)
    return PT.Element.Range(elt_n)[0] - 1
  else:
    return 0

def get_relative_pl(node, part_zone):
  ref_zsr_node = node
  bc_descriptor_n = PT.get_child_from_name(node, 'BCRegionName')
  gc_descriptor_n = PT.get_child_from_name(node, 'GridConnectivityRegionName')
  assert not (bc_descriptor_n and gc_descriptor_n)
  if bc_descriptor_n is not None:
    bc_name      = PT.get_value(bc_descriptor_n)
    ref_zsr_node = PT.get_child_from_predicates(part_zone, f'ZoneBC/{bc_name}')
  elif gc_descriptor_n is not None:
    gc_name      = PT.get_value(gc_descriptor_n)
    ref_zsr_node = PT.get_child_from_predicates(part_zone, f'ZoneGridConnectivity_t/{gc_name}')
  point_list_node  = PT.get_child_from_name(ref_zsr_node, 'PointList')
  return point_list_node

def get_partial_container_stride_and_order(part_zones, container_name, gridLocation, ptp, comm):
  pl_gnum1 = list()
  stride   = list()

  for i_part, part_zone in enumerate(part_zones):
    container = PT.get_child_from_name(part_zone, container_name)
    if container is not None:
      # > Get the right node to get PL (if ZSR linked to BC or GC)
      point_list_node = get_relative_pl(container, part_zone)
      point_list  = point_list_node[1][0] - local_pl_offset(part_zone, LOC_TO_DIM[gridLocation]) # Gnum start at 1

    # Get p2p gnums
    part_gnum1_idx = ptp.get_gnum1_come_from() [i_part]['come_from_idx'] # Get partition order
    part_gnum1     = ptp.get_gnum1_come_from() [i_part]['come_from']     # Get partition order
    ref_lnum2      = ptp.get_referenced_lnum2()[i_part]                  # Get partition order

    if container is None or point_list.size==0 or ref_lnum2.size==0:
      stride_tmp   = np.zeros(part_gnum1_idx[-1],dtype=np.int32)
      pl_gnum1_tmp = np.empty(0,dtype=np.int32)
      stride  .append(stride_tmp)
      pl_gnum1.append(pl_gnum1_tmp)
    else:
      order    = np.argsort(ref_lnum2)                 # Sort order of point_list ()
      idx      = np.searchsorted(ref_lnum2,point_list,sorter=order)
      pl_mask  = point_list==ref_lnum2[np.take(order, idx, mode='clip')]
      true_idx = idx[pl_mask]

      # Number of part1 elements in an element of part2 
      n_elt_of1_in2 = np.diff(part_gnum1_idx)[true_idx]

      sort_true_idx = np.argsort(true_idx)

      # PL in part2 order
      pl_gnum1_tmp = np.arange(0, point_list.shape[0], dtype=np.int32)[pl_mask][sort_true_idx]
      pl_gnum1_tmp = np.repeat(pl_gnum1_tmp, n_elt_of1_in2)
      pl_gnum1.append(pl_gnum1_tmp)

      # PL in gnum1 order
      pl_to_gnum1_start = part_gnum1_idx[true_idx]         
      pl_to_gnum1_stop  = pl_to_gnum1_start+n_elt_of1_in2
      pl_to_gnum1 = np_utils.multi_arange(pl_to_gnum1_start, pl_to_gnum1_stop)
      
      # Stride variable
      stride_tmp    = np.zeros(part_gnum1_idx[-1], dtype=np.int32)
      stride_tmp[pl_to_gnum1] = 1
      stride.append(stride_tmp)
    
  return pl_gnum1, stride
