import maia.pytree as PT
from   maia.utils    import np_utils

import numpy as np

LOC_TO_DIM   = {'Vertex':0, 'EdgeCenter':1, 'FaceCenter':2, 'CellCenter':3}

def local_pl_offset(part_zone, dim):
  """
  Return the shift related to the element of the dimension to apply to a point_list so it starts to 1.
  This function assumes that there will be only one type of element per dimension on zone
  (because isosurface and extract_part returns trees with this property).
  Works only for ngon/nface 3D meshes and elements meshes.
  """
  if   dim == 3:
    nface = PT.Zone.NFaceNode(part_zone)
    if nface is None:
      # Extract_part and isosurfaces use trees with NGon and NFace
      raise ValueError("NGon trees must have NFace connectivity to extract pl offset")
    return PT.Element.Range(nface)[0] - 1
  elif dim == 2:
    if PT.Zone.has_ngon_elements(part_zone):
      ngon = PT.Zone.NGonNode(part_zone)
      return PT.Element.Range(ngon)[0] - 1
    else:
      tri_or_quad_elts = lambda n: PT.get_label(n)=='Elements_t' and PT.Element.Dimension(n)==2
      elt_n     = PT.get_child_from_predicate(part_zone, tri_or_quad_elts)
      return PT.Element.Range(elt_n)[0] - 1
  elif dim == 1:
    bar_elts  = lambda n: PT.get_label(n)=='Elements_t' and PT.Element.Dimension(n)==1
    elt_n     = PT.get_child_from_predicate(part_zone, bar_elts)
    return PT.Element.Range(elt_n)[0] - 1
  else:
    return 0

def get_relative_pl(container, part_zone):
  """Return the point_list node related to a container (from BC, GC or itself)."""
  if PT.get_label(container)=="FlowSolution_t":
    relative_n = container
  else:
    relative_n = PT.get_node_from_path(part_zone, PT.getSubregionExtent(container, part_zone))
  return PT.get_child_from_name(relative_n, "PointList")

def get_partial_container_stride_and_order(part_zones, container_name, gridLocation, ptp, comm):
  """
  Return two list of arrays :
    - pl_gnum1 : the order to apply to container fields defined with a point_list to fit the gnum1_come_from order
    - stride   : stride of shape gnum1_come_from indexing which elt will get data
  """
  pl_gnum1 = list()
  stride   = list()

  for i_part, part_zone in enumerate(part_zones):
    container = PT.get_child_from_name(part_zone, container_name)
    if container is not None:
      # > Get the right node to get PL (if ZSR linked to BC or GC)
      point_list_n = get_relative_pl(container, part_zone)
      point_list   = PT.get_value(point_list_n)[0] - local_pl_offset(part_zone, LOC_TO_DIM[gridLocation]) # Gnum start at 1

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
      stride_tmp = np.zeros(part_gnum1_idx[-1], dtype=np.int32)
      stride_tmp[pl_to_gnum1] = 1
      stride.append(stride_tmp)
    
  return pl_gnum1, stride
