from   mpi4py             import MPI
import numpy              as     np

import maia.pytree        as PT

from maia.utils            import np_utils, as_pdm_gnum
from maia.algo.dist.s_to_u import compute_transform_matrix, apply_transform_matrix,\
                                  gc_is_reference, guess_bnd_normal_index

def check_datasize(tree):
  """
  Warns if a heavy array is not distributed
  """
  is_heavy_node = lambda n: PT.get_label(n) in ['DataArray_t', 'IndexArray_t'] \
                            and n[1] is not None and n[1].size > 100
  heavy_arrays = [PT.get_name(node) for node in \
          PT.iter_nodes_from_predicate(tree, is_heavy_node, explore='deep')]

  if heavy_arrays:
    print(f"Warning -- Some heavy data are not distributed: {heavy_arrays}")

def fix_zone_datatype(size_tree, size_data):
  """
  Cassiopee always read zones as int32. Fix it if input type was int64
  """
  for zone_path in PT.predicates_to_paths(size_tree, "CGNSBase_t/Zone_t"):
    if size_data["/" + zone_path][1] == 'I8':
      zone = PT.get_node_from_path(size_tree, zone_path)
      zone[1] = zone[1].astype(np.int64)

def fix_point_ranges(size_tree):
  """
  Permute start and end of PointRange or PointRangeDonor nodes found in GridConnectivity1to1_t
  in order to
  a. be consistent with the transform node
  b. keep the symmetry PR|a->b = PRDonor|b->a
  """
  permuted = False
  gc_t_path = 'CGNSBase_t/Zone_t/ZoneGridConnectivity_t/GridConnectivity1to1_t'
  for base, zone, zgc, gc in PT.iter_children_from_predicates(size_tree, gc_t_path, ancestors=True):
    base_name = PT.get_name(base)
    zone_name = PT.get_name(zone)
    gc_path     = base_name + '/' + zone_name
    gc_opp_path = PT.get_value(gc)
    if not '/' in gc_opp_path:
      gc_opp_path = base_name + '/' + gc_opp_path
    # WARNING: for hybrid case structured zone could have PointList, PointListDonor.
    if PT.get_child_from_label(gc, 'IndexRange_t') is not None:
      transform     = PT.get_value(PT.get_child_from_name(gc, 'Transform'))
      point_range   = PT.get_value(PT.get_child_from_name(gc, 'PointRange'))
      point_range_d = PT.get_value(PT.get_child_from_name(gc, 'PointRangeDonor'))

      donor_dir    = abs(transform) - 1
      nb_points    = point_range[:,1] - point_range[:,0]
      nb_points_d  = np.sign(transform)*(point_range_d[donor_dir,1] - point_range_d[donor_dir,0])
      dir_to_swap  = (nb_points != nb_points_d)

      if dir_to_swap.any():
        permuted = True
        if gc_is_reference(gc, gc_path, gc_opp_path):

          opp_dir_to_swap = np.empty_like(dir_to_swap)
          opp_dir_to_swap[donor_dir] = dir_to_swap

          point_range_d[opp_dir_to_swap, 0], point_range_d[opp_dir_to_swap, 1] = \
              point_range_d[opp_dir_to_swap, 1], point_range_d[opp_dir_to_swap, 0]
        else:
          point_range[dir_to_swap, 0], point_range[dir_to_swap, 1] = \
              point_range[dir_to_swap, 1], point_range[dir_to_swap, 0]

      T = compute_transform_matrix(transform)
      assert (point_range_d[:,1] == \
          apply_transform_matrix(point_range[:,1], point_range[:,0], point_range_d[:,0], T)).all()
  if permuted:
    print(f"Warning -- Some GridConnectivity1to1_t PointRange have been swapped because Transform specification was invalid")

def add_missing_pr_in_bcdataset(tree):
  """
  When the GridLocation values of BC and BCDataSet are respectively 'Vertex' and '*FaceCenter',
  if the PointRange is not given in the BCDataSet, the function compute it
  Remark : if the shape of DataArrays in BCDataSet is coherent with a '*FaceCenter' GridLocation
  but the GridLocation node is not defined, this function does not add the PointRange
  """
  pr_added = False
  bc_t_path = 'CGNSBase_t/Zone_t/ZoneBC_t/BC_t'
  for base, zone, zbc, bc in PT.iter_children_from_predicates(tree, bc_t_path, ancestors=True):
    if PT.get_value(PT.get_child_from_label(zone, 'ZoneType_t')) == 'Unstructured':
      continue
    if PT.get_child_from_label(bc, 'BCDataSet_t') is None:
      continue
    bc_grid_location = PT.Subset.GridLocation(bc)
    bc_point_range   = PT.get_value(PT.get_child_from_name(bc, 'PointRange'))
    for bcds in PT.get_children_from_label(bc, 'BCDataSet_t'):
      if PT.get_child_from_name(bcds, 'PointRange') is not None:
        continue
      bcds_grid_location = PT.Subset.GridLocation(bcds)
      if not (bcds_grid_location.endswith('FaceCenter') and bc_grid_location == 'Vertex'):
        continue
      face_dir   = guess_bnd_normal_index(bc_point_range,  bc_grid_location)
      bcds_point_range             = bc_point_range.copy(order='F')
      bcds_point_range[:,1]       -= 1
      bcds_point_range[face_dir,1] = bcds_point_range[face_dir,0]
      new_pr = PT.new_PointRange(value=bcds_point_range, parent=bcds)
      pr_added = True
      # print(f"Warning -- PointRange has been added on BCDataSet {zone[0]}/{bc[0]}/{bcds[0]}"
             # " since data shape was no consistent with BC PointRange")
  if pr_added:
    print(f"Warning -- PointRange has been added on some BCDataSet nodes because data size was inconsistent")

def _enforce_pdm_dtype(tree):
  """
  Convert the index & connectivity arrays to expected pdm_g_num_t
  TODO : find better pattern for the "Subset iterator" and factorize it
  """
  for zone in PT.get_all_Zone_t(tree):
    zone[1] = as_pdm_gnum(zone[1])
    for elmt in PT.iter_children_from_label(zone, 'Elements_t'):
      for name in ['ElementRange', 'ElementConnectivity', 'ElementStartOffset', 'ParentElements']:
        node = PT.get_child_from_name(elmt, name)
        if node:
          node[1] = as_pdm_gnum(node[1])
    for pl in PT.iter_nodes_from_label(zone, 'IndexArray_t'):
      pl[1] = as_pdm_gnum(pl[1])
   
def ensure_PE_global_indexing(dist_tree):
  """
  This function ensures that the ParentElements array of the NGonElements, if existing,
  is compliant with the CGNS standard ie refers faces using absolute numbering.
  This function works under the following assumptions (which could be released,
  but also seems to be imposed by the standard)
   - At most one NGonElements node exists
   - NGonElements and standard elements can not be mixed together
  """
  n_shifted = 0
  for zone in PT.get_all_Zone_t(dist_tree):
    elts = PT.get_children_from_label(zone, 'Elements_t')
    ngon_nodes = [elt for elt in elts if PT.Element.CGNSName(elt)=='NGON_n']
    oth_nodes  = [elt for elt in elts if PT.Element.CGNSName(elt)!='NGON_n']
    if ngon_nodes == []:
      continue
    elif len(ngon_nodes) == 1:
      if len(oth_nodes) > 1 or (len(oth_nodes) == 1 and PT.Element.CGNSName(oth_nodes[0]) != 'NFACE_n'):
        raise RuntimeError(f"Zone {PT.get_name(zone)} has both NGon and Std elements nodes, which is not supported")
    else:
      raise RuntimeError(f"Multiple NGon nodes found in zone {PT.get_name(zone)}")

    ngon_n = ngon_nodes[0]
    ngon_pe_n = PT.get_child_from_name(ngon_n, 'ParentElements')
    if ngon_pe_n:
      n_faces = PT.Element.Size(ngon_n)
      ngon_pe = ngon_pe_n[1]
      if PT.Element.Range(ngon_n)[0] == 1 and ngon_pe.shape[0] > 0 and ngon_pe[0].max() <= n_faces:
        np_utils.shift_nonzeros(ngon_pe, n_faces)
        n_shifted += 1

  return n_shifted

def rm_legacy_nodes(tree):
  eh_paths = PT.predicates_to_paths(tree, 'CGNSBase_t/Zone_t/:elsA#Hybrid')
  if len(eh_paths) > 0:
    print(f"Warning -- Legacy nodes ':elsA#Hybrid' skipped when reading file")
    for eh_path in eh_paths:
      PT.rm_node_from_path(tree, eh_path)
  
  arrays_removed = False
  for zone in PT.iter_all_Zone_t(tree):
    for fs in PT.iter_nodes_from_label(zone, 'FlowSolution_t'):
      all_arrays  = PT.get_names(PT.get_children_from_label(fs, 'DataArray_t'))
      size_arrays = [name for name in all_arrays if name.endswith('#Size')]
      has_no_size = lambda n: not PT.get_name(n).endswith('#Size') and f'{PT.get_name(n)}#Size' not in size_arrays

      n_child_bck = len(PT.get_children(fs))
      PT.rm_children_from_predicate(fs, lambda n: PT.get_label(n) == 'DataArray_t' and has_no_size(n))
      arrays_removed = arrays_removed or len(PT.get_children(fs)) < n_child_bck
  if arrays_removed:
    print(f"Warning -- Some empty arrays under FlowSolution_t nodes have been skipped when reading file")
