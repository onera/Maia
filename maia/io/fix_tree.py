from   mpi4py             import MPI
import numpy              as     np
import Converter.Filter   as     CFilter

import maia.pytree        as PT

from maia                  import npy_pdm_gnum_dtype
from maia.utils            import np_utils
from maia.algo.dist.s_to_u import compute_transform_matrix, apply_transform_matrix, gc_is_reference

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

def load_grid_connectivity_property(filename, tree):
  """
  Load the GridConnectivityProperty_t nodes that may be present in joins.
  Because the transformation data is stored as numpy array, these nodes
  are not loaded on the previous step.
  """
  # Prepare pathes
  zgc_t_path = 'CGNSBase_t/Zone_t/ZoneGridConnectivity_t'
  is_gc = lambda n : PT.get_label(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t']
  gc_prop_pathes = []
  for base,zone,zone_gc in PT.iter_children_from_predicates(tree, zgc_t_path, ancestors=True):
    for gc in PT.iter_children_from_predicate(zone_gc, is_gc):
      gc_prop = PT.get_child_from_label(gc, 'GridConnectivityProperty_t')
      if gc_prop is not None:
        gc_prop_path = '/'.join([base[0], zone[0], zone_gc[0], gc[0], gc_prop[0]])
        gc_prop_pathes.append(gc_prop_path)

  # Load
  gc_prop_nodes = CFilter.readNodesFromPaths(filename, gc_prop_pathes)

  # Replace with loaded data
  for path, gc_prop in zip(gc_prop_pathes, gc_prop_nodes):
    gc_node_path = '/'.join(path.split('/')[:-1])
    gc_node = PT.get_node_from_path(tree, gc_node_path)
    PT.rm_children_from_label(gc_node, 'GridConnectivityProperty_t')
    PT.add_child(gc_node, gc_prop)

def _enforce_pdm_dtype(tree):
  """
  Convert the index & connectivity arrays to expected pdm_g_num_t
  TODO : find better pattern for the "Subset iterator" and factorize it
  """
  for zone in PT.get_all_Zone_t(tree):
    zone[1] = zone[1].astype(npy_pdm_gnum_dtype)
    for elmt in PT.iter_children_from_label(zone, 'Elements_t'):
      for name in ['ElementRange', 'ElementConnectivity', 'ElementStartOffset', 'ParentElements']:
        node = PT.get_child_from_name(elmt, name)
        if node:
          node[1] = node[1].astype(npy_pdm_gnum_dtype)
    for pl in PT.iter_nodes_from_label(zone, 'IndexArray_t'):
      pl[1] = pl[1].astype(npy_pdm_gnum_dtype)
   
def ensure_PE_global_indexing(dist_tree):
  """
  This function ensures that the ParentElements array of the NGonElements, if existing,
  is compliant with the CGNS standard ie refers faces using absolute numbering.
  This function works under the following assumptions (which could be released,
  but also seems to be imposed by the standard)
   - At most one NGonElements node exists
   - NGonElements and standard elements can not be mixed together
  """
  for zone in PT.get_all_Zone_t(dist_tree):
    elts = PT.get_children_from_label(zone, 'Elements_t')
    ngon_nodes = [elt for elt in elts if PT.Element.CGNSName(elt)=='NGON_n']
    oth_nodes  = [elt for elt in elts if PT.Element.CGNSName(elt)!='NGON_n']
    if ngon_nodes == []:
      return
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
        print(f"Warning -- NGon/ParentElements have been shift on zone {PT.get_name(zone)} to be CGNS compliant")
