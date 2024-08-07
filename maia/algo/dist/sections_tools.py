import numpy as np

import maia.pytree as PT

from maia.utils import np_utils

def reorder_sections(tree, permutation):
  """ Reorder the sections of the input tree by appling the permutation
  function on each zone, and update all the DataArray/IndexArray refering to it.

  Permutation function will be called on a list of elt nodes, and must return a list of elts nodes
  
  Nb : this function does not manage :CGNS#GlobalNumbering arrays and does not search jns on other
  ranks, which is why partitioned trees are not supported
  """

  is_zone_u = lambda n : PT.get_label(n) == 'Zone_t' and PT.Zone.Type(n) == 'Unstructured'
  for base, zone in PT.iter_children_from_predicates(tree, ['CGNSBase_t', is_zone_u], ancestors=True):

    elts_cur_ord = PT.Zone.get_ordered_elements(zone)
    elts_new_ord = permutation(elts_cur_ord)

    cur_names = [PT.get_name(e) for e in elts_cur_ord]
    new_names = [PT.get_name(e) for e in elts_new_ord]
    assert sorted(cur_names) == sorted(new_names) # All elements must appear in new order list

    offset = [None] * len(elts_cur_ord)
    cur = 1

    # Loop in new elt order to know (by increment) the new ElementRange[0], and retrieve
    # old position of element to compute offset for this section
    for elt in elts_new_ord:
      pos = cur_names.index(PT.get_name(elt))                       # Corresponding position in original elt ordering
      offset[pos] = cur - PT.Element.Range(elts_cur_ord[pos])[0]    # Cur == new ElementRange[0] for this elt, so offset is new - old
      cur += PT.Element.Size(elt)
    assert None not in offset
    offset = np.array(offset)


    cur_idx = np_utils.sizes_to_indices([PT.Element.Size(e) for e in PT.Zone.get_ordered_elements(zone)]) # NB assert elt start at 1

    # Renumber elt data (Range, ParentElements, ElementConnectivity (if needed))
    for i,elt in enumerate(elts_cur_ord):
      PT.get_child_from_name(elt, 'ElementRange')[1] += offset[i]
      
      # Special case of NFace (connectivity is signed, and does not indicates vertices)
      if PT.Element.CGNSName(elt) == 'NFACE_n':
        ec = PT.get_child_from_name(elt, 'ElementConnectivity')
        sign = np.sign(ec[1])
        val  = np.abs(ec[1])
        r = np.searchsorted(cur_idx, val)
        ec[1][:] = sign*(val + offset[r-1])

      if (pe := PT.get_child_from_name(elt, 'ParentElements')) is not None:
        r = np.searchsorted(cur_idx, pe[1])
        pe[1] += offset[r-1] * (pe[1] > 0)

    # Renumber PointLists
    opp_zone_paths = []
    for subset in PT.iter_all_subsets(zone):
      if PT.Subset.GridLocation(subset) == 'Vertex':
        continue

      if (pr := PT.get_child_from_name(subset, 'PointRange')) is not None:
        # PointRange may cross several sections, so we extend it
        distri = PT.get_value(distri_n) if (distri_n := PT.maia.getDistribution(subset, 'Index')) is not None else None
        pl = np_utils.single_dim_pr_to_pl(pr[1], distri)
        PT.update_node(pr, 'PointList', 'IndexArray_t', pl)

      pl = PT.get_child_from_name(subset, 'PointList')
      r = np.searchsorted(cur_idx, pl[1])
      pl[1] += offset[r-1]

      if PT.get_label(subset) == 'GridConnectivity_t' and PT.GridConnectivity.is1to1(subset):
        opp_zone_paths.append(PT.GridConnectivity.ZoneDonorPath(subset, PT.get_name(base)))

    # Update PointListDonor on opposite zones
    cur_zone_path = f'{PT.get_name(base)}/{PT.get_name(zone)}'
    for opp_zone_path in set(opp_zone_paths):
      opp_base_name = PT.utils.path_head(opp_zone_path)
      opp_zone = PT.get_node_from_path(tree, opp_zone_path)
      is_gc_to_update = lambda n : PT.get_label(n) == 'GridConnectivity_t' and \
                                   PT.GridConnectivity.is1to1(n) and \
                                   PT.Subset.GridLocation(n) != 'Vertex' and \
                                   PT.GridConnectivity.ZoneDonorPath(n, opp_base_name) == cur_zone_path
      for gc in PT.get_children_from_predicates(opp_zone, ['ZoneGridConnectivity_t', is_gc_to_update]):
        pld = PT.get_child_from_name(gc, 'PointListDonor')
        r = np.searchsorted(cur_idx, pld[1])
        pld[1] += offset[r-1]
      
      
def reorder_elt_sections_from_dim(dist_tree, reverse=False):
  """ Reorder the Elements_t sections of the input tree according to their dimension.

  By default, Elements_t nodes are sorted in increasing dimension order (1D, then 2D, then 3D).
  Decreasing dimension order (3D, then 2D, then 1D) can be obtained using ``reverse=True``.

  Input tree is modified inplace.

  Args:
    dist_tree   (CGNSTree): Distributed tree
    reverse (bool, optional): If True, elements of the higher dimension get the lower ElementRange.
      Defaults to ``False``.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #reorder_elt_sections_from_dim@start
        :end-before: #reorder_elt_sections_from_dim@end
        :dedent: 2
  """

  # This is to break tie between 2 elements of same dimension
  base_elts = ['NODE', 'BAR', 'TRI', 'QUAD', 'NGON', 'TETRA', 'PYRA', 'PENTA', 'HEXA', 'NFACE']
  sign = -1 if reverse else 1 # To have increasing of decreasing dim order
  def key_func(e):
    idx = base_elts.index(PT.Element.CGNSName(e).split('_')[0])
    return (sign * PT.Element.Dimension(e), idx)

  reorder_sections(dist_tree, lambda elts: sorted(elts, key=key_func))