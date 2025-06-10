import numpy as np

from maia.typing import *
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.transfer import protocols as EP
from maia.utils import np_utils, par_utils

def concatenate_elt_sections(dist_tree: CGNSDistTree, comm: MPIComm) -> None:
  """ Gather the Element_t sections of same ElementType into a single one.

  Resulting sections are named after their ElementType. Note that :

  - Sections of same kind must be contiguous to be gathered. This can be achieved
    using :func:`reorder_elt_sections_from_dim` function.
  - ``NGON_n``, ``NFACE_n`` and ``MIXED`` element kind are not supported.

  Input tree is modified inplace.

  Args:
    dist_tree (CGNSDistTree) : Distributed tree
    comm (MPIComm)           : MPI communicator

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #concatenate_elt_sections@start
        :end-before: #concatenate_elt_sections@end
        :dedent: 2
  """
  MT.check_cgns_dist_tree(dist_tree)
  for zone in PT.iter_all_Zone_t(dist_tree):

    to_gather:Dict[str, List[CGNSTree]] = {}
    for elt in PT.get_children_from_label(zone, 'Elements_t'):
      if (kind := PT.Element.CGNSName(elt)) in to_gather:
        to_gather[kind].append(elt)
      else:
        to_gather[kind] = [elt]
    
    # Dont forget to sort ! Because order of apparition in tree is not
    # necessarily increasing
    to_gather = {kind : sorted(elts, key=lambda e: PT.Element.Range(e)[0]) \
                 for kind, elts in to_gather.items()}

    # Sections can be concatenated only if they are contiguous
    for elts in to_gather.values():
      if len(elts) > 1:
        for prev, elt in zip(elts[:-1], elts[1:]):
          if PT.Element.Range(elt)[0] != (PT.Element.Range(prev)[1]+1):
            msg = "Element sections of same kind are not contiguous, and thus can not be concatenated.\n"\
                  "Consider reordering the elements, for example with reorder_elt_sections_from_dim function."
            raise RuntimeError(msg)

    for kind, elts in to_gather.items():
      if len(elts) > 1:
        elts = sorted(elts, key=lambda e: PT.Element.Range(e)[0]) # Dont forget to sort!
        tot_size = sum([PT.Element.Size(e) for e in elts])
        merged_distri = par_utils.uniform_distribution(tot_size, comm)

        # Initially, each section is distributed, we need to "uninterlace" 
        # to map global distribution without changing order
        start = 0
        ec_to_merge = []
        for elt in elts:
          end = start + PT.Element.Size(elt)
          distri = MT.distribution_value(elt, 'Element')
          ec = PT.find_child_from_name(elt, 'ElementConnectivity')[1]
          distri_out = distri.copy()
          distri_out[0] = max(min(merged_distri[0], end), start) - start
          distri_out[1] = max(min(merged_distri[1], end), start) - start

          # NB : if we had block_to_block with preallocated buffer,
          # we could directly fill global array
          btb = EP.BlockToBlock(distri, distri_out, comm)
          ec_to_merge.append(btb.exchange(ec, PT.Element.NVtx(elt)))
          start = end

        merged_ec = np_utils.concatenate_np_arrays(ec_to_merge)[1]
        merged_range = np.empty(2, merged_ec.dtype)
        merged_range[0] = PT.Element.Range(elts[0] )[0]
        merged_range[1] = PT.Element.Range(elts[-1])[1]
        merged_elt = PT.new_Elements(f'{kind}', kind, erange=merged_range, econn=merged_ec)
        MT.new_Distribution({'Element' : merged_distri}, merged_elt)

        for elt in elts:
          PT.rm_child(zone, elt)
        PT.add_child(zone, merged_elt)

      else:
        # To be consistent, we just rename using elt kind
        PT.set_name(elts[0], kind)
    


def reorder_sections(tree:CGNSTree, permutation:Callable[[List[CGNSTree]], List[CGNSTree]]) -> None:
  """ Reorder the sections of the input tree by appling the permutation
  function on each zone, and update all the DataArray/IndexArray refering to it.

  Permutation function will be called on a list of elt nodes, and must return a list of elts nodes
  
  Nb : this function does not manage :CGNS#GlobalNumbering arrays and does not search jns on other
  ranks, which is why partitioned trees are not supported
  """

  for base, zone in PT.iter_children_from_predicates(tree, ['CGNSBase_t', PT.pred.IS_U_ZONE], ancestors=True):

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
    offset = np.array(offset) #type:ignore[assignment] #(reuse same var)


    cur_idx = np_utils.sizes_to_indices([PT.Element.Size(e) for e in PT.Zone.get_ordered_elements(zone)]) # NB assert elt start at 1

    # Renumber elt data (Range, ParentElements, ElementConnectivity (if needed))
    for i,elt in enumerate(elts_cur_ord):
      erange = PT.Element.Range(elt)
      erange += offset[i]
      
      # Special case of NFace (connectivity is signed, and does not indicates vertices)
      if PT.Element.CGNSName(elt) == 'NFACE_n':
        ec = PT.find_child_from_name(elt, 'ElementConnectivity')
        ec_val = PT.get_np_value(ec)
        sign = np.sign(ec_val)
        val  = np.abs(ec_val)
        r = np.searchsorted(cur_idx, val)
        ec_val[:] = sign*(val + offset[r-1])

      if (pe := PT.get_child_from_name(elt, 'ParentElements')) is not None:
        pe_val = PT.get_np_value(pe)
        r = np.searchsorted(cur_idx, pe_val)
        pe_val += offset[r-1] * (pe_val > 0)

    # Renumber PointLists
    opp_zone_paths = []
    for subset in PT.iter_all_subsets(zone):
      if PT.Subset.GridLocation(subset) == 'Vertex':
        continue

      if (pr := PT.get_child_from_name(subset, 'PointRange')) is not None:
        # PointRange may cross several sections, so we extend it
        distri = PT.get_np_value(distri_n) if (distri_n := MT.get_Distribution(subset, 'Index')) is not None else None
        new_pl = np_utils.single_dim_pr_to_pl(PT.get_np_value(pr), distri)
        PT.update_node(pr, 'PointList', 'IndexArray_t', new_pl)

      pl = PT.find_child_from_name(subset, 'PointList')
      pl_value = PT.get_np_value(pl)
      r = np.searchsorted(cur_idx, pl_value)
      pl_value += offset[r-1]

      if PT.get_label(subset) == 'GridConnectivity_t' and PT.GridConnectivity.is1to1(subset):
        opp_zone_paths.append(PT.GridConnectivity.ZoneDonorPath(subset, PT.get_name(base)))

    # Update PointListDonor on opposite zones
    cur_zone_path = f'{PT.get_name(base)}/{PT.get_name(zone)}'
    for opp_zone_path in set(opp_zone_paths):
      opp_base_name = PT.utils.path_head(opp_zone_path)
      opp_zone = PT.find_node_from_path(tree, opp_zone_path)
      matches_zone = PT.pred.UnaryPredicate(lambda n : PT.GridConnectivity.ZoneDonorPath(n, opp_base_name) == cur_zone_path)
      is_gc_to_update = PT.pred.is_gc_with(match=True) & ~PT.pred.has_location('Vertex') & matches_zone
      for gc in PT.get_children_from_predicates(opp_zone, ['ZoneGridConnectivity_t', is_gc_to_update]):
        pld = PT.find_child_from_name(gc, 'PointListDonor')
        pld_value = PT.get_np_value(pld)
        r = np.searchsorted(cur_idx, pld_value)
        pld_value += offset[r-1]
      
      
def reorder_elt_sections_from_dim(dist_tree: CGNSDistTree, reverse: bool = False) -> None:
  """ Reorder the Elements_t sections of the input tree according to their dimension.

  By default, Elements_t nodes are sorted in increasing dimension order (1D, then 2D, then 3D).
  Decreasing dimension order (3D, then 2D, then 1D) can be obtained using ``reverse=True``.

  Input tree is modified inplace.

  Args:
    dist_tree (CGNSDistTree)  : Distributed tree
    reverse (bool, optional)  : If True, elements of the higher dimension get the lower ElementRange.
      Defaults to ``False``.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #reorder_elt_sections_from_dim@start
        :end-before: #reorder_elt_sections_from_dim@end
        :dedent: 2
  """
  MT.check_cgns_dist_tree(dist_tree)
  # This is to break tie between 2 elements of same dimension
  base_elts = ['NODE', 'BAR', 'TRI', 'QUAD', 'NGON', 'TETRA', 'PYRA', 'PENTA', 'HEXA', 'NFACE']
  sign = -1 if reverse else 1 # To have increasing of decreasing dim order
  def key_func(e):
    idx = base_elts.index(PT.Element.CGNSName(e).split('_')[0])
    return (sign * PT.Element.Dimension(e), idx)

  reorder_sections(dist_tree, lambda elts: sorted(elts, key=key_func))
