import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils import np_utils, par_utils
from maia.transfer import protocols as EP

from .merge_ids import remove_distributed_ids

def remove_element(zone, element):
  """
  Remove one Element_t node from the Elements of the zone and perform the following
  operations :
    - Shift other elements ElementRange
    - Update other elements ParentElements
    - Update element supported PointLists
  Note : PointListDonor are not currently supported
  This function only shift arrays. Trouble will occurs later if trying to remove
  elements that are indexed by a PointList, a ParentElement or anything else.
  """
  target_range = PT.Element.Range(element)
  target_size  = PT.Element.Size(element)
  for elem in PT.get_children_from_label(zone, 'Elements_t'):
    if elem != element:
      #Shift ER
      er   = PT.Element.Range(elem)
      if target_range[0] < er[0]:
        er -= target_size

      elem_pe_n = PT.get_child_from_name(elem, 'ParentElements')
      if elem_pe_n is not None:
        elem_pe = PT.get_value(elem_pe_n)
        # This will raise if PE actually refers to the section to remove
        assert not np_utils.any_in_range(elem_pe, *target_range), \
            f"Can not remove element {PT.get_name(element)}, indexed by {PT.get_name(elem)}/PE"
        # We shift the element index of the sections that have been shifted
        elem_pe -= target_size * (target_range[0] < elem_pe)

  #Shift pointList
  subset_pathes = ['ZoneBC_t/BC_t', 'ZoneBC_t/BC_t/BCDataSet_t', 'ZoneGridConnectivity_t/GridConnectivity_t',\
                   'FlowSolution_t', 'ZoneSubRegion_t']
  for subset_path in subset_pathes:
    for subset in PT.iter_children_from_predicates(zone, subset_path):
      pl_n = PT.get_child_from_name(subset, 'PointList')
      if PT.Subset.GridLocation(subset) != 'Vertex' and pl_n is not None:
        pl = pl_n[1]
        #Ensure that PL is not refering to section to remove
        assert not np_utils.any_in_range(pl, *target_range), \
          f"Can not remove element {PT.get_name(element)}, indexed by at least one PointList"
        #Shift
        pl -= target_size * (target_range[0] < pl)

  PT.rm_child(zone, element)

def remove_ngons(dist_ngon, ngon_to_remove, comm):
  """
  Remove a list of NGon in an NGonElements distributed node.
  Global data such as distribution are updated.
  ngon_to_remove is the list of the ids of the ngon to be remove, *local*,
  (start at 0), each proc having its own list (and it must know these ngons!)

  ! This function only works on NGon : in particular, vertices or
  PointList are not removed
  """
  ngon_to_remove = np.asarray(ngon_to_remove)

  pe_n  = PT.get_child_from_name(dist_ngon, 'ParentElements')
  eso_n = PT.get_child_from_name(dist_ngon, 'ElementStartOffset')
  ec_n  = PT.get_child_from_name(dist_ngon, 'ElementConnectivity')
  er_n  = PT.get_child_from_name(dist_ngon, 'ElementRange')

  local_eso = eso_n[1] - eso_n[1][0] #Make working ElementStartOffset start at 0
  new_pe = np.delete(pe_n[1], ngon_to_remove, axis=0) #Remove faces in PE

  ec_to_remove = np_utils.multi_arange(local_eso[ngon_to_remove], local_eso[ngon_to_remove+1])
  PT.set_value(ec_n, np.delete(ec_n[1], ec_to_remove))

  # Eso is more difficult, we have to shift when deleting, locally and then globally
  new_local_eso = np_utils.sizes_to_indices(np.delete(np.diff(local_eso), ngon_to_remove))

  new_eso_shift = par_utils.gather_and_shift(new_local_eso[-1], comm)
  new_eso       = new_local_eso + new_eso_shift[comm.Get_rank()]
  PT.set_value(eso_n, new_eso)

  #Update distributions
  n_rmvd_local   = len(ngon_to_remove)
  n_rmvd_offset  = par_utils.gather_and_shift(n_rmvd_local, comm)
  n_rmvd_total   = n_rmvd_offset[-1]

  n_rmvd_ec_local   = len(ec_to_remove)
  n_rmvd_ec_offset  = par_utils.gather_and_shift(n_rmvd_ec_local, comm)
  n_rmvd_ec_total   = n_rmvd_ec_offset[-1]

  ngon_distri = PT.get_value(MT.getDistribution(dist_ngon, 'Element'))
  ngon_distri[0] -= n_rmvd_offset[comm.Get_rank()]
  ngon_distri[1] -= (n_rmvd_offset[comm.Get_rank()] + n_rmvd_local)
  ngon_distri[2] -= n_rmvd_total

  ngon_distri_ec_n = MT.getDistribution(dist_ngon, 'ElementConnectivity')
  if ngon_distri_ec_n is not None:
    ngon_distri_ec     = PT.get_value(ngon_distri_ec_n)
    ngon_distri_ec[0] -= n_rmvd_ec_offset[comm.Get_rank()]
    ngon_distri_ec[1] -= (n_rmvd_ec_offset[comm.Get_rank()] + n_rmvd_ec_local)
    ngon_distri_ec[2] -= n_rmvd_ec_total

  # If NGon were first in tree, cell range has moved so pe must be offseted
  if er_n[1][0] == 1:
    np_utils.shift_nonzeros(new_pe, -n_rmvd_total)
  PT.set_value(pe_n, new_pe)

  #Update ElementRange and size data (global)
  er_n[1][1] -= n_rmvd_total

def remove_elts_from_pl(zone, elt_n, elt_pl, comm):
  """
  For a given Elements_t node, remove the entity spectified
  in the distributed array `elt_pl`.
  Elt_pl follows cgns convention, must be included in ElementRange
  bounds of the given element node.
  The Elements node it self is updated, as well as the following data:
  - Zone dimension & distribution
  - BCs

  Note: Zone must have decreasing element ordering
  """
  DIM_TO_LOC = ['Vertex', 'EdgeCenter', 'FaceCenter', 'CellCenter']

  # > Get element information
  elt_dim    = PT.Element.Dimension(elt_n)
  elt_size   = PT.Element.NVtx(elt_n)
  elt_offset = PT.Element.Range(elt_n)[0]

  is_elt_bc = lambda n: PT.get_label(n)=='BC_t' and PT.Subset.GridLocation(n)==DIM_TO_LOC[elt_dim]

  ec_n = PT.get_child_from_name(elt_n, 'ElementConnectivity')
  ec   = PT.get_value(ec_n)
  er_n = PT.get_child_from_name(elt_n, 'ElementRange')
  er   = PT.get_value(er_n)
  old_er = np.copy(er)

  assert er[0]<=np.min(elt_pl) and np.max(elt_pl)<=er[1]

  elt_distrib_n = PT.maia.getDistribution(elt_n, distri_name='Element')
  elt_distri = elt_distrib_n[1]

  # > Get old_to_new indirection, -1 means that elt must be removed
  elt_pl_shifted = elt_pl - elt_offset + 1 # Elt_pl, but starting at 1 (remove cgns offset)
  old_to_new_elt = remove_distributed_ids(elt_distri, elt_pl_shifted, comm)
  ids_to_remove = np.where(old_to_new_elt==-1)[0]

  n_elt_to_rm_l = ids_to_remove.size
  rm_distrib = par_utils.dn_to_distribution(n_elt_to_rm_l, comm)
  n_elt_to_rm = rm_distrib[2]

  # > Updating element range, connectivity and distribution
  pl_c  = -np.ones(n_elt_to_rm_l*elt_size, dtype=np.int32)
  for i_size in range(elt_size):
    pl_c[i_size::elt_size] = elt_size*ids_to_remove+i_size
  ec = np.delete(ec, pl_c)
  PT.set_value(ec_n, ec)
  er[1] -= n_elt_to_rm

  n_elt = elt_distri[1]-elt_distri[0]
  new_elt_distrib = par_utils.dn_to_distribution(n_elt-n_elt_to_rm_l, comm)
  if new_elt_distrib[2]==0:
    PT.rm_child(zone, elt_n)
  else:
    PT.set_value(elt_distrib_n, new_elt_distrib)


  # > Update BC PointList related to removed element node
  zone_bc_n = PT.get_child_from_label(zone, 'ZoneBC_t')
  for bc_n in PT.get_children_from_predicate(zone_bc_n, is_elt_bc):
    bc_pl_n = PT.get_child_from_name(bc_n, 'PointList')
    bc_pl   = PT.get_value(bc_pl_n)[0]
    mask_in_elt = (old_er[0] <= bc_pl ) & (bc_pl<=old_er[1])

    # > Update numbering of PointList defined over other element nodes
    new_bc_pl  = bc_pl
    bc_elt_ids = new_bc_pl[mask_in_elt]-elt_offset+1
    new_gn = EP.block_to_part(old_to_new_elt, elt_distri, [bc_elt_ids], comm)[0]
    new_gn[new_gn>0] += elt_offset-1
    if mask_in_elt.any():
      new_bc_pl[mask_in_elt] = new_gn
    new_bc_pl = new_bc_pl[new_bc_pl>0]

    new_bc_distrib = par_utils.dn_to_distribution(new_bc_pl.size, comm)

    # > Update BC
    if new_bc_distrib[2]==0:
      PT.rm_child(zone_bc_n, bc_n)
    else:
      PT.set_value(bc_pl_n, new_bc_pl.reshape((1,-1), order='F'))
      PT.maia.newDistribution({'Index' : new_bc_distrib}, bc_n)

  # > Update zone size and distribution
  if elt_dim==PT.Zone.CellDimension(zone):
    zone[1][:,1] -= n_elt_to_rm
    distri_cell = PT.maia.getDistribution(zone, 'Cell')[1]
    distri_cell -= rm_distrib

  # > Shift other Element Range, if the have higher ids
  for elt_n in PT.get_nodes_from_predicate(zone, 'Elements_t'):
    elt_range = PT.Element.Range(elt_n)
    if elt_range[0] > old_er[1]:
      elt_range -= n_elt_to_rm
  # > Shift PointList related to elements with higher ids
  for bc_n in PT.get_nodes_from_predicates(zone, 'ZoneBC_t/BC_t'):
    pl = PT.get_child_from_name(bc_n, 'PointList')[1]
    pl[old_er[1]<pl] -= n_elt_to_rm
