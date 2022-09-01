import Converter.Internal as I
import numpy as np

import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils import np_utils, par_utils

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
        elem_pe = I.getValue(elem_pe_n)
        # This will raise if PE actually refers to the section to remove
        assert not np_utils.any_in_range(elem_pe, *target_range), \
            f"Can not remove element {I.getName(element)}, indexed by {I.getName(elem)}/PE"
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
          f"Can not remove element {I.getName(element)}, indexed by at least one PointList"
        #Shift
        pl -= target_size * (target_range[0] < pl)

  I._rmNode(zone, element)

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
  I.setValue(ec_n, np.delete(ec_n[1], ec_to_remove))

  # Eso is more difficult, we have to shift when deleting, locally and then globally
  new_local_eso = np_utils.sizes_to_indices(np.delete(np.diff(local_eso), ngon_to_remove))

  new_eso_shift = par_utils.gather_and_shift(new_local_eso[-1], comm)
  new_eso       = new_local_eso + new_eso_shift[comm.Get_rank()]
  I.setValue(eso_n, new_eso)

  #Update distributions
  n_rmvd_local   = len(ngon_to_remove)
  n_rmvd_offset  = par_utils.gather_and_shift(n_rmvd_local, comm)
  n_rmvd_total   = n_rmvd_offset[-1]

  n_rmvd_ec_local   = len(ec_to_remove)
  n_rmvd_ec_offset  = par_utils.gather_and_shift(n_rmvd_ec_local, comm)
  n_rmvd_ec_total   = n_rmvd_ec_offset[-1]

  ngon_distri = I.getVal(MT.getDistribution(dist_ngon, 'Element'))
  ngon_distri[0] -= n_rmvd_offset[comm.Get_rank()]
  ngon_distri[1] -= (n_rmvd_offset[comm.Get_rank()] + n_rmvd_local)
  ngon_distri[2] -= n_rmvd_total

  ngon_distri_ec_n = MT.getDistribution(dist_ngon, 'ElementConnectivity')
  if ngon_distri_ec_n is not None:
    ngon_distri_ec     = I.getVal(ngon_distri_ec_n)
    ngon_distri_ec[0] -= n_rmvd_ec_offset[comm.Get_rank()]
    ngon_distri_ec[1] -= (n_rmvd_ec_offset[comm.Get_rank()] + n_rmvd_ec_local)
    ngon_distri_ec[2] -= n_rmvd_ec_total

  # If NGon were first in tree, cell range has moved so pe must be offseted
  if er_n[1][0] == 1:
    np_utils.shift_nonzeros(new_pe, -n_rmvd_total)
  I.setValue(pe_n, new_pe)

  #Update ElementRange and size data (global)
  er_n[1][1] -= n_rmvd_total
