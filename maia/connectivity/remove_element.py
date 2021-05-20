import Converter.Internal as I
import numpy as np

from maia.sids import sids
from maia.sids import Internal_ext as IE
from maia.utils.py_utils import any_in_range

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
  target_range = sids.ElementRange(element)
  target_size  = sids.ElementSize(element)
  for elem in I.getNodesFromType1(zone, 'Elements_t'):
    if elem != element:
      #Shift ER
      er   = sids.ElementRange(elem)
      if target_range[0] < er[0]:
        er -= target_size

      elem_pe_n = I.getNodeFromName1(elem, 'ParentElements')
      if elem_pe_n is not None:
        elem_pe = I.getValue(elem_pe_n)
        # This will raise if PE actually refers to the section to remove
        assert not any_in_range(elem_pe, *target_range), \
            f"Can not remove element {I.getName(element)}, indexed by {I.getName(elem)}/PE"
        # We shift the element index of the sections that have been shifted
        elem_pe -= target_size * (target_range[0] < elem_pe)

  #Shift pointList
  subset_pathes = ['ZoneBC_t/BC_t', 'ZoneBC_t/BC_t/BCDataSet_t', 'ZoneGridConnectivity_t/GridConnectivity_t',\
                   'FlowSolution_t', 'ZoneSubRegion_t']
  for subset_path in subset_pathes:
    for subset in IE.getNodesFromTypeMatching(zone, subset_path):
      pl_n = I.getNodeFromName1(subset, 'PointList')
      if sids.GridLocation(subset) != 'Vertex' and pl_n is not None:
        pl = pl_n[1]
        #Ensure that PL is not refering to section to remove
        assert not any_in_range(pl, *target_range), \
          f"Can not remove element {I.getName(element)}, indexed by at least one PointList"
        #Shift
        pl -= target_size * (target_range[0] < pl)

  I._rmNode(zone, element)

