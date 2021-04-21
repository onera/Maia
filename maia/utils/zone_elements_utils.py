
import Converter.Internal as I
import maia.sids.Internal_ext as IE
import numpy              as np

from maia.sids import sids

# -----------------------------------------------------------------------------
type_to_npe = { 3 :  2, #Bar2
                5 :  3, #Tri3
                7 :  4, #Quad4
              # 9 :  8, #Quad8
               10 :  4, #Tetra4
               12 :  5, #Pyra5
               14 :  6, #Penta6
               17 :  8, #Hexa8
              #19 : 20, #Hexa20
              }
type_to_pdm = { 3 :  1, #Bar2
                5 :  2, #Tri3
                7 :  3, #Quad4
              # 9 :  9, #Quad8
               10 :  5, #Tetra4
               12 :  6, #Pyra5
               14 :  7, #Penta6
               17 :  8, #Hexa8
              #19 : 10, #Hexa20
              }

# -----------------------------------------------------------------------------
def get_npe_with_element_type_cgns(elmt_type):
  """
  """
  try:
    return type_to_npe[elmt_type]
  except KeyError:
    raise NotImplementedError('Not implemented elements type')

def get_paradigm_type_with_element_type_cgns(elmt_type):
  """
  """
  try:
    return type_to_pdm[elmt_type]
  except KeyError:
    raise NotImplementedError('Not implemented elements type')


def get_ordered_elements_std(zone_tree):
  """
  Return the elements nodes in inscreasing order wrt ElementRange
  """
  return sorted(I.getNodesFromType1(zone_tree, 'Elements_t'), 
                key = lambda item : sids.ElementRange(item)[0])

# --------------------------------------------------------------------------
def collect_pdm_type_and_nelemts(elmts):
  """
  """
  to_pdm_type  = lambda e : get_paradigm_type_with_element_type_cgns(sids.ElementType(e))
  to_elmt_size = lambda e : IE.getDistribution(e, 'Element')[1] - IE.getDistribution(e, 'Element')[0]

  return np.array([to_pdm_type(e) for e in elmts],  dtype=np.int32),\
         np.array([to_elmt_size(e) for e in elmts], dtype=np.int32)

def collect_connectity(elmts):
  """
  """
  return [I.getNodeFromName1(elmt, "ElementConnectivity")[1] for elmt in elmts]

def get_next_elements_range(zone):
  """
  Return the maximum element id found in the zone (?)
  """
  return max([sids.ElementRange(elem)[1] for elem in I.getNodesFromType1(zone, 'Elements_t')])

def get_range_of_ngon(zone):
  """
  Return the ElementRange array of the NGON elements
  """
  ngons = [elem for elem in I.getNodesFromType1(zone, 'Elements_t') if sids.ElementType(elem) == 22]
  assert len(ngons) == 1
  return sids.ElementRange(ngons[0])
