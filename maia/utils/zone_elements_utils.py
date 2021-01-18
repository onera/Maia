
import Converter.Internal as I
import numpy              as NPY

from   Converter import cgnskeywords as CGK

# --------------------------------------------------------------------------
def get_npe_with_element_type_cgns(elmt_type):
  """
  """
  if(  elmt_type ==  3):    # Bar2
    npe   = 2
  elif(  elmt_type ==  5):  # Tri3
    npe   = 3
  elif(elmt_type ==  7):  # Quad4
    npe   = 4
  # elif(elmt_type ==  9):  # Quad8
  #   npe   = 8
  elif(elmt_type == 10):  # Tetra4
    npe   = 4
  elif(elmt_type == 12):  # Pyra5
    npe   = 5
  elif(elmt_type == 14):  # Penta6
    npe   = 6
  elif(elmt_type == 17):  # Hexa8
    npe   = 8
  # elif(elmt_type == 19):  # Hexa20
  #   npe   = 20
  else:
    raise NotImplementedError('Not implemented elements type')

  return npe

# --------------------------------------------------------------------------
def get_paradigm_type_with_element_type_cgns(elmt_type):
  """
  """
  if(  elmt_type ==  3):  # Bar2
    PDM_Type = 1
  elif(  elmt_type ==  5):  # Tri3
    PDM_Type = 2
  elif(elmt_type ==  7):  # Quad4
    PDM_Type = 3
  # elif(elmt_type ==  9):  # Quad8
  #   PDM_Type = 9
  elif(elmt_type == 10):  # Tetra4
    PDM_Type = 5
  elif(elmt_type == 12):  # Pyra5
    PDM_Type = 6
  elif(elmt_type == 14):  # Penta6
    PDM_Type = 7
  elif(elmt_type == 17):  # Hexa8
    PDM_Type = 8
  # elif(elmt_type == 19):  # Hexa20
  #   PDM_Type = 10
  else:
    raise NotImplementedError('Not implemented elements type')

  return PDM_Type

# --------------------------------------------------------------------------
# def get_ordered_elements_std(zone_tree):
#   """
#   Return in increasing order the elements
#   """
#   # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   elmts_ini = I.getNodesFromType1(zone_tree, 'Elements_t')
#   # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#   # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   elmts_list_3d = list()
#   elmts_list_2d = list()
#   elmts_list_1d = list()
#   # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#   # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   next_to_find = 1
#   while len(elmts_list_3d)+len(elmts_list_2d)+len(elmts_list_1d) != len(elmts_ini):
#     # print("next_to_find", next_to_find)
#     # ------------------------------------------------------------
#     for idx_elmt, elmt in enumerate(elmts_ini):
#       # print("next_to_find", next_to_find)
#       # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#       if(I.getNodeFromName1(elmt, 'ElementRange')[1][0] == next_to_find):

#         # > Verbose
#         # print("----", elmt[0], elmt[1], next_to_find)

#         # > Offset Current Element
#         next_to_find += ( I.getNodeFromName1(elmt, 'ElementRange')[1][1] - I.getNodeFromName1(elmt, 'ElementRange')[1][0] + 1 )

#         # > Verbose
#         # print "++++", next_to_find
#         if(elmt[1][0] in CGK.ElementType1D):
#           elmts_list_1d.append(elmt)
#         elif(elmt[1][0] in CGK.ElementType2D):
#           elmts_list_2d.append(elmt)
#         else:
#           assert(elmt[1][0] in CGK.ElementType3D)
#           elmts_list_3d.append(elmt)
#       # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     # ------------------------------------------------------------
#   # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#   return elmts_list_3d, elmts_list_2d, elmts_list_1d

# --------------------------------------------------------------------------
def get_ordered_elements_std(zone_tree):
  """
  Return in increasing order the elements
  """
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  elmts_ini = I.getNodesFromType1(zone_tree, 'Elements_t')
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  elmts_list = list()
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  next_to_find = 1
  while len(elmts_list) != len(elmts_ini):
    # print("next_to_find", next_to_find)
    # ------------------------------------------------------------
    for idx_elmt, elmt in enumerate(elmts_ini):
      # print("next_to_find", next_to_find)
      # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      if(I.getNodeFromName1(elmt, 'ElementRange')[1][0] == next_to_find):

        # > Verbose
        # print("----", elmt[0], elmt[1], next_to_find)

        # > Offset Current Element
        next_to_find += ( I.getNodeFromName1(elmt, 'ElementRange')[1][1] - I.getNodeFromName1(elmt, 'ElementRange')[1][0] + 1 )

        # > Verbose
        # print "++++", next_to_find
        elmts_list.append(elmt)
      # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ------------------------------------------------------------
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  return elmts_list

# --------------------------------------------------------------------------
def collect_pdm_type_and_nelemts(elmts):
  """
  """
  # ************************************************************************
  # > Declaration
  # ************************************************************************

  elmt_type   = NPY.zeros(len(elmts), dtype='int32', order='F')
  elmt_n_elmt = NPY.zeros(len(elmts), dtype='int32', order='F')

  for i_elmt, elmt in enumerate(elmts):
    distrib_ud   = I.getNodeFromName1(elmt      , ':CGNS#Distribution')
    distrib_elmt = I.getNodeFromName1(distrib_ud, 'Distribution')[1]
    dn_elmt      = distrib_elmt[1] - distrib_elmt[0]

    elmt_type  [i_elmt] = get_paradigm_type_with_element_type_cgns(elmt[1][0])
    elmt_n_elmt[i_elmt] = dn_elmt

  return elmt_type, elmt_n_elmt

# --------------------------------------------------------------------------
def collect_connectity(elmts):
  """
  """
  elmts_connectivity = list()
  for elmt in elmts:
    connectivity = I.getNodeFromName1(elmt, "ElementConnectivity")[1]
    elmts_connectivity.append(connectivity)

  return elmts_connectivity


def get_next_elements_range(zone):
  """
  """
  elmts = I.getNodesFromType2(zone, 'Elements_t')
  emax = - 1000000
  for e in elmts:
     ERElements = I.getNodeFromName1(e, 'ElementRange')[1]
     emax = max(emax, ERElements[1])
  return emax

def get_range_of_ngon(zone):
  """
  """
  elmts = I.getNodesFromType2(zone, 'Elements_t')
  emax = - 1000000
  found = False
  for elmt in elmts:
    if(elmt[1][0] == 22):
      erange = I.getNodeFromName1(elmt, 'ElementRange')[1]
      assert(found is False)
      found = True
      beg, end = erange[0], erange[1]
  return beg, end
