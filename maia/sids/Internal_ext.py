import Converter.Internal as I

def getVal(t):
  return t[1]

def newDistribution(distri_list = dict(), parent=None):
  """
  Create and return a CGNSNode to be used to store distribution data
  Attach it to parent node if not None
  In addition, add distribution arrays specified in distri_list dictionnary.
  distri_list must be a dictionnary {DistriName : distri_array}
  """
  distri_node = I.newUserDefinedData(':CGNS#Distribution', None, parent)
  for name, value in distri_list.items():
    I.newDataArray(name, value, parent=distri_node)
  return distri_node

def newGlobalNumbering(lngn_list = dict(), parent=None):
  """
  Create and return a CGNSNode to be used to store distribution data
  Attach it to parent node if not None
  In addition, add global numbering arrays specified in lngn_list dictionnary.
  lngn_list must be a dictionnary {NumberingName : lngn_array}
  """
  lngn_node = I.newUserDefinedData(':CGNS#GlobalNumbering', None, parent)
  for name, value in lngn_list.items():
    I.newDataArray(name, value, parent=lngn_node)
  return lngn_node
