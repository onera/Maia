import Converter.Internal as I

def getVal(t):
  return t[1]

def newDistribution(distributions = dict(), parent=None):
  """
  Create and return a CGNSNode to be used to store distribution data
  Attach it to parent node if not None
  In addition, add distribution arrays specified in distributions dictionnary.
  distributions must be a dictionnary {DistriName : distri_array}
  """
  distri_node = I.newUserDefinedData(':CGNS#Distribution', None, parent)
  for name, value in distributions.items():
    I.newDataArray(name, value, parent=distri_node)
  return distri_node

def newGlobalNumbering(glob_numberings = dict(), parent=None):
  """
  Create and return a CGNSNode to be used to store distribution data
  Attach it to parent node if not None
  In addition, add global numbering arrays specified in glob_numberings dictionnary.
  glob_numberings must be a dictionnary {NumberingName : lngn_array}
  """
  lngn_node = I.newUserDefinedData(':CGNS#GlobalNumbering', None, parent)
  for name, value in glob_numberings.items():
    I.newDataArray(name, value, parent=lngn_node)
  return lngn_node

def getDistribution(node, distri_name=None):
  """
  Starting from node, return the CGNS#Distribution node if distri_name is None
  or the value of the requested distribution if distri_name is not None
  """
  return getVal(I.getNodeFromPath(node, ':CGNS#Distribution/' + distri_name)) if distri_name \
      else I.getNodeFromName1(node, ':CGNS#Distribution')

def getGlobalNumbering(node, lngn_name=None):
  """
  Starting from node, return the CGNS#GlobalNumbering node if lngn_name is None
  or the value of the requested globalnumbering if lngn_name is not None
  """
  return getVal(I.getNodeFromPath(node, ':CGNS#GlobalNumbering/' + lngn_name)) if lngn_name \
      else I.getNodeFromName1(node, ':CGNS#GlobalNumbering')
