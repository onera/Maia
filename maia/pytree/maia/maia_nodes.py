from maia.pytree import walk
from maia.pytree import create_nodes as CN

def getDistribution(node, distri_name=None):
  """
  Starting from node, return the CGNS#Distribution node if distri_name is None
  or the value of the requested distribution if distri_name is not None
  """
  return walk.get_node_from_path(node, '/'.join([':CGNS#Distribution', distri_name])) if distri_name \
      else walk.get_child_from_name(node, ':CGNS#Distribution')

def getGlobalNumbering(node, lngn_name=None):
  """
  Starting from node, return the CGNS#GlobalNumbering node if lngn_name is None
  or the value of the requested globalnumbering if lngn_name is not None
  """
  return walk.get_node_from_path(node, '/'.join([':CGNS#GlobalNumbering', lngn_name])) if lngn_name \
      else walk.get_child_from_name(node, ':CGNS#GlobalNumbering')

# --------------------------------------------------------------------------
def newDistribution(distributions = dict(), parent=None):
  """
  Create and return a CGNSNode to be used to store distribution data
  Attach it to parent node if not None
  In addition, add distribution arrays specified in distributions dictionnary.
  distributions must be a dictionnary {DistriName : distri_array}
  """
  if parent:
    distri_node = CN.update_child(parent, ':CGNS#Distribution', 'UserDefinedData_t')
  else:
    distri_node = CN.new_node(':CGNS#Distribution', 'UserDefinedData_t')
  for name, value in distributions.items():
    CN.update_child(distri_node, name, 'DataArray_t', value)
  return distri_node

def newGlobalNumbering(glob_numberings = dict(), parent=None):
  """
  Create and return a CGNSNode to be used to store distribution data
  Attach it to parent node if not None
  In addition, add global numbering arrays specified in glob_numberings dictionnary.
  glob_numberings must be a dictionnary {NumberingName : lngn_array}
  """
  if parent:
    lngn_node = CN.update_child(parent, ':CGNS#GlobalNumbering', 'UserDefinedData_t')
  else:
    lngn_node = CN.new_node(':CGNS#GlobalNumbering', 'UserDefinedData_t')
  for name, value in glob_numberings.items():
    CN.update_child(lngn_node, name, 'DataArray_t', value)
  return lngn_node

# --------------------------------------------------------------------------

get_distribution                      = getDistribution
get_global_numbering                  = getGlobalNumbering
new_distribution                      = newDistribution
new_global_numbering                  = newGlobalNumbering
