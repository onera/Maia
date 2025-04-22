import numpy as np

from maia.pytree.typing import *
from maia.pytree.meta   import begin_api_export, end_api_export, for_all_methods, check_is_label

from maia.pytree import walk as W
from maia.pytree import node as N
from maia.pytree import sids as S
from maia.utils import vstride as vs

begin_api_export()

def getDistribution(node:CGNSTree, distri_name:Optional[str]=None) -> Optional[CGNSTree]:
  """
  Starting from node, return the CGNS#Distribution node if distri_name is None
  or the value of the requested distribution if distri_name is not None
  """
  return W.get_node_from_path(node, '/'.join([':CGNS#Distribution', distri_name])) if distri_name \
      else W.get_child_from_name(node, ':CGNS#Distribution')

def getGlobalNumbering(node:CGNSTree, lngn_name:Optional[str]=None) -> Optional[CGNSTree]:
  """
  Starting from node, return the CGNS#GlobalNumbering node if lngn_name is None
  or the value of the requested globalnumbering if lngn_name is not None
  """
  return W.get_node_from_path(node, '/'.join([':CGNS#GlobalNumbering', lngn_name])) if lngn_name \
      else W.get_child_from_name(node, ':CGNS#GlobalNumbering')
      
def newDistribution(distributions:Dict[str, NDArray] = dict(), parent:Optional[CGNSTree]=None) -> CGNSTree:
  """
  Create and return a CGNSNode to be used to store distribution data
  Attach it to parent node if not None
  In addition, add distribution arrays specified in distributions dictionnary.
  distributions must be a dictionnary {DistriName : distri_array}
  """
  if parent:
    distri_node = N.update_child(parent, ':CGNS#Distribution', 'UserDefinedData_t')
  else:
    distri_node = N.new_node(':CGNS#Distribution', 'UserDefinedData_t')
  for name, value in distributions.items():
    N.update_child(distri_node, name, 'DataArray_t', value)
  return distri_node

def newGlobalNumbering(glob_numberings:Dict[str, NDArray] = dict(), parent:Optional[CGNSTree]=None) -> CGNSTree:
  """
  Create and return a CGNSNode to be used to store distribution data
  Attach it to parent node if not None
  In addition, add global numbering arrays specified in glob_numberings dictionnary.
  glob_numberings must be a dictionnary {NumberingName : lngn_array}
  """
  if parent:
    lngn_node = N.update_child(parent, ':CGNS#GlobalNumbering', 'UserDefinedData_t')
  else:
    lngn_node = N.new_node(':CGNS#GlobalNumbering', 'UserDefinedData_t')
  for name, value in glob_numberings.items():
    N.update_child(lngn_node, name, 'DataArray_t', value)
  return lngn_node

# --------------------------------------------------------------------------

get_distribution                      = getDistribution
get_global_numbering                  = getGlobalNumbering
new_distribution                      = newDistribution
new_global_numbering                  = newGlobalNumbering

class Zone:

  @staticmethod
  def EdgeNode(zone_node:CGNSTree) -> CGNSTree:
    is_edge = lambda n : N.get_label(n) == 'Elements_t' and S.Element.CGNSName(n) == 'BAR_2'
    edge_elts_nodes = W.get_children_from_predicate(zone_node, is_edge)
    assert len(edge_elts_nodes) == 1, "Exactly one EdgeElements_t node must be defined"
    return edge_elts_nodes[0]

@for_all_methods(check_is_label("Elements_t"))
class Element:

    @staticmethod
    def connectivity(elt_node:CGNSTree) -> vs.VStrideArray:  
      eso = W.get_child_from_name(elt_node, 'ElementStartOffset')
      ec  = W.request_child_from_name(elt_node, 'ElementConnectivity')
      assert ec[1] is not None

      is_distri = W.get_child_from_name(elt_node, ':CGNS#Distribution') is not None

      if eso is not None:
        assert eso is not None and eso[1] is not None
        eso_val = eso[1] - eso[1][0] if is_distri else eso[1]
        return vs.from_displs(eso_val, ec[1])
      else:
        assert S.Element.CGNSName(elt_node) not in ['NGON_n', 'NFACE_n', 'MIXED']
        counts = S.Element.NVtx(elt_node)
        return vs.from_counts(ec[1].dtype.type(counts), ec[1])
      

end_api_export()
