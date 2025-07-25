import typing
import numpy as np

from maia.pytree.typing import *
from maia.typing import MPIComm
from maia.pytree.meta   import begin_api_export, end_api_export, for_all_methods, check_is_label, CGNSNodeNotFoundError

from maia.pytree import walk as W
from maia.pytree import node as N
from maia.pytree import sids as S
from maia.utils import vstride as vs
from maia.utils import par_utils

begin_api_export()

DISTRI_NAME = ':CGNS#Distribution'
GLBNUM_NAME = ':CGNS#GlobalNumbering'

def get_Distribution(root:CGNSTree, distri_name:Optional[str]=None) -> Optional[CGNSTree]:
  """
  Starting from node, return the CGNS#Distribution node if distri_name is None
  or the value of the requested distribution if distri_name is not None
  """
  path = f'{DISTRI_NAME}/{distri_name}' if distri_name else DISTRI_NAME
  return W.get_node_from_path(root, path)

def find_Distribution(root:CGNSTree, distri_name:Optional[str]=None) -> CGNSTree:
  if (node := get_Distribution(root, distri_name)) is not None:
    return node
  raise CGNSNodeNotFoundError(root, DISTRI_NAME)

def distribution_value(root:CGNSTree, distri_name:str) -> NDArray:
  return N.get_np_value(find_Distribution(root, distri_name))

def get_GlobalNumbering(root:CGNSTree, lngn_name:Optional[str]=None) -> Optional[CGNSTree]:
  """
  Starting from node, return the CGNS#GlobalNumbering node if lngn_name is None
  or the value of the requested globalnumbering if lngn_name is not None
  """
  path = f'{GLBNUM_NAME}/{lngn_name}' if lngn_name else GLBNUM_NAME
  return W.get_node_from_path(root, path)

def find_GlobalNumbering(root:CGNSTree, lngn_name:Optional[str]=None) -> CGNSTree:
  if (node := get_GlobalNumbering(root, lngn_name)) is not None:
    return node
  raise CGNSNodeNotFoundError(root, GLBNUM_NAME)

def globalnumbering_value(root:CGNSTree, lngn_name:str) -> NDArray:
  return N.get_np_value(find_GlobalNumbering(root, lngn_name))


def new_Distribution(distributions:Dict[str, NDArray] = dict(), parent:Optional[CGNSTree]=None) -> CGNSTree:
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

def new_GlobalNumbering(glob_numberings:Dict[str, NDArray] = dict(), parent:Optional[CGNSTree]=None) -> CGNSTree:
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
def is_single_node(X:Union[CGNSTree, List[CGNSTree]]) -> bool:
  if len(X) != 4:
    return False
  else:
    return isinstance(X[0], str)

def _n_entity(input:Union[CGNSTree, List[CGNSTree]], comm:Optional[MPIComm], name:str) -> int:
  if is_single_node(input):
    # Distributed implementation
    node = typing.cast(CGNSTree, input)
    distri = distribution_value(node, name)
    return int(distri[2])
  else:
    # Partitioned implementation
    nodes = typing.cast(List[CGNSTree], input)
    assert comm is not None
    gnum_l = [globalnumbering_value(n, name) for n in nodes]
    return int(par_utils.arrays_max(gnum_l, comm))

class Zone:

  @staticmethod
  def dn_cell(zone_node:CGNSTree) -> int:
    # Return the local number of cells (only for distributed zones)
    distri = distribution_value(zone_node, 'Cell')
    return int(distri[1] - distri[0])

  @staticmethod
  def pn_cell(zone_node:CGNSTree) -> int:
    # Return the local number of cells (only for partitioned zones)
    gnum = globalnumbering_value(zone_node, 'Cell')
    return gnum.size

  @staticmethod
  def n_cell(zone_node:Union[CGNSTree, List[CGNSTree]], comm:Optional[MPIComm]=None) -> int:
    # Return the total number of cells, for partitioned or distributed zone
    # For distributed meshes, a single node is expected,
    # For partitioned meshes, the list of "sister" zones is expected, and comm is mandatory
    return _n_entity(zone_node, comm, 'Cell')

  @staticmethod
  def dn_vtx(zone_node:CGNSTree) -> int:
    # Return the local number of vertices (only for distributed zones)
    distri = distribution_value(zone_node, 'Vertex')
    return int(distri[1] - distri[0])

  @staticmethod
  def pn_vtx(zone_node:CGNSTree) -> int:
    # Return the local number of vertices (only for partitioned zones)
    gnum = globalnumbering_value(zone_node, 'Vertex')
    return gnum.size


  @staticmethod
  def n_vtx(zone_node:Union[CGNSTree, List[CGNSTree]], comm:Optional[MPIComm]=None) -> int:
    # Return the total number of vertices, for partitioned or distributed zone
    # For distributed meshes, a single node is expected,
    # For partitioned meshes, the list of "sister" zones is expected, and comm is mandatory
    return _n_entity(zone_node, comm, 'Vertex')

  @staticmethod
  def EdgeNode(zone_node:CGNSTree) -> CGNSTree:
    is_edge = lambda n : N.get_label(n) == 'Elements_t' and S.Element.Type(n) == 'BAR_2'
    edge_elts_nodes = W.get_children_from_predicate(zone_node, is_edge)
    assert len(edge_elts_nodes) == 1, "Exactly one EdgeElements_t node must be defined"
    return edge_elts_nodes[0]

class Element:

  @staticmethod
  def dn_elt(elt_node:CGNSTree) -> int:
    # Return the local number of elements (only for distributed zones)
    distri = distribution_value(elt_node, 'Element')
    return int(distri[1] - distri[0])

  @staticmethod
  def pn_elt(elt_node:CGNSTree) -> int:
    # Return the local number of elements (only for partitioned zones)
    gnum = globalnumbering_value(elt_node, 'Element')
    return gnum.size

  @staticmethod
  def n_elt(elt_node:Union[CGNSTree, List[CGNSTree]], comm:Optional[MPIComm]=None) -> int:
    # Return the total number of elements for this section, for partitioned or distributed zone
    # For distributed meshes, a single node is expected,
    # For partitioned meshes, the list of "sister" zones is expected, and comm is mandatory
    return _n_entity(elt_node, comm, 'Element')

  @staticmethod
  def connectivity(elt_node:CGNSTree) -> vs.VStrideArray:  
    eso = W.get_child_from_name(elt_node, 'ElementStartOffset')
    ec  = W.find_child_from_name(elt_node, 'ElementConnectivity')
    assert ec[1] is not None

    is_distri = W.get_child_from_name(elt_node, ':CGNS#Distribution') is not None

    if eso is not None:
      assert eso is not None and eso[1] is not None
      eso_val = eso[1] - eso[1][0] if is_distri else eso[1]
      return vs.from_displs(eso_val, ec[1])
    else:
      assert S.Element.Type(elt_node) not in ['NGON_n', 'NFACE_n', 'MIXED']
      counts = S.Element.NVtx(elt_node)
      return vs.from_counts(ec[1].dtype.type(counts), ec[1])
      
class Subset:

  @staticmethod
  def dn_elem(subset_node:CGNSTree) -> int:
    # Return the local number of indices (only for distributed subsets)
    distri = distribution_value(subset_node, 'Index')
    return int(distri[1] - distri[0])

  @staticmethod
  def pn_elem(subset_node:CGNSTree) -> int:
    # Return the local number of indices (only for partitioned subsets)
    # Use PT.Subset to deal missing gnum arrays
    import maia.pytree as PT
    return PT.Subset.n_elem(subset_node)

  @staticmethod
  def n_elem(subset_node:Union[List[CGNSTree], CGNSTree], comm:Optional[MPIComm]=None) -> int:
    # Special case: for partitioned PointRange (S meshes), gnum array is
    # not always created -> we can not process
    if not is_single_node(subset_node):
      # Partitioned case
      assert comm is not None
      nodes = typing.cast(List[CGNSTree], subset_node)
      if not par_utils.exists_everywhere(nodes, f'{GLBNUM_NAME}/Index', comm):
        raise RuntimeError("GlobalNumbering nodes are mandatory to retrieve initial n_elem")

    # Fallback to standard case
    return _n_entity(subset_node, comm, 'Index')

end_api_export()
