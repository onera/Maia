import maia.pytree as PT
from   maia.pytree.typing import *

from maia.pytree.meta import CGNSNodeNotFoundError

from .conventions import DISTRI_NAME, GLBNUM_NAME

__all__ = ['get_Distribution', 'get_GlobalNumbering', 'find_Distribution', 'find_GlobalNumbering']

def get_Distribution(root:CGNSTree, name:Optional[str]=None) -> Optional[CGNSTree]:
  """ Get a distribution node under the specified root

  If ``name`` is not None, the DataArray_t node of corresponding name is returned.
  Otherwise, the distribution container is returned itself.

  Args:
    root (CGNSTree) : Root in which search is performed (distributed)
    name (str, optional): Name a specific array to get
  Example:
    >>> zone = PT.new_Zone(type='Unstructured', size=[[77,60,0]])
    >>> MT.new_Distribution({'Cell' : [0,15,60], 'Vertex' : [0,20,77]}, parent=zone)
    >>> PT.print_tree(MT.get_Distribution(zone))
    :CGNS#Distribution UserDefinedData_t 
    ├───Cell DataArray_t I4 [ 0 15 60]
    └───Vertex DataArray_t I4 [ 0 20 77]
    >>> PT.print_tree(MT.get_Distribution(zone, 'Vertex'))
    Vertex DataArray_t I4 [ 0 20 77]
  """
  path = f'{DISTRI_NAME}/{name}' if name else DISTRI_NAME
  return PT.get_node_from_path(root, path)

def find_Distribution(root:CGNSTree, name:Optional[str]=None) -> CGNSTree:
  if (node := get_Distribution(root, name)) is not None:
    return node
  raise CGNSNodeNotFoundError(root, DISTRI_NAME)


def get_GlobalNumbering(root:CGNSTree, name:Optional[str]=None) -> Optional[CGNSTree]:
  """ Get a global numbering node under the specified root

  If ``name`` is not None, the DataArray_t node of corresponding name is returned.
  Otherwise, the global numbering container is returned itself.

  Args:
    root (CGNSTree) : Root in which search is performed (partitioned)
    name (str, optional): Name a specific array to get
  Example:
    >>> zsr = PT.new_ZoneSubRegion(point_list=[[4,6,2,8]])
    >>> MT.new_GlobalNumbering({'Index' : [9,11,13,14]}, parent=zsr)
    >>> PT.print_tree(MT.get_GlobalNumbering(zsr))
    :CGNS#GlobalNumbering UserDefinedData_t 
    └───Index DataArray_t I4 [ 9 11 13 14]
    >>> MT.get_GlobalNumbering(zsr, 'Index')
    ['Index', array([ 9, 11, 13, 14], dtype=int32), [], 'DataArray_t']
  """
  path = f'{GLBNUM_NAME}/{name}' if name else GLBNUM_NAME
  return PT.get_node_from_path(root, path)

def find_GlobalNumbering(root:CGNSTree, name:Optional[str]=None) -> CGNSTree:
  if (node := get_GlobalNumbering(root, name)) is not None:
    return node
  raise CGNSNodeNotFoundError(root, GLBNUM_NAME)

