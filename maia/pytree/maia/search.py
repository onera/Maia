import maia.pytree as PT
from   maia.typing import *

from maia.pytree.meta import CGNSNodeNotFoundError

from .conventions import DISTRI_NAME, GLBNUM_NAME, get_part_prefix

__all__ = ['get_Distribution', 'get_GlobalNumbering', 'find_Distribution', 'find_GlobalNumbering',
           'get_partitioned_zones']

def get_Distribution(root:CGNSTree, name:Optional[str]=None) -> Optional[CGNSTree]:
  """ Get a distribution node under the specified root

  If ``name`` is not None, the DataArray_t node of corresponding name is returned.
  Otherwise, the distribution container is returned itself.

  Args:
    root (CGNSTree) : Root in which search is performed (distributed)
    name (str, optional): Name a specific array to get
  Returns:
    CGNSTree or None: Node found
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
  if PT.get_child_from_name(root, DISTRI_NAME) is not None:
    raise CGNSNodeNotFoundError(root, f'{DISTRI_NAME}/{name}')
  else:
    raise CGNSNodeNotFoundError(root, DISTRI_NAME)


def get_GlobalNumbering(root:CGNSTree, name:Optional[str]=None) -> Optional[CGNSTree]:
  """ Get a global numbering node under the specified root

  If ``name`` is not None, the DataArray_t node of corresponding name is returned.
  Otherwise, the global numbering container is returned itself.

  Args:
    root (CGNSTree) : Root in which search is performed (partitioned)
    name (str, optional): Name a specific array to get
  Returns:
    CGNSTree or None: Node found
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
  if PT.get_child_from_name(root, GLBNUM_NAME) is not None:
    raise CGNSNodeNotFoundError(root, f'{GLBNUM_NAME}/{name}')
  else:
    raise CGNSNodeNotFoundError(root, GLBNUM_NAME)

def get_partitioned_zones(part_tree: CGNSPartTree, zone_path: CGNSPath) -> List[CGNSPartTree]:
  """ Return the list of partitioned zones created from the specified initial domain.

  Search can be performed from tree level, in which case ``zone_path`` has the pattern
  ``'BaseName/ZoneName'``,
  or from base level, in which case ``zone_path`` has the pattern ``'ZoneName'``.

  Args:
    part_tree (CGNSPartTree) : Partitioned tree in which search is performed
    zone_path (str): searched pattern (see above)
  Returns:
    list of CGNSTree: Partitioned zones found
  Example:
    >>> part_tree = PT.yaml.to_cgns_tree('''
    ... Base CGNSBase_t:
    ...   Zone1.P0.N1 Zone_t:
    ...   Zone1.P0.N2 Zone_t:
    ...   Zone2.P0.N0 Zone_t:
    ...   Zone2.P0.N1 Zone_t:
    ...   Zone3.P0.N0 Zone_t:
    ... ''')
    >>> pzones = MT.get_partitioned_zones(part_tree, 'Base/Zone2')
    >>> [PT.get_name(z) for z in pzones]
    ['Zone2.P0.N0', 'Zone2.P0.N1']
  """
  base_name, zone_name = PT.utils.path_head(zone_path), PT.utils.path_tail(zone_path)
  part_base = PT.get_node_from_path(part_tree, base_name)
  if part_base:
    return [part for part in PT.iter_all_Zone_t(CGNSPartTree(part_base)) if \
        get_part_prefix(PT.get_name(part)) == zone_name]
  else:
    return []