import maia.pytree as PT
from   maia.pytree.typing import *

__all__ = ['new_Distribution', 'new_GlobalNumbering']

from .conventions import DISTRI_NAME, GLBNUM_NAME

def new_Distribution(fields:Mapping[str, ArrayLike] = {},
                     parent:Optional[CGNSTree] = None) -> CGNSTree:
  """
  Create a :CGNS#Distribution node.

  This Maia specific node describes how data is distributed
  (see :ref:`specification <dist_tree>`).

  Args:
    fields (dict) : distribution values to create under the container (see :ref:`fields setting <pt_presets_commun>`)
    parent (CGNSTree): Node to which created distribution should be attached
  Example:
    >>> zone = PT.new_Zone(type='Unstructured', size=[[77,60,0]])
    >>> MT.new_Distribution({'Cell' : [0,15,60], 'Vertex' : [0,20,77]}, parent=zone)
    >>> PT.print_tree(zone)
    Zone Zone_t I4 [[77 60  0]]
    ├───ZoneType ZoneType_t "Unstructured"
    └───:CGNS#Distribution UserDefinedData_t 
        ├───Cell DataArray_t I4 [ 0 15 60]
        └───Vertex DataArray_t I4 [ 0 20 77]
  """
  if parent:
    distri_node = PT.update_child(parent, DISTRI_NAME, 'UserDefinedData_t')
  else:
    distri_node = PT.new_node(DISTRI_NAME, 'UserDefinedData_t')
  for name, value in fields.items():
    PT.update_child(distri_node, name, 'DataArray_t', value)
  return distri_node

def new_GlobalNumbering(fields:Mapping[str, ArrayLike] = {},
                        parent:Optional[CGNSTree] = None) -> CGNSTree:
  """
  Create a :CGNS#GlobalNumbering node.

  This Maia specific node describes how data is reordered after partitioning
  (see :ref:`specification <part_tree>`).

  Args:
    fields (dict) : gnum values to create under the container (see :ref:`fields setting <pt_presets_commun>`)
    parent (CGNSTree): Node to which created numberings should be attached
  Example:
    >>> gn = MT.new_GlobalNumbering({'Cell' : [6,4,7], 'Vertex' : [24,59,23,11,5]})
    >>> PT.print_tree(gn)
    :CGNS#GlobalNumbering UserDefinedData_t 
    ├───Cell DataArray_t I4 [6 4 7]
    └───Vertex DataArray_t I4 [24 59 23 11  5]
  """
  if parent:
    lngn_node = PT.update_child(parent, GLBNUM_NAME, 'UserDefinedData_t')
  else:
    lngn_node = PT.new_node(GLBNUM_NAME, 'UserDefinedData_t')
  for name, value in fields.items():
    PT.update_child(lngn_node, name, 'DataArray_t', value)
  return lngn_node
