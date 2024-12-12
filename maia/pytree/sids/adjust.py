from maia.pytree.typing import *
from maia.pytree.meta import api_export

import maia.pytree as PT

IS_RELATED_ZSR = lambda n : PT.get_label(n) == 'ZoneSubRegion_t' \
                        and PT.get_child_from_name(n, 'BCRegionName') is not None

@api_export
def enforceDonorAsPath(tree:CGNSTree):
  """ Force the GCs to indicate their opposite zone under the form BaseName/ZoneName """
  predicates = ['Zone_t', 'ZoneGridConnectivity_t', lambda n: PT.get_label(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t']]
  for base in PT.iter_all_CGNSBase_t(tree):
    base_n = PT.get_name(base)
    for gc in PT.iter_children_from_predicates(base, predicates):
      PT.set_value(gc, PT.GridConnectivity.ZoneDonorPath(gc, base_n))


@api_export
def subregion_fields_to_bcdataset(tree:CGNSTree, mode:str='move'):
  """ Move the data fields from ZoneSubRegion nodes to their related BC node, if existing.
  
  ZoneSubRegion nodes must be explicitly related to a BC node through the BCRegionName descriptor.
  Data fields will be added in a BCDataSet named as the ZoneSubRegion node.

  The operation performed depends of the value of ``mode`` argument:

  - if ``mode == 'move'``, fields are removed from the ZSR node;
  - if ``mode == 'copy'``, fields remains in the ZSR node and a copy is done in the BCDataSet;
  - if ``mode == 'view'``, fields in the ZSR and in the BCDataSet share the same memory.

  Args:
    tree (CGNSTree): Input tree (starting at root level)
    mode (str, optional): Controls how the BCDataSet fields are created (see above). Defaults to ``'move'``.

  Example:
    >>> tree = PT.yaml.to_cgns_tree('''
    ... Zone Zone_t:
    ...   ZoneBC ZoneBC_t:
    ...     Wing BC_t 'BCWall':
    ...   WingExtraction ZoneSubRegion_t:
    ...     field DataArray_t [10,20,30,40]:
    ...     BCRegionName Descriptor_t "Wing":
    ... ''')
    >>> PT.subregion_fields_to_bcdataset(tree)
    >>> PT.print_tree(PT.get_node_from_label(tree, 'Zone_t'))
    Zone Zone_t 
    ├───ZoneBC ZoneBC_t 
    │   └───Wing BC_t "BCWall"
    │       └───WingExtraction BCDataSet_t "UserDefined"
    │           └───DirichletData BCData_t 
    │               └───field DataArray_t I4 [10 20 30 40]
    └───WingExtraction ZoneSubRegion_t 
        └───BCRegionName Descriptor_t "Wing"
  """
  if mode not in ['move', 'copy', 'view']:
    raise ValueError(f"Unvalid value for argument mode : {mode}")

  copy_or_view = (lambda n : PT.deep_copy(n)) if mode == 'copy' else (lambda n : n)

  for zone in PT.iter_all_Zone_t(tree):
    for zsr_n in PT.get_children_from_predicate(zone, IS_RELATED_ZSR):

      zsr_name = PT.get_name(zsr_n)
      bc_name  = PT.get_value(PT.get_child_from_name(zsr_n, "BCRegionName"))

      bc_n = PT.get_child_from_predicates(zone, f"ZoneBC_t/{bc_name}")
      
      bcdataset_n = PT.update_child(bc_n, f'{zsr_name}', 'BCDataSet_t', 'UserDefined')
      bcdata_n = PT.update_child(bcdataset_n, 'DirichletData', 'BCData_t')

      for fld_n in PT.get_children_from_predicate(zsr_n, "DataArray_t"):
        PT.rm_children_from_name(bcdata_n, PT.get_name(fld_n))
        PT.add_child(bcdata_n, copy_or_view(fld_n))
      if mode == 'move':
        PT.rm_children_from_label(zsr_n, 'DataArray_t')