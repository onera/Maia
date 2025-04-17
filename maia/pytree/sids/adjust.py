import warnings
import numpy as np

from maia.pytree.typing import *
import maia.pytree as PT

IS_RELATED_ZSR = lambda n : PT.get_label(n) == 'ZoneSubRegion_t' \
                        and PT.get_child_from_name(n, 'BCRegionName') is not None

def enforceDonorAsPath(tree:CGNSTree):
  """ Force the GCs to indicate their opposite zone under the form BaseName/ZoneName """
  predicates = ['Zone_t', 'ZoneGridConnectivity_t', lambda n: PT.get_label(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t']]
  for base in PT.iter_all_CGNSBase_t(tree):
    base_n = PT.get_name(base)
    for gc in PT.iter_children_from_predicates(base, predicates):
      PT.set_value(gc, PT.GridConnectivity.ZoneDonorPath(gc, base_n))


def subregion_fields_to_bcdataset(tree:CGNSTree, mode:str='move'):
  """ Move the data fields from ZoneSubRegion nodes to their related BC node, if existing.
  
  ZoneSubRegion nodes must be explicitly related to a BC node through the BCRegionName descriptor.
  Data fields will be added in a BCDataSet named as the ZoneSubRegion (under a DirichletData node).
  This function is particularly useful for viewing BC data using Paraview.

  The operation performed depends of the value of ``mode`` argument:

  - if ``mode == 'move'``, fields are removed from the ZSR node (which is itself preserved);
  - if ``mode == 'copy'``, fields remains in the ZSR node and a copy is placed in the BCDataSet;
  - if ``mode == 'view'``, fields in the ZSR and those added in BCDataSet share the same memory.

  Args:
    tree (CGNSTree): Input tree, starting at Zone_t level or higher
    mode (str, optional): Controls how the fields are created (see above). Defaults to ``'move'``.

  Example:
    >>> zone = PT.yaml.to_zone('''
    ... Zone Zone_t:
    ...   ZoneBC ZoneBC_t:
    ...     Wing BC_t 'BCWall':
    ...   WingExtraction ZoneSubRegion_t:
    ...     field DataArray_t [10,20,30,40]:
    ...     BCRegionName Descriptor_t "Wing":
    ... ''')
    >>> PT.subregion_fields_to_bcdataset(zone)
    >>> PT.print_tree(zone)
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
      bc_n = PT.request_node_from_path(zone, PT.Subset.ZSRExtent(zsr_n, zone))
      
      bcdataset_n = PT.update_child(bc_n, f'{zsr_name}', 'BCDataSet_t', 'UserDefined')
      bcdata_n = PT.update_child(bcdataset_n, 'DirichletData', 'BCData_t')

      for fld_n in PT.get_children_from_predicate(zsr_n, "DataArray_t"):
        PT.rm_children_from_name(bcdata_n, PT.get_name(fld_n))
        PT.add_child(bcdata_n, copy_or_view(fld_n))
      if mode == 'move':
        PT.rm_children_from_label(zsr_n, 'DataArray_t')

def subregion_fields_from_bcdataset(tree:CGNSTree, mode:str='move'):
  """ Move the data fields to ZoneSubRegion nodes from their related BC node, if existing.
  
  ZoneSubRegion nodes must be explicitly related to a BC node through the BCRegionName descriptor.
  Data fields are taken from a *full* (without Subset) BCDataSet node, preferentially 
  named as the ZoneSubRegion node (see :func:`subregion_fields_to_bcdataset`);
  if such a node does not exists, the last *full* BCDataSet is taken.

  The operation performed depends of the value of ``mode`` argument:

  - if ``mode == 'move'``, fields are removed from the BCDataSet node (which is itself preserved);
  - if ``mode == 'copy'``, fields remains in the BCDataSet node and a copy is placed in the ZSR;
  - if ``mode == 'view'``, fields in the BCDataSet and those added in ZSR share the same memory.

  Args:
    tree (CGNSTree): Input tree, starting at Zone_t level or higher
    mode (str, optional): Controls how the fields are created (see above). Defaults to ``'move'``.

  Example:
    >>> zone = PT.yaml.to_node('''
    ... Zone Zone_t:
    ...   ZoneBC ZoneBC_t:
    ...     Wing BC_t 'BCWall':
    ...       WingExtraction BCDataSet_t "UserDefined":
    ...         DirichletData BCData_t:
    ...           field DataArray_t I4 [10, 20, 30, 40]:
    ...   WingExtraction ZoneSubRegion_t:
    ...     BCRegionName Descriptor_t "Wing":
    ... ''')
    >>> PT.subregion_fields_from_bcdataset(zone)
    >>> PT.print_tree(zone, 'Zone_t')
    Zone Zone_t 
    ├───ZoneBC ZoneBC_t 
    │   └───Wing BC_t "BCWall"
    │       └───WingExtraction BCDataSet_t "UserDefined"
    └───WingExtraction ZoneSubRegion_t 
        ├───BCRegionName Descriptor_t "Wing"
        └───field DataArray_t I4 [10 20 30 40]
  """
  if mode not in ['move', 'copy', 'view']:
    raise ValueError(f"Unvalid value for argument mode : {mode}")

  copy_or_view = (lambda n : PT.deep_copy(n)) if mode == 'copy' else (lambda n : n)

  for zone in PT.iter_all_Zone_t(tree):
    for zsr_n in PT.get_children_from_predicate(zone, IS_RELATED_ZSR):

      zsr_name = PT.get_name(zsr_n)
      bc_n = PT.request_node_from_path(zone, PT.Subset.ZSRExtent(zsr_n, zone))

      is_full_bcds = lambda n : PT.get_label(n) == 'BCDataSet_t' \
                            and PT.get_child_from_name(n, 'PointList') is None \
                            and PT.get_child_from_name(n, 'PointRange') is None
      bc_ds_list = PT.get_children_from_predicate(bc_n, is_full_bcds)
      if len(bc_ds_list) > 0:
        # Select relevant BCDS : take the one having ZSR name in priority. Otherwise, take last one
        for bc_ds in bc_ds_list:
          if PT.get_name(bc_ds) == zsr_name:
            break

        for fld_n in PT.get_children_from_predicates(bc_ds, "BCData_t/DataArray_t"):
          PT.rm_children_from_name(zsr_n, PT.get_name(fld_n))
          if PT.get_value(fld_n).size == 1 and (bc_size:=PT.Subset.n_elem(bc_n)) != 1: # Auto extend scalar data
            if mode == 'view':
              msg = f"On ZoneSubRegion '{zsr_name}', can not create a view of scalar data '{fld_n[0]}'" \
                    f" from BCDataSet '{bc_ds[0]}', a copy is done instead"
              warnings.warn(msg, stacklevel=2)
            fld_n = PT.shallow_copy(fld_n)
            fld_val = PT.get_value(fld_n)
            PT.set_value(fld_n, fld_val*np.ones(bc_size, fld_val.dtype))

          PT.add_child(zsr_n, copy_or_view(fld_n))
        if mode == 'move':
          PT.rm_children_from_label(bc_ds, 'BCData_t')
