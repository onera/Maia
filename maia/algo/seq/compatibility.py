import numpy as np

import maia
from maia.typing import *
import maia.pytree as PT
import maia.pytree.maia as MT
from maia.utils import np_utils

def indexed_to_interleaved_connectivity(node: CGNSTree) -> None:
  offset = PT.find_child_from_name(node, 'ElementStartOffset')
  connec = PT.find_child_from_name(node, 'ElementConnectivity')

  new_val = np_utils.indexed_to_interlaced(PT.get_np_value(offset),
                                           PT.get_np_value(connec))

  PT.set_value(connec, new_val)
  PT.rm_child(node, offset)

def interlaced_to_indexed_connectivity(node: CGNSTree) -> None:
  n_elem = PT.Element.Size(node)
  connec = PT.find_child_from_name(node, 'ElementConnectivity')
  idx, array = np_utils.interlaced_to_indexed(int(n_elem), PT.get_np_value(connec))

  PT.new_DataArray('ElementStartOffset', value=idx, parent=node)
  PT.set_value(connec, array)

def create_mixed_elts_eso(node: CGNSTree) -> None:
  from cmaia.utils import layouts
  ec_n = PT.find_node_from_name(node, 'ElementConnectivity')
  ec = PT.get_np_value(ec_n)
  eso = np.empty(PT.Element.Size(node)+1, ec.dtype)
  layouts.create_mixed_elts_eso(ec, eso)
  PT.new_DataArray('ElementStartOffset', eso, parent=node)

def enforce_ngon_pe_local(full_tree: CGNSTree) -> None:
  """
  Shift the ParentElements values in order to make it start at 1, as requested by legacy tools.

  The tree is modified in place.

  Args:
    full_tree (CGNSTree): Tree starting at Zone_t level or higher.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #enforce_ngon_pe_local@start
        :end-before: #enforce_ngon_pe_local@end
        :dedent: 2

  """
  MT.check_cgns_full_tree(full_tree)
  for zone in PT.iter_all_Zone_t(full_tree):
    try:
      ngon_node = PT.Zone.NGonNode(zone)
    except RuntimeError: #If no NGon, go to next zone
      continue
    pe = PT.find_child_from_name(ngon_node, 'ParentElements')
    PT.set_value(pe, maia.algo.indexing.get_pe_local(ngon_node))

def poly_new_to_old(full_tree: CGNSTree, full_onera_compatibility: bool = True) -> None:
  """
  Transform a tree with polyhedral unstructured connectivity with new CGNS 4.x conventions to old CGNS 3.x conventions.

  The tree is modified in place.

  Args:
    full_tree (CGNSTree): Tree described with new CGNS convention.
    full_onera_compatibility (bool): if ``True``, shift NFace and ParentElements ids to begin at 1, irrespective of the NGon and NFace ElementRanges, and make the NFace connectivity unsigned

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #poly_new_to_old@start
        :end-before: #poly_new_to_old@end
        :dedent: 2
  """
  MT.check_cgns_full_tree(full_tree)
  cg_version_node = PT.find_child_from_label(full_tree, 'CGNSLibraryVersion_t')
  PT.set_value(cg_version_node, 3.1)
  for z in PT.get_all_Zone_t(full_tree):
    if PT.Zone.Type(z) != 'Unstructured':
      continue
    elif PT.Zone.has_ngon_elements(z):

      has_nface = PT.Zone.has_nface_elements(z)

      ngon  = maia.pytree.Zone.NGonNode (z)
      ngon_range   = PT.Element.Range(ngon)
      if has_nface:
        nface = maia.pytree.Zone.NFaceNode(z)
        nface_range  = PT.Element.Range(nface)
        nface_connec = PT.get_np_value(PT.find_child_from_name(nface, "ElementConnectivity"))

      if full_onera_compatibility:
        # 1. shift ParentElements to 1
        pe_node = PT.get_child_from_name(ngon,"ParentElements")
        if pe_node:
          # pe = PT.get_value(pe_node)
          # pe += (-nface_range[0]+1)*(pe>0)
          PT.set_value(pe_node, maia.algo.indexing.get_pe_local(ngon))

        if has_nface:
          # 2. do not use a signed NFace connectivity
          np.absolute(nface_connec,out=nface_connec)

          # 3. shift NFace connectivity to 1
          nface_connec += -ngon_range[0]+1

      # 4. indexed to interleaved
      indexed_to_interleaved_connectivity(ngon)
      if has_nface:
        indexed_to_interleaved_connectivity(nface)

    else: # No NGon / NFace, but we may have to deal with MIXED elements
      for elt in PT.iter_children_from_predicate(z, PT.pred.is_element_of_type('MIXED')):
        PT.rm_children_from_name(elt, 'ElementStartOffset')



def poly_old_to_new(full_tree: CGNSTree) -> None:
  """
  Transform a tree with polyhedral unstructured connectivity with old CGNS 3.x conventions to new CGNS 4.x conventions.

  The tree is modified in place.

  This function accepts trees with old ONERA conventions where NFace and ParentElements ids begin at 1, irrespective of the NGon and NFace ElementRanges, and where the NFace connectivity is unsigned. The resulting tree has the correct CGNS/SIDS conventions.

  Args:
    full_tree (CGNSTree): Tree described with old CGNS convention.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #poly_old_to_new@start
        :end-before: #poly_old_to_new@end
        :dedent: 2
  """
  MT.check_cgns_full_tree(full_tree)
  cg_version_node = PT.find_child_from_label(full_tree, 'CGNSLibraryVersion_t')
  PT.set_value(cg_version_node, 4.2)
  for z in PT.get_all_Zone_t(full_tree):
    if PT.Zone.Type(z) != 'Unstructured':
      continue
    elif PT.Zone.has_ngon_elements(z):
      has_nface = PT.Zone.has_nface_elements(z)
      ngon  = maia.pytree.Zone.NGonNode (z)
      ngon_range   = PT.Element.Range(ngon)
      if has_nface:
        nface = maia.pytree.Zone.NFaceNode(z)
        nface_range  = PT.Element.Range(nface)

      # 1. interleaved to indexed
      interlaced_to_indexed_connectivity(ngon)

      # 2. shift ParentElements if necessary
      pe_node = PT.get_child_from_name(ngon, "ParentElements")
      if pe_node:
        if not has_nface: #Induce NFace range for PE reconstruction
          nface_range  = np.array([ngon_range[1]+1, ngon_range[1]+PT.Zone.n_cell(z)])
        pe = PT.get_np_value(pe_node)
        pe_no_0 = pe[pe>0]
        min_pe = np.min(pe_no_0)
        max_pe = np.max(pe_no_0)
        if not (min_pe==nface_range[0] and max_pe==nface_range[1]):
          if min_pe!=1:
            raise RuntimeError("ParentElements values are not SIDS-compliant, and they do not start at 1")
          else:
            pe += (+nface_range[0]-1)*(pe>0)

      # 3. NFace
      if has_nface:
        nface_connec = PT.get_np_value(PT.find_child_from_name(nface, "ElementConnectivity"))
        n_cell = nface_range[1] - nface_range[0]
        if np.min(nface_connec)<0 or n_cell==1: # NFace is signed (if only one cell, it is signed despite being positive)
          # 3.1. interleaved to indexed
          interlaced_to_indexed_connectivity(nface)
          nface_connec = PT.get_np_value(PT.find_child_from_name(nface, "ElementConnectivity"))
    
          # 3.2. shift
          sign_nf = np.sign(nface_connec)
          abs_nf = np.absolute(nface_connec)
          min_nf = np.min(abs_nf)
          max_nf = np.max(abs_nf)
          if not (min_nf==ngon_range[0] and max_nf==ngon_range[1]):
            if min_nf!=1:
              raise RuntimeError("NFace ElementConnectivity values are not SIDS-compliant, and they do not start at 1")
            else:
              abs_nf += +ngon_range[0]-1
              nface_connec[:] = abs_nf * sign_nf
        else: # NFace is not signed: need to recompute it
          PT.rm_child(z,nface)
          if not pe_node:
            raise RuntimeError("NFace is not signed: this is not compliant. However, a ParentElements is needed to recompute a correct NFace")
          if ngon_range[0] != 1:
            raise NotImplementedError("NFace is not signed: this is not compliant. It needs to be recomputed, but not implemented in case NGon is not first")
          maia.algo.pe_to_nface(z)

    else: # No NGon / NFace, but we may have to deal with MIXED elements
      for elt in PT.iter_children_from_predicate(z, PT.pred.is_element_of_type('MIXED')):
        create_mixed_elts_eso(elt)
