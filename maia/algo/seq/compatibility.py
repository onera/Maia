from cmaia import tree_algo as ctree_algo
from maia.algo.apply_function_to_nodes import zones_iterator

import numpy as np
import maia
import maia.pytree as PT

def enforce_ngon_pe_local(t):
  """
  Shift the ParentElements values in order to make it start at 1, as requested by legacy tools.

  The tree is modified in place.

  Args:
    t (CGNSTree(s)): Tree (or sequences of) starting at Zone_t level or higher.

  """
  for zone in zones_iterator(t):
    try:
      ngon_node = PT.Zone.NGonNode(zone)
    except RuntimeError: #If no NGon, go to next zone
      continue
    pe = PT.get_child_from_name(ngon_node, 'ParentElements')
    pe[1] = maia.algo.indexing.get_ngon_pe_local(ngon_node)

def poly_new_to_old(tree, full_onera_compatibility=True):
  """
  Transform a tree with polyhedral unstructured connectivity with new CGNS 4.x conventions to old CGNS 3.x conventions.

  The tree is modified in place.

  Args:
    tree (CGNSTree): Tree described with new CGNS convention.
    full_onera_compatibility (bool): if ``True``, shift NFace and ParentElements ids to begin at 1, irrespective of the NGon and NFace ElementRanges, and make the NFace connectivity unsigned
  """
  cg_version_node = PT.get_child_from_label(tree, 'CGNSLibraryVersion_t')
  PT.set_value(cg_version_node, 3.1)
  for z in PT.get_all_Zone_t(tree):
    ngon  = maia.pytree.Zone.NGonNode (z)
    nface = maia.pytree.Zone.NFaceNode(z)
    ngon_range   = PT.get_value(PT.get_child_from_name(ngon , "ElementRange"       ))
    nface_range  = PT.get_value(PT.get_child_from_name(nface, "ElementRange"       ))
    nface_connec = PT.get_value(PT.get_child_from_name(nface, "ElementConnectivity"))

    if full_onera_compatibility:
      # 1. shift ParentElements to 1
      pe_node = PT.get_child_from_name(ngon,"ParentElements")
      if pe_node:
        pe = PT.get_value(pe_node)
        pe += (-nface_range[0]+1)*(pe>0)

      # 2. do not use a signed NFace connectivity
      np.absolute(nface_connec,out=nface_connec)

      # 3. shift NFace connectivity to 1
      nface_connec += -ngon_range[0]+1

    # 4. indexed to interleaved
    ctree_algo.indexed_to_interleaved_connectivity(ngon)
    ctree_algo.indexed_to_interleaved_connectivity(nface)


def poly_old_to_new(tree):
  """
  Transform a tree with polyhedral unstructured connectivity with old CGNS 3.x conventions to new CGNS 4.x conventions.

  The tree is modified in place.

  This function accepts trees with old ONERA conventions where NFace and ParentElements ids begin at 1, irrespective of the NGon and NFace ElementRanges, and where the NFace connectivity is unsigned. The resulting tree has the correct CGNS/SIDS conventions.

  Args:
    tree (CGNSTree): Tree described with old CGNS convention.
  """
  cg_version_node = PT.get_child_from_label(tree, 'CGNSLibraryVersion_t')
  PT.set_value(cg_version_node, 4.2)
  for z in PT.get_all_Zone_t(tree):
    ngon  = maia.pytree.Zone.NGonNode (z)
    nface = maia.pytree.Zone.NFaceNode(z)
    ngon_range   = PT.get_value(PT.get_child_from_name(ngon , "ElementRange"))
    nface_range  = PT.get_value(PT.get_child_from_name(nface, "ElementRange"))

    # 1. interleaved to indexed
    ctree_algo.interleaved_to_indexed_connectivity(ngon)

    # 2. shift ParentElements if necessary
    pe_node = PT.get_child_from_name(ngon,"ParentElements")
    if pe_node:
      pe = PT.get_value(pe_node)
      pe_no_0 = pe[pe>0]
      min_pe = np.min(pe_no_0)
      max_pe = np.max(pe_no_0)
      if not (min_pe==nface_range[0] and max_pe==nface_range[1]):
        if min_pe!=1:
          raise RuntimeError("ParentElements values are not SIDS-compliant, and they do not start at 1")
        else:
          pe += (+nface_range[0]-1)*(pe>0)

    # 3. NFace
    nface_connec = PT.get_value(PT.get_child_from_name(nface, "ElementConnectivity"))
    n_cell = nface_range[1] - nface_range[0]
    if np.min(nface_connec)<0 or n_cell==1: # NFace is signed (if only one cell, it is signed despite being positive)
      # 3.1. interleaved to indexed
      ctree_algo.interleaved_to_indexed_connectivity(nface)
      nface_connec = PT.get_value(PT.get_child_from_name(nface, "ElementConnectivity"))

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
