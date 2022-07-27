from cmaia import tree_algo as ctree_algo

import numpy as np
import Converter.Internal as I
import maia

def poly_new_to_old(t, full_onera_compatibility=True):
  for z in I.getZones(t):
    ngon  = maia.pytree.Zone.NGonNode (z)
    nface = maia.pytree.Zone.NFaceNode(z)
    ngon_range   = I.getVal(I.getNodeFromName1(ngon , "ElementRange"       ))
    nface_range  = I.getVal(I.getNodeFromName1(nface, "ElementRange"       ))
    nface_connec = I.getVal(I.getNodeFromName1(nface, "ElementConnectivity"))

    if full_onera_compatibility:
      # 1. shift ParentElements to 1
      pe_node = I.getNodeFromName1(ngon,"ParentElements")
      if pe_node:
        pe = I.getVal(pe_node)
        pe += (-nface_range[0]+1)*(pe>0)

      # 2. do not use a signed NFace connectivity
      np.absolute(nface_connec,out=nface_connec)

      # 3. shift NFace connectivity to 1
      nface_connec += -ngon_range[0]+1

    # 4. indexed to interleaved
    ctree_algo.indexed_to_interleaved_connectivity(ngon)
    ctree_algo.indexed_to_interleaved_connectivity(nface)


def poly_old_to_new(t):
  for z in I.getZones(t):
    ngon  = maia.pytree.Zone.NGonNode (z)
    nface = maia.pytree.Zone.NFaceNode(z)
    ngon_range   = I.getVal(I.getNodeFromName1(ngon , "ElementRange"       ))
    nface_range  = I.getVal(I.getNodeFromName1(nface, "ElementRange"       ))

    # 1. interleaved to indexed
    ctree_algo.interleaved_to_indexed_connectivity(ngon)

    # 2. shift ParentElements if necessary
    pe_node = I.getNodeFromName1(ngon,"ParentElements")
    if pe_node:
      pe = I.getVal(pe_node)
      pe_no_0 = pe[pe>0]
      min_pe = np.min(pe_no_0)
      max_pe = np.max(pe_no_0)
      if not (min_pe==nface_range[0] and max_pe==nface_range[1]):
        if min_pe!=1:
          raise RuntimeError("ParentElements values are not SIDS-compliant, and they do not start at 1")
        else:
          pe += (+nface_range[0]-1)*(pe>0)

    # 3. NFace
    nface_connec = I.getVal(I.getNodeFromName1(nface, "ElementConnectivity"))
    if np.min(nface_connec)<0: # NFace is signed
      # 3.1. interleaved to indexed
      ctree_algo.interleaved_to_indexed_connectivity(nface)
      nface_connec = I.getVal(I.getNodeFromName1(nface, "ElementConnectivity"))

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
      I._rmNode(z,nface)
      if not pe_node:
        raise RuntimeError("NFace is not signed: this is not compliant. However, a ParentElements is needed to recompute a correct NFace")
      if ngon_range[0] != 1:
        raise NotImplementedError("NFace is not signed: this is not compliant. It needs to be recomputed, but not implemented in case NGon is not first")
      maia.algo.pe_to_nface(z)
