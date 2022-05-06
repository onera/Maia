import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.factory       import dcube_generator as DCG
from maia.algo.part     import connectivity_transform as CNT
from maia.algo.indexing import get_ngon_pe_local

@mark_mpi_test(1)
def test_enforce_boundary_pe_left(sub_comm):
  tree = DCG.dcube_generate(3, 1., [0., 0., 0.], sub_comm)
  zone = I.getZones(tree)[0]
  #### Add nface (todo : use maia func when available; reput in eso form after)
  #Careful, fix ngon do crazy stuff if pe start at 1
  pe_node = I.getNodeFromPath(zone, 'NGonElements/ParentElements')
  pe_node[1] = get_ngon_pe_local(I.getNodeFromName(zone, 'NGonElements'))
  I._fixNGon(zone)
  for node_name, lsize in zip(['NFaceElements', 'NGonElements'], [6,4]):
    node = I.getNodeFromName1(zone, node_name)
    ec = I.getNodeFromName1(node, 'ElementConnectivity')
    er = I.getNodeFromName1(node, 'ElementRange')
    n_elem = er[1][1] - er[1][0] + 1
    eso = np.arange(0, (n_elem+1)*lsize, lsize, np.int32)
    ec[1] = np.delete(ec[1], eso[:-1] + np.arange(n_elem))
    I.newDataArray('ElementStartOffset', eso, node)
  pe_node[1] += n_elem*(pe_node[1] > 0)
  #### Todo : remove preceding stuff when we have pe -> nface conversion
  pe_bck = pe_node[1].copy()
  zone_bck = I.copyTree(zone)
  CNT.enforce_boundary_pe_left(zone)
  assert PT.is_same_tree(zone, zone_bck)

  #Test with swapped pe
  pe_node[1][2] = pe_node[1][2][::-1]
  CNT.enforce_boundary_pe_left(zone)

  assert I.getNodeFromPath(zone, 'NFaceElements/ElementConnectivity')[1][12] == -3
  expt_ng_ec = I.getNodeFromPath(zone_bck, 'NGonElements/ElementConnectivity')[1].copy()
  expt_ng_ec[4*2 : 4*3] = [5, 4, 7, 8]
  assert (I.getNodeFromPath(zone, 'NGonElements/ElementConnectivity')[1] == expt_ng_ec).all()

  #Test with no NFace
  zone = I.copyTree(zone_bck)
  pe_node = I.getNodeFromPath(zone, 'NGonElements/ParentElements')
  pe_node[1][2] = pe_node[1][2][::-1]
  I._rmNodesByName(zone, 'NFaceElements')
  CNT.enforce_boundary_pe_left(zone)
  assert (I.getNodeFromPath(zone, 'NGonElements/ElementConnectivity')[1] == expt_ng_ec).all()
