import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia
from maia.factory       import dcube_generator as DCG
from maia.algo.part     import connectivity_transform as CNT

@mark_mpi_test(1)
def test_enforce_boundary_pe_left(sub_comm):
  tree = DCG.dcube_generate(3, 1., [0., 0., 0.], sub_comm)
  zone = PT.get_all_Zone_t(tree)[0]
  maia.algo.pe_to_nface(zone, sub_comm)
  pe_node = PT.get_node_from_path(zone, 'NGonElements/ParentElements')
  pe_bck = pe_node[1].copy()
  zone_bck = PT.deep_copy(zone)
  CNT.enforce_boundary_pe_left(zone)
  assert PT.is_same_tree(zone, zone_bck)

  #Test with swapped pe
  pe_node[1][2] = pe_node[1][2][::-1]
  CNT.enforce_boundary_pe_left(zone)

  assert PT.get_node_from_path(zone, 'NFaceElements/ElementConnectivity')[1][12] == -29
  expt_ng_ec = PT.get_node_from_path(zone_bck, 'NGonElements/ElementConnectivity')[1].copy()
  expt_ng_ec[4*2 : 4*3] = [5, 4, 7, 8]
  assert (PT.get_node_from_path(zone, 'NGonElements/ElementConnectivity')[1] == expt_ng_ec).all()

  #Test with no NFace
  zone = PT.deep_copy(zone_bck)
  pe_node = PT.get_node_from_path(zone, 'NGonElements/ParentElements')
  pe_node[1][2] = pe_node[1][2][::-1]
  PT.rm_children_from_name(zone, 'NFaceElements')
  CNT.enforce_boundary_pe_left(zone)
  assert (PT.get_node_from_path(zone, 'NGonElements/ElementConnectivity')[1] == expt_ng_ec).all()
