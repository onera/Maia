import pytest
from pytest_mpi_check._decorator import mark_mpi_test

import numpy as np

import maia.pytree as PT

from maia.factory    import dcube_generator  as DCG
from maia.algo.part  import ngon_tools as NGT

def as_partitioned(zone):
  PT.rm_nodes_from_name(zone, ":CGNS#Distribution")
  predicate = lambda n : PT.get_label(n) in ['Zone_t', 'DataArray_t', 'IndexArray_t', 'IndexRange_t'] \
      and PT.get_value(n).dtype == np.int64
  for array in PT.iter_nodes_from_predicate(zone, predicate, explore='deep'):
    array[1] = array[1].astype(np.int32)

@mark_mpi_test([1])
def test_pe_to_nface(sub_comm):
  tree = DCG.dcube_generate(3,1.,[0,0,0], sub_comm)
  zone = PT.get_all_Zone_t(tree)[0]
  as_partitioned(zone)

  nface_er_exp  = np.array([37,44], np.int32)
  nface_eso_exp = np.array([0, 6, 12, 18, 24, 30, 36, 42, 48], np.int32)
  nface_ec_exp  = np.array([1,5,13,17,25,29,    2,6,21,27,31,-17,
                            3,7,14,18,33,-29,   4,8,22,35,-18,-31,
                            9,15,19,26,30,-5,   10,23,28,32,-6,-19,
                            11,16,20,34,-7,-30, 12,24,36,-8,-20,-32], np.int32)
  nface_exp_f = PT.new_NFaceElements('NFaceElements', erange=nface_er_exp, eso=nface_eso_exp, ec=nface_ec_exp)
  nface_exp = nface_exp_f

  NGT.pe_to_nface(zone, remove_PE=True)
  nface = PT.Zone.NFaceNode(zone)

  assert PT.is_same_tree(nface, nface_exp)
  assert PT.get_node_from_name(zone, "ParentElements") is None

@mark_mpi_test([1])
@pytest.mark.parametrize("rmNFace",[False, True])
def test_nface_to_pe(rmNFace, sub_comm):
  tree = DCG.dcube_generate(3,1.,[0,0,0], sub_comm)
  zone = PT.get_all_Zone_t(tree)[0]
  as_partitioned(zone)
  pe_bck = PT.get_node_from_path(zone, 'NGonElements/ParentElements')[1]
  NGT.pe_to_nface(zone, True)
  nface_bck = PT.get_child_from_name(zone, 'NFaceElements')

  NGT.nface_to_pe(zone, rmNFace)

  assert (PT.get_node_from_path(zone, 'NGonElements/ParentElements')[1] == pe_bck).all()
  nface_cur = PT.get_child_from_name(zone, 'NFaceElements')
  if rmNFace:
    assert nface_cur is None
  else:
    assert PT.is_same_tree(nface_bck, nface_cur)
