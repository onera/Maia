from pytest_mpi_check._decorator import mark_mpi_test

import numpy as np

import maia.pytree as PT

from maia.factory    import dcube_generator  as DCG
from maia.factory    import full_to_dist     as F2D
from maia.algo.dist  import ngon_tools as NGT

@mark_mpi_test([1,3])
def test_pe_to_nface(sub_comm):
  # 1. Create test input
  tree = DCG.dcube_generate(3,1.,[0,0,0], sub_comm)
  zone = PT.get_node_from_label(tree, 'Zone_t')
  dtype = PT.get_node_from_name(zone, 'ParentElements')[1].dtype

  # 2. Creating expected values
  nface_er_exp  = np.array([37,44], dtype)
  nface_eso_exp = np.array([0, 6, 12, 18, 24, 30, 36, 42, 48], dtype)
  nface_ec_exp  = np.array([1,5,13,17,25,29,    -17,2,6,21,27,31,
                            -29,3,7,14,18,33,   -31,-18,4,8,22,35,
                            -5,9,15,19,26,30,   -19,-6,10,23,28,32,
                            -30,-7,11,16,20,34, -32,-20,-8,12,24,36], dtype)
  nface_exp_f = PT.new_NFaceElements('NFaceElements', erange=nface_er_exp, eso=nface_eso_exp, ec=nface_ec_exp)
  nface_exp = F2D.distribute_element_node(nface_exp_f, sub_comm)

  # 3. Tested function
  NGT.pe_to_nface(zone, sub_comm, True)

  # 4. Check results
  nface = PT.Zone.NFaceNode(zone)
  assert PT.is_same_tree(nface, nface_exp)
  assert PT.get_node_from_name(zone, "ParentElements") is None

@mark_mpi_test([1,3])
def test_nface_to_pe(sub_comm):
  # 1. Create test input
  tree = DCG.dcube_generate(3,1.,[0,0,0], sub_comm)
  zone = PT.get_node_from_label(tree, 'Zone_t')
  pe_bck = PT.get_node_from_path(zone, 'NGonElements/ParentElements')[1]

  NGT.pe_to_nface(zone, sub_comm, True)
  nface_bck = PT.get_node_from_name(zone, 'NFaceElements')

  # 2. Tested function
  rmNface = (sub_comm.size != 3)
  NGT.nface_to_pe(zone, sub_comm, rmNface)
  
  # 3. Check results
  assert (PT.get_node_from_path(zone, 'NGonElements/ParentElements')[1] == pe_bck).all()
  nface_cur = PT.get_node_from_name(zone, 'NFaceElements')
  if rmNface:
    assert nface_cur is None
  else:
    assert PT.is_same_tree(nface_bck, nface_cur)
