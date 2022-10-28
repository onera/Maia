import pytest

import numpy as np
import os
from mpi4py import MPI

from maia.utils import test_utils as TU
from maia.pytree.yaml   import parse_yaml_cgns

import cmaia
import maia
import maia.pytree as PT


@pytest.fixture
def poly_tree_new():
  filename = os.path.join(TU.mesh_dir,'hex_2_prism_2.yaml')
  t = maia.io.file_to_dist_tree(filename,MPI.COMM_SELF)
  maia.algo.dist.elements_to_ngons(t,MPI.COMM_SELF)
  maia.io.distribution_tree.clean_distribution_info(t) # remove distribution info to make it a regular pytree
  return t

@pytest.mark.skipif(not cmaia.cpp20_enabled, reason="Require ENABLE_CPP20 compilation flag")
def test_enfore_ngon_pe_local(poly_tree_new):
  t = poly_tree_new
  maia.algo.seq.enforce_ngon_pe_local(t)

  ngon = PT.get_node_from_name(t,"NGON_n")
  #Those two are not modified
  assert PT.get_value(PT.get_child_from_name(ngon,"ElementStartOffset")).size == 19
  assert PT.get_value(PT.get_child_from_name(ngon,"ElementConnectivity")).size == 69
  pe = PT.get_value(PT.get_child_from_name(ngon,"ParentElements"))

  assert (pe == np.array([[1,0],[2,0],[3,0],[4,0],[1,0],[3,0],[2,0],[4,0],[1,0],
                          [2,0],[1,0],[2,0],[3,0],[4,0],[3,4],[1,3],[1,2],[2,4]])).all()

@pytest.mark.skipif(not cmaia.cpp20_enabled, reason="Require ENABLE_CPP20 compilation flag")
def test_poly_new_to_old(poly_tree_new):
  t = poly_tree_new
  maia.algo.seq.poly_new_to_old(t)

  ngon = PT.get_node_from_name(t,"NGON_n")
  nface = PT.get_node_from_name(t,"NFACE_n")
  ngon_ec = PT.get_value(PT.get_child_from_name(ngon,"ElementConnectivity"))
  pe = PT.get_value(PT.get_child_from_name(ngon,"ParentElements"))
  nface_ec = PT.get_value(PT.get_child_from_name(nface,"ElementConnectivity"))

  assert (ngon_ec == np.array([ 4, 1,6, 9,4,    4, 6 ,11,14, 9,    4, 3, 5,10, 8,    4,  8,10,15,13,
                                4, 1,2, 7,6,    4, 2 , 3, 8, 7,    4, 6, 7,12,11,    4,  7, 8,13,12,
                                4, 4,9,10,5,    4, 9 ,14,15,10,    4, 1, 4, 5, 2,    4, 11,12,15,14,
                                3, 2,5,3   ,    3, 12,13,15   ,    3, 7, 8,10,
                                4, 2,5,10,7,    4, 6 , 7,10, 9,    4, 7,10,15,12] )).all()
  assert (pe == np.array([[1,0],[2,0],[3,0],[4,0],[1,0],[3,0],[2,0],[4,0],[1,0],
                          [2,0],[1,0],[2,0],[3,0],[4,0],[3,4],[1,3],[1,2],[2,4]])).all()
  assert (nface_ec == np.array([ 6, 11,5,16,9,1,17,    6, 17,7,18,10,2,12,    5, 6,3,16,13,15,    5, 8,4,18,15,14])).all()


@pytest.mark.skipif(not cmaia.cpp20_enabled, reason="Require ENABLE_CPP20 compilation flag")
def test_poly_new_to_old_only_interleave(poly_tree_new):
  t = poly_tree_new
  maia.algo.seq.poly_new_to_old(t,full_onera_compatibility=False)

  ngon = PT.get_node_from_name(t,"NGON_n")
  nface = PT.get_node_from_name(t,"NFACE_n")
  ngon_ec = PT.get_value(PT.get_child_from_name(ngon,"ElementConnectivity"))
  pe = PT.get_value(PT.get_child_from_name(ngon,"ParentElements"))
  nface_ec = PT.get_value(PT.get_child_from_name(nface,"ElementConnectivity"))

  assert (ngon_ec == np.array([ 4, 1,6, 9,4,    4, 6 ,11,14, 9,    4, 3, 5,10, 8,    4,  8,10,15,13,
                                4, 1,2, 7,6,    4, 2 , 3, 8, 7,    4, 6, 7,12,11,    4,  7, 8,13,12,
                                4, 4,9,10,5,    4, 9 ,14,15,10,    4, 1, 4, 5, 2,    4, 11,12,15,14,
                                3, 2,5,3   ,    3, 12,13,15   ,    3, 7, 8,10,
                                4, 2,5,10,7,    4, 6 , 7,10, 9,    4, 7,10,15,12] )).all()
  assert (pe == np.array([[19,0],[20,0],[21,0],[22,0],[19,0],[21, 0],[20, 0],[22, 0],[19, 0],
                          [20,0],[19,0],[20,0],[21,0],[22,0],[21,22],[19,21],[19,20],[20,22]])).all()
  assert (nface_ec == np.array([ 6, 11,5,16,9,1,17,    6, -17,7,18,10,2,12,    5, 6,3,-16,13,15,    5, 8,4,-18,-15,14])).all()


@pytest.mark.skipif(not cmaia.cpp20_enabled, reason="Require ENABLE_CPP20 compilation flag")
def test_poly_old_to_new(poly_tree_new):
  t = poly_tree_new
  maia.algo.seq.poly_new_to_old(t) # Note: we are not testing that!

  maia.algo.seq.poly_old_to_new(t)
  ngon = PT.get_node_from_name(t,"NGON_n")
  nface = PT.get_node_from_name(t,"NFaceElements")
  ngon_ec = PT.get_value(PT.get_child_from_name(ngon,"ElementConnectivity"))
  ngon_eso = PT.get_value(PT.get_child_from_name(ngon,"ElementStartOffset"))
  pe = PT.get_value(PT.get_child_from_name(ngon,"ParentElements"))
  nface_ec = PT.get_value(PT.get_child_from_name(nface,"ElementConnectivity"))
  nface_eso = PT.get_value(PT.get_child_from_name(nface,"ElementStartOffset"))

  assert (ngon_eso == np.array([0,4,8,12,16,20,24,28,32,36,40,44,48,51,54,57,61,65,69])).all()
  assert (ngon_ec == np.array([ 1,6, 9,4,    6 ,11,14, 9,    3, 5,10, 8,     8,10,15,13,
                                1,2, 7,6,    2 , 3, 8, 7,    6, 7,12,11,     7, 8,13,12,
                                4,9,10,5,    9 ,14,15,10,    1, 4, 5, 2,    11,12,15,14,
                                2,5,3   ,    12,13,15   ,    7, 8,10,
                                2,5,10,7,    6 , 7,10, 9,    7,10,15,12] )).all()
  assert (pe == np.array([[19,0],[20,0],[21,0],[22,0],[19,0],[21, 0],[20, 0],[22, 0],[19, 0],
                          [20,0],[19,0],[20,0],[21,0],[22,0],[21,22],[19,21],[19,20],[20,22]])).all()
  assert (nface_eso == np.array([0,6,12,17,22])).all()
  assert (nface_ec  == np.array([1,5,9,11,16,17,    2,7,10,12,18,-17,    3,6,13,15,-16,    4,8,14,-15,-18])).all()


@pytest.mark.skipif(not cmaia.cpp20_enabled, reason="Require ENABLE_CPP20 compilation flag")
def test_poly_old_to_new_only_index(poly_tree_new):
  t = poly_tree_new
  maia.algo.seq.poly_new_to_old(t,full_onera_compatibility=False) # Note: we are not testing that!

  maia.algo.seq.poly_old_to_new(t)
  ngon = PT.get_node_from_name(t,"NGON_n")
  nface = PT.get_node_from_name(t,"NFACE_n")
  ngon_ec = PT.get_value(PT.get_child_from_name(ngon,"ElementConnectivity"))
  ngon_eso = PT.get_value(PT.get_child_from_name(ngon,"ElementStartOffset"))
  pe = PT.get_value(PT.get_child_from_name(ngon,"ParentElements"))
  nface_ec = PT.get_value(PT.get_child_from_name(nface,"ElementConnectivity"))
  nface_eso = PT.get_value(PT.get_child_from_name(nface,"ElementStartOffset"))

  assert (ngon_eso == np.array([0,4,8,12,16,20,24,28,32,36,40,44,48,51,54,57,61,65,69])).all()
  assert (ngon_ec == np.array([ 1,6, 9,4,    6 ,11,14, 9,    3, 5,10, 8,     8,10,15,13,
                                1,2, 7,6,    2 , 3, 8, 7,    6, 7,12,11,     7, 8,13,12,
                                4,9,10,5,    9 ,14,15,10,    1, 4, 5, 2,    11,12,15,14,
                                2,5,3   ,    12,13,15   ,    7, 8,10,
                                2,5,10,7,    6 , 7,10, 9,    7,10,15,12] )).all()
  assert (pe == np.array([[19,0],[20,0],[21,0],[22,0],[19,0],[21, 0],[20, 0],[22, 0],[19, 0],
                          [20,0],[19,0],[20,0],[21,0],[22,0],[21,22],[19,21],[19,20],[20,22]])).all()
  assert (nface_eso == np.array([0,6,12,17,22])).all()
  assert (nface_ec  == np.array([11,5,16,9,1,17,    -17,7,18,10,2,12,    6,3,-16,13,15,    8,4,-18,-15,14])).all()


@pytest.mark.skipif(not cmaia.cpp20_enabled, reason="Require ENABLE_CPP20 compilation flag")
def test_poly_old_to_new_no_pe(poly_tree_new):
  t = poly_tree_new
  maia.algo.seq.poly_new_to_old(t) # Note: we are not testing that!
  maia.pytree.rm_nodes_from_name(t,"ParentElements")

  with pytest.raises(RuntimeError):
    maia.algo.seq.poly_old_to_new(t)
