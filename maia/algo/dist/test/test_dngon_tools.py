import pytest_parallel

import numpy as np

import maia.pytree as PT

import maia
from maia.factory    import dcube_generator   as DCG
from maia.factory    import dsphere_generator as DSG
from maia.factory    import full_to_dist      as F2D
from maia.algo.dist  import ngon_tools as NGT

@pytest_parallel.mark.parallel([1,3])
def test_pe_to_nface(comm):
  # 1. Create test input
  tree = DCG.dcube_generate(3,1.,[0,0,0], comm)
  zone = PT.get_node_from_label(tree, 'Zone_t')
  dtype = PT.get_node_from_name(zone, 'ParentElements')[1].dtype

  # 2. Creating expected values
  nface_er_exp  = np.array([37,44], dtype)
  nface_eso_exp = np.array([0, 6, 12, 18, 24, 30, 36, 42, 48], dtype)
  nface_ec_exp = np.array([-29,-17,-5,1,13,25, -31,-6,2,17,21,27,  -18,-7,3,14,29,33,
                           -8,4,18,22,31,35,   -30,-19,5,9,15,26,  -32,6,10,19,23,28,
                           -20,7,11,16,30,34,   8,12,20,24,32,36],dtype)
  nface_exp_f = PT.new_NFaceElements('NFaceElements', erange=nface_er_exp, eso=nface_eso_exp, ec=nface_ec_exp)
  nface_exp = F2D.distribute_element_node(nface_exp_f, comm)

  # 3. Tested function
  NGT.pe_to_nface(zone, comm, True)

  # 4. Check results
  nface = PT.Zone.NFaceNode(zone)
  assert PT.is_same_tree(nface, nface_exp)
  assert PT.get_node_from_name(zone, "ParentElements") is None

@pytest_parallel.mark.parallel([1,3])
def test_nface_to_pe(comm):
  # 1. Create test input
  tree = DCG.dcube_generate(3,1.,[0,0,0], comm)
  zone = PT.get_node_from_label(tree, 'Zone_t')
  pe_bck = PT.get_node_from_path(zone, 'NGonElements/ParentElements')[1]

  NGT.pe_to_nface(zone, comm, True)
  nface_bck = PT.get_node_from_name(zone, 'NFaceElements')

  # 2. Tested function
  rmNface = (comm.size != 3)
  NGT.nface_to_pe(zone, comm, rmNface)

  # 3. Check results
  assert (PT.get_node_from_path(zone, 'NGonElements/ParentElements')[1] == pe_bck).all()
  nface_cur = PT.get_node_from_name(zone, 'NFaceElements')
  if rmNface:
    assert nface_cur is None
  else:
    assert PT.is_same_tree(nface_bck, nface_cur)

@pytest_parallel.mark.parallel(2)
def test_ngon_to_pe(comm):
  tree = DCG.generate_dist_block(4, "QUAD_4", comm)
  maia.algo.dist.convert_elements_to_ngon(tree, comm)

  zone = PT.get_node_from_label(tree, 'Zone_t')
  pe_bck = PT.get_node_from_path(zone, 'EdgeElements/ParentElements')[1]

  PT.rm_nodes_from_name(zone, 'ParentElements')
  NGT.ngon_to_edge_pe(zone, comm)
  pe = PT.get_node_from_path(zone, 'EdgeElements/ParentElements')[1]

  assert np.array_equal(pe_bck, pe)

@pytest_parallel.mark.parallel(2)
def test_pe_to_ngon(comm):
  tree = DSG.generate_dist_sphere(5, 'NGON_n', comm)

  zone = PT.get_node_from_label(tree, 'Zone_t')
  ngon_bck = PT.get_node_from_path(zone, 'NGonElements')
  ngon_er_bck  = PT.Element.Range(ngon_bck)
  ngon_eso_bck = PT.get_node_from_path(ngon_bck, 'ElementStartOffset')[1]
  ngon_ec_bck  = PT.get_node_from_path(ngon_bck, 'ElementConnectivity')[1]

  PT.rm_nodes_from_name(zone, 'NGonElements')
  maia.algo.edge_pe_to_ngon(zone, comm)
  ngon = PT.get_node_from_path(zone, 'NGonElements')
  ngon_er  = PT.Element.Range(ngon)
  ngon_eso = PT.get_node_from_path(ngon, 'ElementStartOffset')[1]
  ngon_ec  = PT.get_node_from_path(ngon, 'ElementConnectivity')[1]

  assert np.array_equal(ngon_er_bck, ngon_er)
  assert np.array_equal(ngon_eso_bck, ngon_eso)
  assert np.array_equal(ngon_ec_bck, ngon_ec)
