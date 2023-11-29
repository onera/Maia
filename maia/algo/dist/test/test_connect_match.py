import pytest
import pytest_parallel
import numpy as np

import Pypdm.Pypdm as PDM

import maia
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia             import npy_pdm_gnum_dtype as pdm_dtype
from maia.pytree.yaml import parse_yaml_cgns
from maia.factory.dcube_generator import dcube_generate, dcube_struct_generate

from maia.utils import par_utils

from maia.algo.dist import connect_match
from maia.algo.dist import redistribute
dtype = 'I4' if pdm_dtype == np.int32 else 'I8'


def test_shift_face_num():
  zone = parse_yaml_cgns.to_node(f"""
  Zone Zone_t:
    NGON Elements_t [22,0]:
      ElementRange IndexRange_t [1, 25]:
  """)
  # NGON first
  assert (connect_match._shift_face_num([4,6,10], zone) == [4,6,10]).all()
  # NFace first
  er = PT.get_node_from_name(zone, 'ElementRange')[1]
  er[0] = 11
  assert (connect_match._shift_face_num([20,15], zone) == [10,5]).all()
  assert (connect_match._shift_face_num([10,5], zone, True) == [20,15]).all()

  # Elements
  zone = parse_yaml_cgns.to_node(f"""
  Zone Zone_t:
    Tetra Elements_t [10,0]:
      ElementRange IndexRange_t [1, 20]:
    Tri1 Elements_t [5,0]:
      ElementRange IndexRange_t [21, 30]:
    Tri2 Elements_t [5,0]:
      ElementRange IndexRange_t [31, 40]:
  """)
  assert (connect_match._shift_face_num([38,22,40], zone) == [18,2,20]).all()

def test_nodal_sections_to_face_vtx():
  sections = [
     {'pdm_type': PDM._PDM_MESH_NODAL_QUAD4, 'np_distrib': np.array([0, 2, 6]), 'np_connec': np.array([1,2,5,4, 2,3,6,5])},
     {'pdm_type': PDM._PDM_MESH_NODAL_TRIA3, 'np_distrib': np.array([0, 3, 6]), 'np_connec': np.array([6,2,4, 7,6,2, 6,5,3])}
     ]
  face_vtx_idx, face_vtx = connect_match._nodal_sections_to_face_vtx(sections, 0)
  assert (face_vtx_idx == [0,4,8, 11, 14, 17]).all()
  assert (face_vtx == [1,2,5,4, 2,3,6,5, 6,2,4, 7,6,2, 6,5,3]).all()

@pytest_parallel.mark.parallel([1,3])
@pytest.mark.parametrize("input_loc", ['Vertex', 'FaceCenter'])
@pytest.mark.parametrize("output_loc", ['Vertex', 'FaceCenter'])
def test_simple(input_loc, output_loc, comm):                    #    __ 
  n_vtx = 3                                                      #   |  |
  dcubes = [dcube_generate(n_vtx, 1., [0,0,0], comm),            #   |__|
            dcube_generate(n_vtx, 1., [0,0,1], comm)]            #   |  |
  zones = [PT.get_all_Zone_t(dcube)[0] for dcube in dcubes]      #   |__|
  tree = PT.new_CGNSTree()
  base = PT.new_CGNSBase(parent=tree)
  for i_zone,zone in enumerate(zones):
    zone[0] = f"zone{i_zone+1}"
    PT.add_child(base, zone)

  zmax = PT.get_node_from_name(zones[0], 'Zmax')
  PT.new_child(zmax, 'FamilyName', 'FamilyName_t', 'matchA')
  zmin = PT.get_node_from_name(zones[1], 'Zmin')
  PT.new_child(zmin, 'FamilyName', 'FamilyName_t', 'matchB')

  if input_loc == 'Vertex':
    vtx_distri = par_utils.uniform_distribution(n_vtx**2, comm)
    pl_zmin_f = np.arange(n_vtx**2, dtype=pdm_dtype) + 1
    pl_zmax_f = n_vtx**3 - np.arange(n_vtx**2, dtype=pdm_dtype)
    for node, pl in zip([zmin, zmax], [pl_zmin_f, pl_zmax_f]):
      PT.update_child(node, 'GridLocation', value='Vertex')
      PT.update_child(node, 'PointList', value=pl[vtx_distri[0]:vtx_distri[1]].reshape((1,-1), order='F'))
      MT.newDistribution({'Index' : vtx_distri}, node)

  connect_match.connect_1to1_families(tree, ('matchA', 'matchB'), comm, location=output_loc)

  assert len(PT.get_nodes_from_label(tree, 'BC_t')) == 10
  assert len(PT.get_nodes_from_label(tree, 'GridConnectivity_t')) == 2

  if output_loc == 'FaceCenter':
    expected_pl = np.array([[1,2,3,4]], pdm_dtype)
    expected_pld = np.array([[9,10,11,12]], pdm_dtype)
  elif output_loc == 'Vertex':
    expected_pl = np.array([[1,2,3,4,5,6,7,8,9]], pdm_dtype)
    expected_pld = np.array([[19,20,21,22,23,24,25,26,27]], pdm_dtype)

  # Redistribute and compare on rank 0 to avoid managing parallelism
  expected_zmin_full = parse_yaml_cgns.to_node(f"""
  Zmin_0 GridConnectivity_t "Base/zone1":
    GridConnectivityType GridConnectivityType_t "Abutting1to1":
    GridLocation GridLocation_t "{output_loc}":
    GridConnectivityDonorName Descriptor_t "Zmax_0":
    FamilyName FamilyName_t "matchB":
    :CGNS#Distribution UserDefinedData_t:
      Index DataArray_t {dtype} [0,{expected_pl.size},{expected_pl.size}]:
  """)
  PT.new_child(expected_zmin_full, 'PointList', 'IndexArray_t', expected_pl)
  PT.new_child(expected_zmin_full, 'PointListDonor', 'IndexArray_t', expected_pld)

  expected_zmax_full = parse_yaml_cgns.to_node(f"""
  Zmax_0 GridConnectivity_t "Base/zone2":
    GridConnectivityType GridConnectivityType_t "Abutting1to1":
    GridLocation GridLocation_t "{output_loc}":
    GridConnectivityDonorName Descriptor_t "Zmin_0":
    FamilyName FamilyName_t "matchA":
    :CGNS#Distribution UserDefinedData_t:
      Index DataArray_t {dtype} [0,{expected_pl.size},{expected_pl.size}]:
  """)
  PT.new_child(expected_zmax_full, 'PointList', 'IndexArray_t', expected_pld)
  PT.new_child(expected_zmax_full, 'PointListDonor', 'IndexArray_t', expected_pl)

  gather_distri = lambda n_elt, comm : par_utils.gathering_distribution(0, n_elt, comm)

  zmin = PT.get_node_from_name(tree, 'Zmin_0')
  zmax = PT.get_node_from_name(tree, 'Zmax_0')
  redistribute.redistribute_pl_node(zmin, gather_distri, comm)
  redistribute.redistribute_pl_node(zmax, gather_distri, comm)

  if comm.Get_rank() == 0:
    assert PT.is_same_tree(zmin, expected_zmin_full)
    assert PT.is_same_tree(zmax, expected_zmax_full)


@pytest_parallel.mark.parallel(2)
def test_partial_match(comm):                                 #         
  n_vtx = 3                                                   #       __    
  dcubes = [dcube_generate(n_vtx, 1., [0,0,0], comm),         #    __|  |
            dcube_generate(n_vtx, 1., [1,0,.5], comm)]        #   |  |__|
  zones = [PT.get_all_Zone_t(dcube)[0] for dcube in dcubes]   #   |__|
  tree = PT.new_CGNSTree()                                    # 
  base = PT.new_CGNSBase(parent=tree)
  for i_zone,zone in enumerate(zones):
    zone[0] = f"zone{i_zone+1}"
    PT.add_child(base, zone)

  xmax = PT.get_node_from_name(zones[0], 'Xmax')
  PT.new_child(xmax, 'FamilyName', 'FamilyName_t', 'matchA')
  xmin = PT.get_node_from_name(zones[1], 'Xmin')
  PT.new_child(xmin, 'FamilyName', 'FamilyName_t', 'matchB')

  connect_match.connect_1to1_families(tree, ('matchA', 'matchB'), comm)

  assert len(PT.get_nodes_from_label(tree, 'BC_t')) == 12
  assert len(PT.get_nodes_from_label(tree, 'GridConnectivity_t')) == 2

  not_found_xmax = PT.get_node_from_name(zones[0], 'Xmax_unmatched')
  not_found_xmin = PT.get_node_from_name(zones[1], 'Xmin_unmatched')
  found_xmax = PT.get_node_from_name(zones[0], 'Xmax_0')
  found_xmin = PT.get_node_from_name(zones[1], 'Xmin_0')
  if comm.Get_rank() == 0:
    assert (PT.get_child_from_name(not_found_xmax, 'PointList')[1] == [[21]]).all()
    assert (PT.get_child_from_name(    found_xmax, 'PointList')[1] == [[23]]).all()
    assert (PT.get_child_from_name(not_found_xmin, 'PointList')[1] == [[15]]).all()
    assert (PT.get_child_from_name(    found_xmin, 'PointList')[1] == [[13]]).all()
  elif comm.Get_rank() == 1:
    assert (PT.get_child_from_name(not_found_xmax, 'PointList')[1] == [[22]]).all()
    assert (PT.get_child_from_name(    found_xmax, 'PointList')[1] == [[24]]).all()
    assert (PT.get_child_from_name(not_found_xmin, 'PointList')[1] == [[16]]).all()
    assert (PT.get_child_from_name(    found_xmin, 'PointList')[1] == [[14]]).all()


@pytest_parallel.mark.parallel(1)
@pytest.mark.parametrize("out_loc", ['Vertex', 'FaceCenter'])
def test_multiple_match(out_loc, comm):                                      
  # Can not generate ngon with n_vtx != cst so convert it from S -_-
  dcubes = [dcube_struct_generate([5,3,3], [2.,1,1], [0,0,0], comm),  #    __ __
            dcube_struct_generate(3, 1, [0,0,1], comm),               #   |  |  |
            dcube_struct_generate(3, 1, [1,0,1], comm)]               #   |1_|2_|
  zones = [PT.get_all_Zone_t(dcube)[0] for dcube in dcubes]           #   |     |
  tree = PT.new_CGNSTree()                                            #   |__0__|
  base = PT.new_CGNSBase(parent=tree)                                 #
  for i_zone,zone in enumerate(zones):
    zone[0] = f"zone{i_zone+1}"
    PT.add_child(base, zone)
  tree = maia.algo.dist.convert_s_to_ngon(tree, comm)

  zones = PT.get_all_Zone_t(tree)
  zmax = PT.get_node_from_name(zones[0], 'Zmax')
  PT.new_child(zmax, 'FamilyName', 'FamilyName_t', 'matchA')
  for zone in zones[1:]:
    zmin = PT.get_node_from_name(zone, 'Zmin')
    PT.new_child(zmin, 'FamilyName', 'FamilyName_t', 'matchB')

  connect_match.connect_1to1_families(tree, ('matchA', 'matchB'), comm, location=out_loc)

  assert len(PT.get_nodes_from_label(tree, 'BC_t')) == 15
  assert len(PT.get_nodes_from_label(tree, 'GridConnectivity_t')) == 4

  if out_loc == 'FaceCenter':
    assert (PT.get_node_from_predicates(zones[0], ['Zmax_0', 'PointList'])[1] == [[61,62,65,66]]).all()
    assert (PT.get_node_from_predicates(zones[0], ['Zmax_1', 'PointList'])[1] == [[63,64,67,68]]).all()
    assert (PT.get_node_from_predicates(zones[1], ['Zmin_0', 'PointList'])[1] == [[25,26,27,28]]).all()
    assert (PT.get_node_from_predicates(zones[2], ['Zmin_0', 'PointList'])[1] == [[25,26,27,28]]).all()
  elif out_loc == 'Vertex':
    assert (PT.get_node_from_predicates(zones[0], ['Zmax_0', 'PointList'])[1] == [[31,32,33,36,37,38,41,42,43]]).all()
    assert (PT.get_node_from_predicates(zones[0], ['Zmax_1', 'PointList'])[1] == [[33,34,35,38,39,40,43,44,45]]).all()
    assert (PT.get_node_from_predicates(zones[1], ['Zmin_0', 'PointList'])[1] == [[1,2,3,4,5,6,7,8,9]]).all()
    assert (PT.get_node_from_predicates(zones[2], ['Zmin_0', 'PointList'])[1] == [[1,2,3,4,5,6,7,8,9]]).all()



@pytest.mark.parametrize("mesh_kind", ['Poly', 'HEXA_8'])
@pytest_parallel.mark.parallel(1)
def test_periodic_simple(mesh_kind, comm):         #    __
                                                   #   |  | 
                                                   #   |__|
                                                   #
  tree = maia.factory.generate_dist_block(3, mesh_kind, comm)  
  zone = PT.get_node_from_label(tree, 'Zone_t')
    
  xmin = PT.get_node_from_name(tree, 'Xmin')
  PT.new_child(xmin, 'FamilyName', 'FamilyName_t', 'matchA')
  xmax = PT.get_node_from_name(tree, 'Xmax')
  PT.new_child(xmax, 'FamilyName', 'FamilyName_t', 'matchB')

  periodic = {'translation' : np.array([1.0, 0, 0], np.float32)}
  connect_match.connect_1to1_families(tree, ('matchA', 'matchB'), comm, periodic=periodic)

  assert len(PT.get_nodes_from_label(tree, 'BC_t')) == 4
  assert len(PT.get_nodes_from_label(tree, 'GridConnectivity_t')) == 2

  if mesh_kind == 'Poly':
    assert (PT.get_node_from_predicates(zone, ['Xmin_0', 'PointList'])[1] == [[13,14,15,16]]).all()
    assert (PT.get_node_from_predicates(zone, ['Xmax_0', 'PointList'])[1] == [[21,22,23,24]]).all()
  elif mesh_kind == 'HEXA_8':
    assert (PT.get_node_from_predicates(zone, ['Xmin_0', 'PointList'])[1] == [[17,18,19,20]]).all()
    assert (PT.get_node_from_predicates(zone, ['Xmax_0', 'PointList'])[1] == [[21,22,23,24]]).all()

