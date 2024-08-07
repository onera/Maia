import pytest
import pytest_parallel
import numpy as np

import maia.pytree as PT

import maia

from maia import npy_pdm_gnum_dtype as pdm_dtype
from maia.utils import par_utils
from maia.algo.dist import sections_tools

@pytest_parallel.mark.parallel(3)
def test_gather_sections(comm):
  tree = PT.new_CGNSTree()
  base = PT.new_CGNSBase(parent=tree)
  zone = PT.new_Zone(type='Unstructured', size=[[30,10,0]], parent=base)
  PT.new_Elements('Tri.1', 'TRI_3', erange=[18,20], econn=[22,23,24, 25,26,27, 28,29,30], parent=zone)
  PT.new_Elements('Tri.0', 'TRI_3', erange=[11,17], econn=[1,2,3, 4,5,6, 7,8,9, 10,11,12, 13,14,15, 16,17,18, 19,20,21], parent=zone)
  
  tree = maia.factory.full_to_dist_tree(tree, comm)
  tree_bck = PT.deep_copy(tree)

  sections_tools.concatenate_elt_sections(tree, comm)

  elts = PT.get_nodes_from_label(tree, 'Elements_t')
  assert len(elts) == 1

  expected_ec = [[1,2,3, 4,5,6, 7,8,9, 10,11,12],
                 [13,14,15, 16,17,18, 19,20,21],
                 [22,23,24, 25,26,27, 28,29,30]][comm.Get_rank()]
  expected_distri_f = np.array([0, 4, 7, 10], pdm_dtype)
  expected_distri = par_utils.full_to_partial_distribution(expected_distri_f, comm)
  expected = PT.new_Elements('TRI_3', 'TRI_3', erange=[11, 20], econn=expected_ec)
  PT.maia.newDistribution({'Element' : expected_distri}, expected)

  assert PT.is_same_tree(elts[0], expected)


  # Test failure with non contiguous elts
  PT.get_node_from_name(tree_bck, 'ElementRange')[1] += 3
  with pytest.raises(RuntimeError):
    sections_tools.concatenate_elt_sections(tree_bck, comm)

def test_reorder_elements():
  # Note:  Ids in this tree makes no sense, this is just to test
  tree = PT.new_CGNSTree()
  base = PT.new_CGNSBase(parent=tree)
  zone = PT.new_Zone(type='Unstructured', parent=base)
  PT.new_Elements('Tetra', 'TETRA_4', erange=[1,15], parent=zone)
  tri = PT.new_Elements('Tri', 'TRI_3', erange=[16,25], parent=zone)
  PT.new_DataArray('ParentElements', [[1,26],[30,0],[12,0],[26,15]], parent=tri)
  PT.new_Elements('Pyra', 'PYRA_5', erange=[26,30], parent=zone)
  zbc = PT.new_ZoneBC(parent=zone)
  PT.new_BC('BC1', loc='FaceCenter', point_list=[[4,8,20,24,15,28]], parent=zbc)
  PT.new_BC('BC2', loc='FaceCenter', point_range=[[12,19]], parent=zbc)
  PT.new_BC('BC3', loc='Vertex', point_list=[[1,5,15]], parent=zbc)
  zgc = PT.new_ZoneGridConnectivity(parent=zone)
  PT.new_GridConnectivity('match', 'OppZone', 'Abutting1to1', loc='FaceCenter', point_list=[[23,24,25]], parent=zgc)
  
  opp_zone = PT.new_Zone('OppZone', type='Unstructured', parent=base)
  zgc = PT.new_ZoneGridConnectivity(parent=opp_zone)
  PT.new_GridConnectivity('match', 'Zone', 'Abutting1to1', loc='FaceCenter', point_list_donor=[[23,24,25]], parent=zgc)

  new_ord = [PT.get_child_from_name(zone, e) for e in ['Tri', 'Pyra', 'Tetra']]
  new_ord = lambda elts: [elts[1], elts[2], elts[0]] if len(elts) == 3 else elts # Custom order for this test

  sections_tools.reorder_sections(tree, new_ord)

  assert (PT.Element.Range(PT.get_node_from_name(zone, 'Tri')) == [1,10]).all()
  assert (PT.Element.Range(PT.get_node_from_name(zone, 'Pyra')) == [11,15]).all()
  assert (PT.Element.Range(PT.get_node_from_name(zone, 'Tetra')) == [16,30]).all()

  assert (PT.get_node_from_name(zone, 'ParentElements')[1] == [[16,11],[15,0],[27,0],[11,30]]).all()
  assert (PT.get_node_from_path(zone, 'ZoneBC/BC1/PointList')[1] == [[19,23, 5,9,30,13]]).all()
  assert (PT.get_node_from_path(zone, 'ZoneBC/BC2/PointList')[1] == [[27,28,29,30, 1,2,3,4]]).all()
  assert (PT.get_node_from_path(zone, 'ZoneBC/BC3/PointList')[1] == [[1,5,15]]).all()

  assert (PT.get_node_from_name(opp_zone, 'PointListDonor')[1] == [[8,9,10]]).all()

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize('elt_kind', ['TETRA_4', 'NFACE_n'])
def test_reorder_elements_per_dim(elt_kind, comm):
  tree = maia.factory.generate_dist_block(5, elt_kind, comm)
  if elt_kind == 'NFACE_n':
    maia.algo.nface_to_pe(tree, comm)
  tree_bck = PT.deep_copy(tree)

  # If NGON : reverse = True then False (initially, 2D first)
  # If Tetra : reverse = False then True (initially, 3D first)      
  reverse = elt_kind == 'NFACE_n'
  sections_tools.reorder_elt_sections_from_dim(tree,     reverse)
  sections_tools.reorder_elt_sections_from_dim(tree, not reverse)

  assert PT.is_same_tree(tree, tree_bck)