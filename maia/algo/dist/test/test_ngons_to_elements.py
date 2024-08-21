import pytest
import pytest_parallel
import os
import numpy as np

import maia
import maia.pytree as PT

from maia.utils import test_utils as TU

from maia.algo.dist.ngons_to_elements import _ngon_to_elements_zone

@pytest_parallel.mark.parallel(1)
def test_basic(comm):

  filename = os.path.join(TU.mesh_dir, 'hex_2_prism_2.yaml')

  tree = maia.io.file_to_dist_tree(filename, comm)
  zone = PT.get_node_from_label(tree, 'Zone_t')
  maia.algo.dist.convert_elements_to_ngon(tree, comm) # Note: we are not testing that, its just a way to get an ngon test

  _ngon_to_elements_zone(zone, comm) # apply tested function

  # Checks
  expected_range = [[1,2],    # TRI_3
                    [3,14],   # QUAD_4
                    [15,16],  # PENTA_6
                    [17,18]]  # HEXA_8
  expected_ec = [[5,3,2,13,15,12],
                 [4,5,2,1,  2,7,6,1,  6,9,4,1,  8,7,2,3,  10,8,3,5,  9,10,5,4,  7,12,11,6,  11,14,9,6,  13,12,7,8,  15,13,8,10,  14,15,10,9,  15,14,11,12],
                 [2,3,5,7,8,10,  8,10,7,13,15,12],
                 [1,2,5,4,6,7,10,9,  10,9,6,7,15,14,11,12]]

  for i,kind in enumerate(['TRI_3', 'QUAD_4', 'PENTA_6', 'HEXA_8']):
    elt = PT.get_child_from_name(zone, kind)
    assert PT.Element.CGNSName(elt) == kind
    assert np.array_equal(PT.get_child_from_name(elt, "ElementRange")[1], expected_range[i])
    assert np.array_equal(PT.get_child_from_name(elt, "ElementConnectivity")[1], expected_ec[i])


def test_all_kinds(comm):
  filename = os.path.join(TU.mesh_dir, 'hex_prism_pyra_tet.yaml')

  tree = maia.io.file_to_dist_tree(filename, comm)
  zone = PT.get_node_from_label(tree, 'Zone_t')
  maia.algo.dist.convert_elements_to_ngon(tree, comm) # Note: we are not testing that, its just a way to get an ngon test

  _ngon_to_elements_zone(zone, comm) # apply tested function

   # Checks
  expected_range = [[1,6],    # TRI_3
                    [7,12],   # QUAD_4
                    [13,13],  # TETRA_4
                    [14,14],  # PYRA_5
                    [15,15],  # PENTA_6
                    [16,16]]  # HEXA_8
  expected_ec = [[5,3,2,  11,6,7,  11,9,6,  7,8,11,  8,10,11,  11,10,9],
                 [4,5,2,1,  2,7,6,1,  1,6,9,4,  8,7,2,3,  3,5,10,8,  9,10,5,4],
                 [11,7,10,8],
                 [10,9,6,7,11],
                 [2,3,5,7,8,10,],
                 [1,2,5,4,6,7,10,9]]

  for i,kind in enumerate(['TRI_3', 'QUAD_4', 'TETRA_4', 'PYRA_5', 'PENTA_6', 'HEXA_8']):
    elt = PT.get_child_from_name(zone, kind)
    assert PT.Element.CGNSName(elt) == kind
    assert np.array_equal(PT.get_child_from_name(elt, "ElementRange")[1], expected_range[i])
    assert np.array_equal(PT.get_child_from_name(elt, "ElementConnectivity")[1], expected_ec[i]) # Checks

@pytest_parallel.mark.parallel(2)
def test_multi_sections(comm):
  mesh_file = os.path.join(TU.mesh_dir, 'multi_element.yaml')
  dist_tree = maia.io.file_to_dist_tree(mesh_file, comm)
  zone = PT.get_node_from_label(dist_tree, 'Zone_t')

  # Add a CellCenter field manually
  cell_kind_full = np.concatenate([np.ones(24), 2*np.ones(24), 3*np.ones(120)]) #TETRA, PENTA, TETRA
  cell_distri = PT.maia.getDistribution(zone, 'Cell')[1]
  cell_kind = cell_kind_full[cell_distri[0]:cell_distri[1]]

  PT.new_FlowSolution('FSCC', loc='CellCenter', fields={'IniSection':cell_kind}, parent=zone)

  maia.algo.dist.convert_elements_to_ngon(dist_tree, comm)
  maia.algo.dist.convert_ngon_to_elements(dist_tree, comm)

  assert len(PT.get_children_from_label(zone, 'Elements_t')) == 4
  assert (PT.Element.Range(PT.get_child_from_name(zone, 'TRI_3'))   == [  1,  88]).all()
  assert (PT.Element.Range(PT.get_child_from_name(zone, 'QUAD_4'))  == [ 89, 104]).all()
  assert (PT.Element.Range(PT.get_child_from_name(zone, 'TETRA_4')) == [105, 248]).all()
  assert (PT.Element.Range(PT.get_child_from_name(zone, 'PENTA_6')) == [249, 272]).all()

  for bc in PT.get_nodes_from_label(zone, 'BC_t'):
    pl = PT.get_child_from_name(bc, 'PointList')[1][0]
    if PT.Subset.GridLocation(bc) == 'FaceCenter':
      assert 1 <= pl.min() and pl.min() <= 104
    elif PT.Subset.GridLocation(bc) == 'CellCenter':
      assert 105 <= pl.min() and pl.min() <= 272
  
  cell_kind_full_expt = np.concatenate([np.ones(24), 3*np.ones(120), 2*np.ones(24)]) #TETRA, TETRA, PENTA
  cell_kind_expt = cell_kind_full_expt[cell_distri[0]:cell_distri[1]]

  assert (PT.get_node_from_name(zone, 'IniSection')[1] == cell_kind_expt).all()