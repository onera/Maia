import pytest
import pytest_parallel
import os
import numpy as np

import maia
import maia.pytree as PT

from maia.utils import test_utils as TU

from maia.algo.dist.ngons_to_elements import _ngon_to_elements_zone_3d

def generate_2d_multielt(comm):
  """ Return a NGON 2d tree containing TRI + QUAD elts """

  ftree = PT.yaml.to_cgns_tree("""
  zone Zone_t [[25, 28, 0]]:
    ZoneType ZoneType_t 'Unstructured':
    GridCoordinates GridCoordinates_t:
      CoordinateX DataArray_t:
        R8 : [0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0,
              0.0, 0.25, 0.5, 0.75, 1.0]
      CoordinateY DataArray_t:
        R8 : [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 0.75,
              1.0, 1.0, 1.0, 1.0, 1.0]
      CoordinateZ DataArray_t:
        R8 : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0]
    ZoneBC ZoneBC_t:
      Ymin BC_t 'Null':
        GridLocation GridLocation_t 'EdgeCenter':
        PointList IndexArray_t [[1, 2, 4, 7]]:
      Ymax BC_t 'Null':
        GridLocation GridLocation_t 'EdgeCenter':
        PointList IndexArray_t [[46, 49, 51, 52]]:
      Xmin BC_t 'Null':
        GridLocation GridLocation_t 'EdgeCenter':
        PointList IndexArray_t [[3, 16, 28, 39]]:
      Xmax BC_t 'Null':
        GridLocation GridLocation_t 'EdgeCenter':
        PointList IndexArray_t [[13, 27, 37, 50]]:
    EdgeElements Elements_t [3, 0]:
      ElementRange IndexRange_t [1, 52]:
      ElementConnectivity DataArray_t:
        I4 : [1, 2, 2, 3, 6, 1, 3, 4, 2, 6, 2, 7, 4, 5, 8, 3, 4, 8, 7, 6, 4, 9, 5, 9, 5, 10, 7, 8, 9, 8, 11, 6, 7, 11,
              10, 9, 7, 12, 8, 12, 8, 13, 9, 13, 12, 11, 9, 14, 10, 14, 13, 12, 10, 15, 16, 11, 14, 13, 12, 16, 15, 14,
              12, 17, 13, 17, 13, 18, 17, 16, 14, 19, 15, 20, 18, 17, 21, 16, 19, 18, 17, 21, 17, 22, 19, 20, 18, 22, 18,
              23, 22, 21, 24, 19, 20, 24, 23, 22, 20, 25, 24, 23, 25, 24]
    NGonElements Elements_t [22, 0]:
      ElementRange IndexRange_t [53, 80]:
      ElementStartOffset DataArray_t:
        I4 : [0, 3, 6, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 62, 66, 69, 72, 75, 78, 82,
              85, 88]
      ElementConnectivity DataArray_t:
        I4 : [1, 2, 6, 6, 2, 7, 8, 7, 2, 3, 3, 4, 8, 8, 4, 9, 9, 4, 5, 9, 5, 10, 6, 7, 11, 11, 7, 12, 12, 7, 8, 12, 8,
              13, 13, 8, 9, 13, 9, 14, 14, 9, 10, 14, 10, 15, 11, 12, 16, 16, 12, 17, 17, 12, 13, 17, 13, 18, 18, 13, 14,
              19, 20, 19, 14, 15, 16, 17, 21, 21, 17, 22, 22, 17, 18, 22, 18, 23, 19, 24, 23, 18, 19, 20, 24, 24, 20, 25]
    Sol FlowSolution_t:
      GridLocation GridLocation_t 'CellCenter':
      CellId DataArray_t:
        R8 : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
              20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0]
      nVtx DataArray_t I4 [3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 3, 3, 3, 3, 4, 3, 3]:
  """)
  return maia.factory.full_to_dist_tree(ftree, comm)

@pytest_parallel.mark.parallel(1)
def test_basic(comm):

  filename = os.path.join(TU.mesh_dir, 'hex_2_prism_2.yaml')

  tree = maia.io.file_to_dist_tree(filename, comm)
  zone = PT.get_node_from_label(tree, 'Zone_t')
  maia.algo.dist.convert_elements_to_ngon(tree, comm) # Note: we are not testing that, its just a way to get an ngon test

  _ngon_to_elements_zone_3d(zone, comm) # apply tested function

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

  _ngon_to_elements_zone_3d(zone, comm) # apply tested function

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

@pytest_parallel.mark.parallel(2)
def test_2d_basic(comm):
  tree = maia.factory.generate_dist_block(11, 'QUAD_4', comm)

  maia.algo.dist.convert_elements_to_ngon(tree, comm)
  maia.algo.dist.convert_ngon_to_elements(tree, comm)

  zone = PT.get_all_Zone_t(tree)[0]
  assert PT.get_child_from_name(zone, 'NGonElements') is None
  assert PT.get_child_from_name(zone, 'EdgeElements') is None

  assert (PT.Element.Range(PT.get_child_from_name(zone, 'BAR_2')) == [1, 40]).all()
  assert (PT.Element.Range(PT.get_child_from_name(zone, 'QUAD_4')) == [41, 140]).all()
  for bc in PT.get_nodes_from_label(zone, 'BC_t'):
    pl = PT.get_child_from_name(bc, 'PointList')[1][0]
    assert 1 <= pl.min() and pl.min() <= 40

@pytest_parallel.mark.parallel(2)
def test_2d_multielt(comm):
  tree = generate_2d_multielt(comm)
  maia.algo.dist.convert_ngon_to_elements(tree, comm)

  zone = PT.get_all_Zone_t(tree)[0]
  assert PT.get_child_from_name(zone, 'NGonElements') is None
  assert PT.get_child_from_name(zone, 'EdgeElements') is None

  bar_2 = PT.get_child_from_name(zone, 'BAR_2')
  tri_3 = PT.get_child_from_name(zone, 'TRI_3')
  qua_4 = PT.get_child_from_name(zone, 'QUAD_4')
  assert (PT.Element.Range(bar_2) == [1,  16]).all()
  assert (PT.Element.Range(tri_3) == [17, 40]).all()
  assert (PT.Element.Range(qua_4) == [41, 44]).all()

  tri_expt_f = np.array([1,2,6, 6,2,7, 3,4,8, 8,4,9, 9,4,5, 9,5,10, 6,7,11, 11,7,12, 12,7,8, 12,8,13,
                         13,8,9, 13,9,14, 14,9,10, 14,10,15, 11,12,16, 16,12,17, 17,12,13, 17,13,18,
                         16,17,21, 21,17,22, 22,17,18, 22,18,23, 19,20,24, 24,20,25])
  qua_expt_f = np.array([8,7,2,3, 18,13,14,19, 20,19,14,15, 19,24,23,18])

  tri_distri = PT.maia.getDistribution(tri_3, 'Element')[1]
  qua_distri = PT.maia.getDistribution(qua_4, 'Element')[1]
  assert (PT.get_child_from_name(tri_3, 'ElementConnectivity')[1] == tri_expt_f[3*tri_distri[0]:3*tri_distri[1]]).all()
  assert (PT.get_child_from_name(qua_4, 'ElementConnectivity')[1] == qua_expt_f[4*qua_distri[0]:4*qua_distri[1]]).all()

  cell_id_expt_f = np.array([1., 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
                             17, 18, 19, 22, 23, 24, 25, 27, 28, 3, 20, 21, 26])
  n_vtx_expt_f = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                           3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4])

  cell_distri = PT.maia.getDistribution(zone, 'Cell')[1]
  assert (PT.get_node_from_name(zone, 'CellId')[1] == cell_id_expt_f[cell_distri[0]:cell_distri[1]]).all()
  assert (PT.get_node_from_name(zone, 'nVtx'  )[1] == n_vtx_expt_f  [cell_distri[0]:cell_distri[1]]).all()
