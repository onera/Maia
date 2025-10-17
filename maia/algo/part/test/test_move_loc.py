import os
import pytest
import pytest_parallel
import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

import maia
from   maia.utils       import test_utils as TU

import maia.algo.part.move_loc as ML

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype
dtype = 'I4' if pdm_gnum_dtype == np.int32 else 'I8'

@pytest_parallel.mark.parallel([1,2])
@pytest.mark.parametrize("cross_domain", [False, True])
def test_centers_to_nodes(cross_domain, comm):
  yaml_path = os.path.join(TU.sample_mesh_dir, 'quarter_crown_square_8.yaml')
  dist_tree = maia.io.file_to_dist_tree(yaml_path, comm)
  for label in ['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t']:
    PT.rm_nodes_from_label(dist_tree, label) # Cleanup
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

  # Create sol on partitions
  for part in PT.get_all_Zone_t(part_tree):
    gnum = MT.globalnumbering_value(part, 'Cell')
    PT.new_FlowSolution('FSol', loc='CellCenter', fields={'gnum': gnum}, parent=part)

  ML.centers_to_nodes(part_tree, comm, ['FSol'], idw_power=0, cross_domain=cross_domain)

  maia.transfer.part_tree_to_dist_tree_only_labels(dist_tree, part_tree, ['FlowSolution_t'], comm)
  dsol_vtx = PT.get_node_from_name(dist_tree, 'FSol#Vtx')
  dfield_vtx = PT.get_node_from_name(dsol_vtx, 'gnum')[1]

  if cross_domain:
    expected_dfield_f = np.array([4.,4.5,5.,4.,4.5,5.,4.,4.5,5.,4.,4.5,5.,4.,4.5,5.,4.,4.5,5.,4.,4.5,5.,4.,4.5,5.,4.,4.5,5.])
  else:
    expected_dfield_f = np.array([1.,1.5,2.,2.,2.5,3.,3.,3.5,4.,3.,3.5,4.,4.,4.5,5.,5.,5.5,6.,5.,5.5,6.,6.,6.5,7.,7.,7.5,8.])
  distri_vtx = MT.distribution_value(PT.get_all_Zone_t(dist_tree)[0], 'Vertex')
  expected_dfield = expected_dfield_f[distri_vtx[0]:distri_vtx[1]]

  assert (dfield_vtx == expected_dfield).all()

@pytest_parallel.mark.parallel(2)
def test_centers_to_nodes_with_empty_zone(comm):
  part_tree = PT.new_CGNSTree()
  part_base = PT.new_CGNSBase(parent=part_tree)
  if comm.rank == 1:
    zones = PT.yaml.to_nodes(f"""
    Zone.P1.N0 Zone_t [[8, 1, 0]]:
      ZoneType ZoneType_t "Unstructured":
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t R8 [0., 1., 0., 1., 0., 1., 0., 1.]:
        CoordinateY DataArray_t R8 [0., 0., 1., 1., 0., 0., 1., 1.]:
        CoordinateZ DataArray_t R8 [0., 0., 0., 0., 1., 1., 1., 1.]:
      NGonElements Elements_t [17,0]:
        ElementRange IndexRange_t [1,1]:
        ElementConnectivity DataArray_t [1,2,3,4,5,6,7,8]:
      FSol FlowSolution_t:
        GridLocation GridLocation_t "CellCenter":
        fieldC DataArray_t [1.]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Cell DataArray_t {dtype} [1]:
        Vertex DataArray_t {dtype} [1,2,3,4,5,6,7,8]:
    """)
    PT.add_child(part_base, zones[0])

  ML.centers_to_nodes(part_tree, comm, ["FSol"])

  if comm.rank==1:
    cnt_n = PT.find_node_from_name_and_label(part_tree, "FSol#Vtx", "FlowSolution_t")
    fld_n = PT.find_child_from_name_and_label(cnt_n, "fieldC", "DataArray_t")
    assert np.array_equal(fld_n[1], np.ones(8, dtype=np.double))

@pytest_parallel.mark.parallel(2)
def test_centers_to_nodes_with_different_n_fld(comm):
  part_tree = PT.new_CGNSTree()
  part_base = PT.new_CGNSBase(parent=part_tree)
  if comm.rank == 0:
    zones = PT.yaml.to_nodes(f"""
    Zone.P0.N0 Zone_t [[8, 1, 0]]:
      ZoneType ZoneType_t "Unstructured":
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t R8 [0., 1., 0., 1., 0., 1., 0., 1.]:
        CoordinateY DataArray_t R8 [0., 0., 1., 1., 0., 0., 1., 1.]:
        CoordinateZ DataArray_t R8 [0., 0., 0., 0., 1., 1., 1., 1.]:
      NGonElements Elements_t [17,0]:
        ElementRange IndexRange_t [1,1]:
        ElementConnectivity DataArray_t [1,2,3,4,5,6,7,8]:
      FSol FlowSolution_t:
        GridLocation GridLocation_t "CellCenter":
        fieldC DataArray_t [2.]:
        fieldD DataArray_t [2.]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Cell DataArray_t {dtype} [2]:
        Vertex DataArray_t {dtype} [9,10,11,12,13,14,15,16]:
    """)
  elif comm.rank == 1:
    zones = PT.yaml.to_nodes(f"""
    Zone.P1.N0 Zone_t [[8, 1, 0]]:
      ZoneType ZoneType_t "Unstructured":
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t R8 [0., 1., 0., 1., 0., 1., 0., 1.]:
        CoordinateY DataArray_t R8 [0., 0., 1., 1., 0., 0., 1., 1.]:
        CoordinateZ DataArray_t R8 [0., 0., 0., 0., 1., 1., 1., 1.]:
      NGonElements Elements_t [17,0]:
        ElementRange IndexRange_t [1,1]:
        ElementConnectivity DataArray_t [1,2,3,4,5,6,7,8]:
      FSol FlowSolution_t:
        GridLocation GridLocation_t "CellCenter":
        fieldC DataArray_t [1.]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Cell DataArray_t {dtype} [1]:
        Vertex DataArray_t {dtype} [1,2,3,4,5,6,7,8]:
    """)
  PT.add_child(part_base, zones[0])
  with pytest.raises(ValueError):
    ML.centers_to_nodes(part_tree, comm, ["FSol"])

@pytest_parallel.mark.parallel(2)
def test_centers_to_nodes_with_different_fld_names(comm):
  part_tree = PT.new_CGNSTree()
  part_base = PT.new_CGNSBase(parent=part_tree)
  if comm.rank == 0:
    zones = PT.yaml.to_nodes(f"""
    Zone.P0.N0 Zone_t [[8, 1, 0]]:
      ZoneType ZoneType_t "Unstructured":
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t R8 [0., 1., 0., 1., 0., 1., 0., 1.]:
        CoordinateY DataArray_t R8 [0., 0., 1., 1., 0., 0., 1., 1.]:
        CoordinateZ DataArray_t R8 [0., 0., 0., 0., 1., 1., 1., 1.]:
      NGonElements Elements_t [17,0]:
        ElementRange IndexRange_t [1,1]:
        ElementConnectivity DataArray_t [1,2,3,4,5,6,7,8]:
      FSol FlowSolution_t:
        GridLocation GridLocation_t "CellCenter":
        fieldD DataArray_t [2.]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Cell DataArray_t {dtype} [2]:
        Vertex DataArray_t {dtype} [9,10,11,12,13,14,15,16]:
    """)
  elif comm.rank == 1:
    zones = PT.yaml.to_nodes(f"""
    Zone.P1.N0 Zone_t [[8, 1, 0]]:
      ZoneType ZoneType_t "Unstructured":
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t R8 [0., 1., 0., 1., 0., 1., 0., 1.]:
        CoordinateY DataArray_t R8 [0., 0., 1., 1., 0., 0., 1., 1.]:
        CoordinateZ DataArray_t R8 [0., 0., 0., 0., 1., 1., 1., 1.]:
      NGonElements Elements_t [17,0]:
        ElementRange IndexRange_t [1,1]:
        ElementConnectivity DataArray_t [1,2,3,4,5,6,7,8]:
      FSol FlowSolution_t:
        GridLocation GridLocation_t "CellCenter":
        fieldC DataArray_t [1.]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Cell DataArray_t {dtype} [1]:
        Vertex DataArray_t {dtype} [1,2,3,4,5,6,7,8]:
    """)
  PT.add_child(part_base, zones[0])
  with pytest.raises(ValueError):
    ML.centers_to_nodes(part_tree, comm, ["FSol"])

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("from_api", [False, True])
def test_nodes_to_centers(from_api, comm):
  dist_tree = maia.factory.generate_dist_block([6,4,2], 'HEXA_8', comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

  # Create sol on partitions
  for part in PT.get_all_Zone_t(part_tree):
    gnum = MT.globalnumbering_value(part, 'Vertex')
    PT.new_FlowSolution('FSol', loc='Vertex', fields={'gnum': gnum}, parent=part)

  if from_api:
    ML.nodes_to_centers(part_tree, comm, ["FSol"])
  else:
    node_to_center = ML.NodeToCenter(part_tree, comm)
    node_to_center.move_fields("FSol")

  maia.transfer.part_tree_to_dist_tree_only_labels(dist_tree, part_tree, ['FlowSolution_t'], comm)
  dsol_cell   = PT.get_node_from_name(dist_tree, 'FSol#Cell')
  dfield_cell = PT.get_node_from_name(dsol_cell, 'gnum')[1]

  elt = PT.get_node_from_predicate(dist_tree, PT.pred.is_element_of_type('HEXA_8'))
  ec = PT.get_child_from_name(elt, 'ElementConnectivity')[1]
  expected_dfield = np.add.reduceat(ec, 8*np.arange(0,ec.size//8)) / 8.

  assert np.allclose(dfield_cell, expected_dfield)

def test_nodes_to_centers_S(comm) :
  dist_tree = maia.factory.generate_dist_block(4, 'S', comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

  zone = PT.get_all_Zone_t(part_tree)[0]
  cx, cy, cz = PT.Zone.coordinates(zone)

  PT.new_FlowSolution('FlowSolution', loc='Vertex', fields={'cX': cx, 'cY': cy, 'cZ': cz}, parent=zone)
  expected = maia.algo.part.geometry._compute_elements_center(zone,3)

  ML.nodes_to_centers(part_tree, comm, ["FlowSolution"])
  sol_cell = PT.find_node_from_name(part_tree, 'FlowSolution#Cell')
  assert PT.get_label(sol_cell) == 'FlowSolution_t'
  for i, dir in enumerate(['X', 'Y', 'Z']):
    field = PT.get_np_value(PT.find_node_from_name(sol_cell, f'c{dir}'))
    assert field.shape == (3,3,3) and field.dtype == float
    assert np.allclose(field.flatten(order='F'), expected[i::3])


def test_centers_to_node_S(comm) :
  dist_tree = maia.factory.generate_dist_block(3, 'S', comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

  maia.algo.compute_elements_center(part_tree, 3, comm)

  ML.centers_to_nodes(part_tree, comm, 'ALL')

  expected_vtx = [[0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 0.25, 0.5,
                    0.75, 0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 0.25, 0.5, 0.75],
                  [0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.25, 0.25, 0.25, 0.5, 0.5,
                    0.5, 0.75, 0.75, 0.75, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75],
                  [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5,
                    0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]]
  sol_cell = PT.find_node_from_name(part_tree, 'Geometry_3d#Vtx')
  assert PT.get_label(sol_cell) == 'DiscreteData_t'
  for i, dir in enumerate(['X', 'Y', 'Z']):
    field = PT.get_np_value(PT.find_node_from_name(sol_cell, f'Center{dir}'))
    assert field.shape == (3,3,3) and field.dtype == float
    assert np.allclose(field.flatten(order='F'), expected_vtx[i])

@pytest_parallel.mark.parallel(3)
def test_all_containers(comm):
  if comm.rank == 0:
    zones = PT.yaml.to_nodes("""
    Zone.P0.N0 Zone_t:
      CellFS FlowSolution_t:
        GridLocation GridLocation_t "CellCenter":
      VtxFS FlowSolution_t:
        GridLocation GridLocation_t "Vertex":
      OtherVtxFS DiscreteData_t: # Skipped because do not exist on P1
      ZSR ZoneSubRegion_t:
      SecondCellFS FlowSolution_t:
        GridLocation GridLocation_t "CellCenter":
    """)
  elif comm.rank == 1:
    zones = PT.yaml.to_nodes("""
    Zone.P1.N0 Zone_t:
      CellFS FlowSolution_t:
        GridLocation GridLocation_t "CellCenter":
      SecondCellFS FlowSolution_t:
        GridLocation GridLocation_t "CellCenter":
      VtxFS FlowSolution_t:
      ZSR ZoneSubRegion_t:
    Zone.P1.N1 Zone_t:
      CellFS FlowSolution_t:
        GridLocation GridLocation_t "CellCenter":
      VtxFS FlowSolution_t:
      OtherCellFS DiscreteData_t: # Skipped because do not exist on other zone
        GridLocation GridLocation_t "CellCenter":
      ZSR ZoneSubRegion_t:
      SecondCellFS FlowSolution_t:
        GridLocation GridLocation_t "CellCenter":
    """)
  else:
    zones = []
  # Add fake array, otherwise containers are not selected
  is_cnt = PT.pred.label_in(['ZoneSubRegion_t', 'FlowSolution_t', 'DiscreteData_t'])
  for zone in zones:
    for cnt in PT.get_children_from_predicate(zone, is_cnt):
      PT.new_DataArray('Pressure', None, parent=cnt)

  # Use fake objs for this test
  CTN = ML.CenterToNode.__new__(ML.CenterToNode)
  CTN.parts = zones
  CTN.comm = comm
  NTC = ML.NodeToCenter.__new__(ML.NodeToCenter)
  NTC.parts = zones
  NTC.comm = comm
  assert CTN.all_containers() == ['CellFS', 'SecondCellFS']
  assert NTC.all_containers() == ['VtxFS']