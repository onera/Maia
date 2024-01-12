import os
import pytest
import pytest_parallel
import numpy as np

import maia.pytree as PT

import maia
from   maia.utils       import test_utils as TU

import maia.algo.part.move_loc as ML

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
    gnum = PT.maia.getGlobalNumbering(part, 'Cell')[1]
    PT.new_FlowSolution('FSol', loc='CellCenter', fields={'gnum': gnum}, parent=part)

  ML.centers_to_nodes(part_tree, comm, ['FSol'], idw_power=0, cross_domain=cross_domain)

  maia.transfer.part_tree_to_dist_tree_only_labels(dist_tree, part_tree, ['FlowSolution_t'], comm)
  dsol_vtx = PT.get_node_from_name(dist_tree, 'FSol#Vtx')
  dfield_vtx = PT.get_node_from_name(dsol_vtx, 'gnum')[1]

  if cross_domain:
    expected_dfield_f = np.array([4.,4.5,5.,4.,4.5,5.,4.,4.5,5.,4.,4.5,5.,4.,4.5,5.,4.,4.5,5.,4.,4.5,5.,4.,4.5,5.,4.,4.5,5.])
  else:
    expected_dfield_f = np.array([1.,1.5,2.,2.,2.5,3.,3.,3.5,4.,3.,3.5,4.,4.,4.5,5.,5.,5.5,6.,5.,5.5,6.,6.,6.5,7.,7.,7.5,8.])
  distri_vtx = PT.maia.getDistribution(PT.get_all_Zone_t(dist_tree)[0], 'Vertex')[1]
  expected_dfield = expected_dfield_f[distri_vtx[0]:distri_vtx[1]]

  assert (dfield_vtx == expected_dfield).all()

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("from_api", [False, True])
def test_nodes_to_centers(from_api, comm):
  dist_tree = maia.factory.generate_dist_block([6,4,2], 'HEXA_8', comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

  # Create sol on partitions
  for part in PT.get_all_Zone_t(part_tree):
    gnum = PT.maia.getGlobalNumbering(part, 'Vertex')[1]
    PT.new_FlowSolution('FSol', loc='Vertex', fields={'gnum': gnum}, parent=part)

  if from_api:
    ML.nodes_to_centers(part_tree, comm, ["FSol"])
  else:
    node_to_center = ML.NodeToCenter(part_tree, comm)
    node_to_center.move_fields("FSol")

  maia.transfer.part_tree_to_dist_tree_only_labels(dist_tree, part_tree, ['FlowSolution_t'], comm)
  dsol_cell   = PT.get_node_from_name(dist_tree, 'FSol#Cell')
  dfield_cell = PT.get_node_from_name(dsol_cell, 'gnum')[1]

  elt = PT.get_node_from_predicate(dist_tree, lambda n : PT.get_label(n) == 'Elements_t' \
                                                     and PT.Element.CGNSName(n)=='HEXA_8')
  ec = PT.get_child_from_name(elt, 'ElementConnectivity')[1]
  expected_dfield = np.add.reduceat(ec, 8*np.arange(0,ec.size//8)) / 8.

  assert np.allclose(dfield_cell, expected_dfield)

def test_nodes_to_centers_S(comm) : 
    dist_tree = maia.factory.generate_dist_block(4, 'S', comm)
    part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
    
    zone = PT.get_all_Zone_t(part_tree)[0] 
    cx = PT.get_node_from_name(zone, 'CoordinateX')[1]
    cy = PT.get_node_from_name(zone, 'CoordinateY')[1]
    cz = PT.get_node_from_name(zone, 'CoordinateZ')[1]
    PT.new_FlowSolution('FlowSolution', loc='Vertex', fields={'cX': cx, 'cY': cy, 'cZ': cz}, parent=zone)
    expected = maia.algo.part.compute_cell_center(zone)

    ML.nodes_to_centers(part_tree, comm, ["FlowSolution"])
    sol_cell = PT.get_node_from_name(part_tree, 'FlowSolution#Cell')
    for i, dir in enumerate(['X', 'Y', 'Z']):
      field = PT.get_node_from_name(sol_cell, f'c{dir}')[1]
      assert field.shape == (3,3,3) and field.dtype == float
      assert np.allclose(field.flatten(order='F'), expected[i::3])


def test_centers_to_node_S(comm) : 
    dist_tree = maia.factory.generate_dist_block(3, 'S', comm)
    part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
    
    zone = PT.get_all_Zone_t(part_tree)[0] 
    cx, cy, cz = PT.Zone.coordinates(zone)
    expected_center = maia.algo.part.compute_cell_center(zone)
    cx = expected_center[0::3].reshape(PT.Zone.CellSize(zone), order='F')
    cy = expected_center[1::3].reshape(PT.Zone.CellSize(zone), order='F')
    cz = expected_center[2::3].reshape(PT.Zone.CellSize(zone), order='F')
    PT.new_FlowSolution('FlowSolution', loc='CellCenter', fields={'cX': cx, 'cY': cy, 'cZ': cz}, parent=zone)

    ML.centers_to_nodes(part_tree, comm, ["FlowSolution"])
    sol_cell = PT.get_node_from_name(part_tree, 'FlowSolution#Vtx')
    for i, dir in enumerate(['X', 'Y', 'Z']):
      field = PT.get_node_from_name(sol_cell, f'c{dir}')[1]
      expected_vtx = [[0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 0.25, 0.5,
                       0.75, 0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 0.25, 0.5, 0.75],
                      [0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.25, 0.25, 0.25, 0.5, 0.5,
                       0.5, 0.75, 0.75, 0.75, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75],
                      [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5,
                       0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]]
      assert field.shape == (3,3,3) and field.dtype == float
      assert np.allclose(field.flatten(order='F'), expected_vtx[i])


import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD
if __name__ == '__main__':
  # test_nodes_to_centers_S(comm)
  test_centers_to_node_S(comm)
  