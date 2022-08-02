import pytest
from   pytest_mpi_check._decorator import mark_mpi_test
import os
import numpy as np

import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils import test_utils as TU

from maia.io       import file_to_dist_tree, dist_tree_to_file
from maia.factory  import generate_dist_block

from maia.algo.dist import duplicate as DUP

@mark_mpi_test([2])
@pytest.mark.parametrize("fields", [True, False])
def test_translate_cube(sub_comm, fields, write_output):
  # Generate a disttree with one zone
  dist_tree = generate_dist_block(11, "Poly", sub_comm, origin=[0., -.5, -.5])
  dist_zone = I.getZones(dist_tree)[0]

  # Initialise some fields
  cell_distri = MT.getDistribution(dist_zone, 'Cell')[1]
  n_cell_loc =  cell_distri[1] - cell_distri[0]
  fs = I.newFlowSolution('FlowSolution', gridLocation='CellCenter', parent=dist_zone)
  I.newDataArray('scalar', np.random.random(n_cell_loc), parent=fs)
  I.newDataArray('vectX', np.random.random(n_cell_loc), parent=fs)
  I.newDataArray('vectY', np.random.random(n_cell_loc), parent=fs)
  I.newDataArray('vectZ', np.random.random(n_cell_loc), parent=fs)

  # One can apply a rotation and/or a translation to the zone
  transformed_zone = DUP.duplicate_zone_with_transformation(dist_zone, 'DuplicatedZone', \
      rotation_angle = [0, 0, np.pi], apply_to_fields=fields)

  # Coordinates are moved :
  coords    = PT.Zone.coordinates(dist_zone)
  tr_coords = PT.Zone.coordinates(transformed_zone)
  assert np.allclose(-coords[0], tr_coords[0])
  assert np.allclose(-coords[1], tr_coords[1])
  assert np.allclose(coords[2], tr_coords[2])

  # If apply_to_fields is True, vectorial fields are moved as well :
  assert PT.is_same_node(I.getNodeFromName(dist_zone, 'scalar'), I.getNodeFromName(transformed_zone, 'scalar'))
  assert PT.is_same_node(I.getNodeFromName(dist_zone, 'vectZ'), I.getNodeFromName(transformed_zone, 'vectZ'))
  if fields:
    assert np.allclose(-I.getNodeFromName(dist_zone, 'vectX')[1], I.getNodeFromName(transformed_zone, 'vectX')[1])
    assert np.allclose(-I.getNodeFromName(dist_zone, 'vectY')[1], I.getNodeFromName(transformed_zone, 'vectY')[1])
  else:
    assert PT.is_same_node(I.getNodeFromName(dist_zone, 'vectX'), I.getNodeFromName(transformed_zone, 'vectX'))
    assert PT.is_same_node(I.getNodeFromName(dist_zone, 'vectY'), I.getNodeFromName(transformed_zone, 'vectY'))

  if write_output:
    dist_base = I.getBases(dist_tree)[0]
    I._addChild(dist_base, transformed_zone)
    out_dir = TU.create_pytest_output_dir(sub_comm)
    dist_tree_to_file(dist_tree, os.path.join(out_dir, 'duplicated.hdf'), sub_comm)

@mark_mpi_test([1])
def test_duplicate_from_periodic(sub_comm, write_output):
  dist_tree = generate_dist_block(11, "Poly", sub_comm)
  # Lets create a periodic join for this cube
  dist_zone = I.getZones(dist_tree)[0]
  bottom = I.getNodeFromName(dist_zone, 'Zmin')
  top    = I.getNodeFromName(dist_zone, 'Zmax')
  for bc in [bottom, top]:
    PT.rm_nodes_from_name(dist_zone, I.getName(bc))
    I.setType(bc, 'GridConnectivity_t')
    I.setValue(bc, I.getName(dist_zone))
    I.newGridConnectivityType('Abutting1to1', bc)
  I.newIndexArray('PointListDonor', I.getNodeFromName(top, 'PointList')[1].copy(), parent=bottom) #OK in sequential
  I.newIndexArray('PointListDonor', I.getNodeFromName(bottom, 'PointList')[1].copy(), parent=top) #OK in sequential
  I.createChild(bottom, 'GridConnectivityProperty', 'GridConnectivityProperty_t', children=[I.newPeriodic(translation=[0.,0, 1])])
  I.createChild(top,    'GridConnectivityProperty', 'GridConnectivityProperty_t', children=[I.newPeriodic(translation=[0.,0, -1])])
  I.createChild(dist_zone, 'ZoneGridConnectivity', 'ZoneGridConnectivity_t', children=[bottom, top])

  # If a periodic (rotation and/or translation) JN is present in the mesh, the next function can duplicate
  # the zones and move it at the interface location, any number of time
  first_side_jn  = [f'Base/zone/ZoneGridConnectivity/{I.getName(bottom)}']
  second_side_jn = [f'Base/zone/ZoneGridConnectivity/{I.getName(top)}']
  opposite_jns = [first_side_jn, second_side_jn]

  assert len(I.getZones(dist_tree)) == 1
  DUP.duplicate_from_periodic_jns(dist_tree, ['Base/zone'], opposite_jns, 4, sub_comm)
  assert len(I.getZones(dist_tree)) == 4+1

  # We still have only 2 periodic jns in the tree, and their translation value is updated
  assert len(PT.get_nodes_from_label(dist_tree, 'Periodic_t')) == 2
  for t in PT.iter_nodes_from_name(dist_tree, 'Translation'):
    assert np.allclose(np.abs(I.getVal(t)), [0, 0, 5])

  if write_output:
    out_dir = TU.create_pytest_output_dir(sub_comm)
    dist_tree_to_file(dist_tree, os.path.join(out_dir, 'duplicated.hdf'), sub_comm)
  

@mark_mpi_test([4])
def test_duplicate_360(sub_comm, write_output):
  mesh_file = os.path.join(TU.mesh_dir, 'U_ATB_45.yaml')
  dist_tree = file_to_dist_tree(mesh_file, sub_comm)

  # When working with an angular section of a cylindric object, we can easily duplicate the section
  # until the original object is reconstructed
  first_side_jn  = ['Base/bump_45/ZoneGridConnectivity/matchA']
  second_side_jn = ['Base/bump_45/ZoneGridConnectivity/matchB']
  opposite_jns = [first_side_jn, second_side_jn]

  assert len(I.getZones(dist_tree)) == 1
  DUP.duplicate_from_rotation_jns_to_360(dist_tree, ['Base/bump_45'], opposite_jns, sub_comm)
  assert len(I.getZones(dist_tree)) == 45

  # There is no more periodic joins in the tree
  for gc in PT.iter_nodes_from_label(dist_tree, 'GridConnectivity_t'):
    assert PT.request_node_from_label(gc, 'Periodic_t') is None

  if write_output:
    out_dir = TU.create_pytest_output_dir(sub_comm)
    dist_tree_to_file(dist_tree, os.path.join(out_dir, '360.hdf'), sub_comm)

