import pytest
import pytest_parallel
import os
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils import test_utils as TU

from maia.io       import file_to_dist_tree, dist_tree_to_file
from maia.factory  import generate_dist_block

from maia.algo      import transform_affine
from maia.algo.dist import duplicate as DUP

@pytest_parallel.mark.parallel([2])
@pytest.mark.parametrize("fields", [True, False])
def test_translate_cube(comm, fields, write_output):
  # Generate a disttree with one zone
  dist_tree = generate_dist_block(11, "Poly", comm, origin=[0., -.5, -.5])
  dist_zone = PT.get_all_Zone_t(dist_tree)[0]

  # Initialise some fields
  cell_distri = MT.getDistribution(dist_zone, 'Cell')[1]
  n_cell_loc =  cell_distri[1] - cell_distri[0]
  fs = PT.new_FlowSolution('FlowSolution', loc='CellCenter', parent=dist_zone)
  PT.new_DataArray('scalar', np.random.random(n_cell_loc), parent=fs)
  PT.new_DataArray('vectX', np.random.random(n_cell_loc), parent=fs)
  PT.new_DataArray('vectY', np.random.random(n_cell_loc), parent=fs)
  PT.new_DataArray('vectZ', np.random.random(n_cell_loc), parent=fs)

  # One can apply a rotation and/or a translation to the zone
  transformed_zone = PT.deep_copy(dist_zone)
  PT.set_name(transformed_zone, 'DuplicatedZone')
  transform_affine(transformed_zone, rotation_angle=np.array([0.,0.,np.pi]), apply_to_fields=fields)

  # Coordinates are moved :
  coords    = PT.Zone.coordinates(dist_zone)
  tr_coords = PT.Zone.coordinates(transformed_zone)
  assert np.allclose(-coords[0], tr_coords[0])
  assert np.allclose(-coords[1], tr_coords[1])
  assert np.allclose(coords[2], tr_coords[2])

  # If apply_to_fields is True, vectorial fields are moved as well :
  assert PT.is_same_node(PT.get_node_from_name(dist_zone, 'scalar'), PT.get_node_from_name(transformed_zone, 'scalar'))
  assert PT.is_same_node(PT.get_node_from_name(dist_zone, 'vectZ'), PT.get_node_from_name(transformed_zone, 'vectZ'))
  if fields:
    assert np.allclose(-PT.get_node_from_name(dist_zone, 'vectX')[1], PT.get_node_from_name(transformed_zone, 'vectX')[1])
    assert np.allclose(-PT.get_node_from_name(dist_zone, 'vectY')[1], PT.get_node_from_name(transformed_zone, 'vectY')[1])
  else:
    assert PT.is_same_node(PT.get_node_from_name(dist_zone, 'vectX'), PT.get_node_from_name(transformed_zone, 'vectX'))
    assert PT.is_same_node(PT.get_node_from_name(dist_zone, 'vectY'), PT.get_node_from_name(transformed_zone, 'vectY'))

  if write_output:
    dist_base = PT.get_all_CGNSBase_t(dist_tree)[0]
    PT.add_child(dist_base, transformed_zone)
    out_dir = TU.create_pytest_output_dir(comm)
    dist_tree_to_file(dist_tree, os.path.join(out_dir, 'duplicated.hdf'), comm)

@pytest_parallel.mark.parallel([1])
def test_duplicate_from_periodic(comm, write_output):
  dist_tree = generate_dist_block(11, "Poly", comm)
  # Lets create a periodic join for this cube
  dist_zone = PT.get_all_Zone_t(dist_tree)[0]
  bottom = PT.get_node_from_name(dist_zone, 'Zmin')
  top    = PT.get_node_from_name(dist_zone, 'Zmax')
  for bc in [bottom, top]:
    PT.rm_nodes_from_name(dist_zone, PT.get_name(bc))
    PT.set_label(bc, 'GridConnectivity_t')
    PT.set_value(bc, PT.get_name(dist_zone))
    PT.new_GridConnectivityType('Abutting1to1', bc)
  PT.new_IndexArray('PointListDonor', PT.get_node_from_name(top, 'PointList')[1].copy(), parent=bottom) #OK in sequential
  PT.new_IndexArray('PointListDonor', PT.get_node_from_name(bottom, 'PointList')[1].copy(), parent=top) #OK in sequential
  PT.new_GridConnectivityProperty(periodic={'translation': [0.,0.,1]},  parent=bottom)
  PT.new_GridConnectivityProperty(periodic={'translation': [0.,0.,-1]}, parent=top)
  PT.new_child(dist_zone, 'ZoneGridConnectivity', 'ZoneGridConnectivity_t', children=[bottom, top])

  # If a periodic (rotation and/or translation) JN is present in the mesh, the next function can duplicate
  # the zones and move it at the interface location, any number of time
  first_side_jn  = [f'Base/zone/ZoneGridConnectivity/{PT.get_name(bottom)}']
  second_side_jn = [f'Base/zone/ZoneGridConnectivity/{PT.get_name(top)}']
  opposite_jns = [first_side_jn, second_side_jn]

  assert len(PT.get_all_Zone_t(dist_tree)) == 1
  DUP.duplicate_from_periodic_jns(dist_tree, ['Base/zone'], opposite_jns, 4, comm)
  assert len(PT.get_all_Zone_t(dist_tree)) == 4+1

  # We still have only 2 periodic jns in the tree, and their translation value is updated
  assert len(PT.get_nodes_from_label(dist_tree, 'Periodic_t')) == 2
  for t in PT.iter_nodes_from_name(dist_tree, 'Translation'):
    assert np.allclose(np.abs(PT.get_value(t)), [0, 0, 5])

  if write_output:
    out_dir = TU.create_pytest_output_dir(comm)
    dist_tree_to_file(dist_tree, os.path.join(out_dir, 'duplicated.hdf'), comm)
  

@pytest_parallel.mark.parallel([4])
def test_duplicate_360(comm, write_output):
  mesh_file = os.path.join(TU.mesh_dir, 'U_ATB_45.yaml')
  dist_tree = file_to_dist_tree(mesh_file, comm)

  # When working with an angular section of a cylindric object, we can easily duplicate the section
  # until the original object is reconstructed
  first_side_jn  = ['Base/bump_45/ZoneGridConnectivity/matchA']
  second_side_jn = ['Base/bump_45/ZoneGridConnectivity/matchB']
  opposite_jns = [first_side_jn, second_side_jn]

  assert len(PT.get_all_Zone_t(dist_tree)) == 1
  DUP.duplicate_from_rotation_jns_to_360(dist_tree, ['Base/bump_45'], opposite_jns, comm)
  assert len(PT.get_all_Zone_t(dist_tree)) == 45

  # There is no more periodic joins in the tree
  for gc in PT.iter_nodes_from_label(dist_tree, 'GridConnectivity_t'):
    assert PT.get_node_from_label(gc, 'Periodic_t') is None

  if write_output:
    out_dir = TU.create_pytest_output_dir(comm)
    dist_tree_to_file(dist_tree, os.path.join(out_dir, '360.hdf'), comm)

