import pytest
import pytest_parallel

import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.utils import par_utils
from maia.utils.test_utils import mesh_dir

from maia.algo.dist import agglomeration as AGL
from maia.algo.dist import multigrid as MG


def make_bcs_vertex_located(tree, comm):
  for bc in PT.get_nodes_from_label(tree, 'BC_t'):
    loc = PT.Subset.GridLocation(bc)
    pr = PT.get_np_value(PT.find_child_from_name(bc, 'PointRange'))
    if loc != 'Vertex':
      axe = PT.Subset.normal_axis(bc)
      mask = np.ones(3, bool)
      mask[axe] = False
      pr[mask, 1] += 1
      if loc == 'CellCenter' and pr[axe, 0] != 1:
        pr[~mask, :] += 1
      size = PT.Subset.n_elem(bc)
      PT.update_child(bc, 'GridLocation', value='Vertex')
      MT.new_Distribution({'Index': par_utils.uniform_distribution(size, comm)}, bc)

@pytest_parallel.mark.parallel(2)
def test_mg_s_to_ngon(comm):
  tree = maia.io.file_to_dist_tree(mesh_dir / 'S_twoblocks.yaml', comm)
  make_bcs_vertex_located(tree, comm)

  maia.algo.dist.agglomerate_cells(tree, 1, comm)

  MG.convert_s_to_ngon(tree, comm)
  
  fine   = AGL.single_level_tree(tree, 0)
  coarse = AGL.single_level_tree(tree, 1)

  # Check sols
  for zone in PT.get_all_Zone_t(coarse):
    offset = MT.distribution_value(zone, 'Cell')[0] + 1 
    PT.new_FlowSolution('CurIdx', loc='CellCenter', fields={'Idx' : np.arange(MT.Zone.dn_cell(zone)) + offset}, parent=zone)
  maia.algo.interpolate(coarse, fine, comm, ['CurIdx'], 'CellCenter')

  for zone in PT.get_all_Zone_t(fine):
    assert (PT.get_np_value(PT.find_node_from_path(zone, 'MultiGridCellInfo/CoarseIdx')) == \
            PT.get_np_value(PT.find_node_from_path(zone, 'CurIdx/Idx'))).all()

  large_bc = PT.find_node_from_name(PT.find_node_from_name(fine, 'Large'), 'Right1')
  small_bc = PT.find_node_from_name(PT.find_node_from_name(fine, 'Small'), 'Front')
  if comm.rank == 0:
    expt_large = [9, 9, 9, 9, 45, 45]
    expt_small = [4, 4, 8, 8, 12, 12, 16, 16, 4, 4, 8, 8, 12, 12, 16, 16]
  else:
    expt_large = [45, 45, 81, 81, 81, 81]
    expt_small = [20, 20, 24, 24, 28, 28, 32, 32, 20, 20, 24, 24, 28, 28, 32, 32]
  assert (PT.get_np_value(PT.find_node_from_name(large_bc, 'CoarseIdx')) == expt_large).all()
  assert (PT.get_np_value(PT.find_node_from_name(small_bc, 'CoarseIdx')) == expt_small).all()
  

@pytest_parallel.mark.parallel(3)
def test_mg_merge(comm):
  tree = maia.io.file_to_dist_tree(mesh_dir / 'S_twoblocks.yaml', comm)
  make_bcs_vertex_located(tree, comm)

  maia.algo.dist.agglomerate_cells(tree, 1, comm)

  MG.convert_s_to_ngon(tree, comm)
  MG.merge_connected_zones(tree, comm)
  
  fine   = AGL.single_level_tree(tree, 0)
  coarse = AGL.single_level_tree(tree, 1)

  # Check sols
  for zone in PT.get_all_Zone_t(coarse):
    offset = MT.distribution_value(zone, 'Cell')[0] + 1 
    PT.new_FlowSolution('CurIdx', loc='CellCenter', fields={'Idx' : np.arange(MT.Zone.dn_cell(zone)) + offset}, parent=zone)
  maia.algo.interpolate(coarse, fine, comm, ['CurIdx'], 'CellCenter')

  for zone in PT.get_all_Zone_t(fine):
    assert (PT.get_np_value(PT.find_node_from_path(zone, 'MultiGridCellInfo/CoarseIdx')) == \
            PT.get_np_value(PT.find_node_from_path(zone, 'CurIdx/Idx'))).all()

  bc = PT.find_node_from_name(fine, 'Front')
  if comm.rank == 0:
    expt = [109, 109, 110, 110, 111, 111, 112, 112, 113, 113, 114, 114, 115, 115, 116,
            116, 109, 109, 110, 110, 111, 111, 112, 112, 113, 113, 114, 114, 115, 115,
            116, 116, 149, 149, 150, 150, 151, 151, 152, 152, 153, 153, 154]
  elif comm.rank == 1:
    expt = [154, 155, 155, 156, 156, 149, 149, 150, 150, 151, 151, 152, 152, 153, 153,
            154, 154, 155, 155, 156, 156, 189, 189, 190, 190, 191, 191, 192, 192, 193,
            193, 194, 194, 195, 195, 196, 196, 189, 189, 190, 190, 191, 191]
  else:
    expt = [192, 192, 193, 193, 194, 194, 195, 195, 196, 196, 360, 360, 364, 364, 368,
            368, 372, 372, 360, 360, 364, 364, 368, 368, 372, 372, 376, 376, 380, 380,
            384, 384, 388, 388, 376, 376, 380, 380, 384, 384, 388, 388]
  
  assert (PT.get_np_value(PT.find_node_from_name(bc, 'CoarseIdx')) == expt).all()
  
