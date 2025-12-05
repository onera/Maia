import pytest
import pytest_parallel

import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.utils import par_utils
from maia.utils.test_utils import mesh_dir

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
def test_mgjc(comm):
  tree = maia.io.file_to_dist_tree(mesh_dir / 'S_twoblocks.yaml', comm)
  make_bcs_vertex_located(tree, comm)

  maia.algo.dist.agglomerate_cells(tree, 1, comm)

  MG.convert_s_to_ngon(tree, comm)
  MG.merge_connected_zones(tree, comm)
  ptree = MG.partition_dist_tree(tree, comm)


  # For checks
  for pzone in PT.get_all_Zone_t(ptree):
    mg = PT.get_child_from_name(pzone, 'MultiGridCellInfo')
    if mg is not None:
      PT.set_label(mg, 'FlowSolution_t')
      PT.new_DataArray('Id', np.arange(PT.Zone.n_cell(pzone)), parent=mg)
    else:
      mg = PT.new_FlowSolution('MultiGridCellInfo', loc='CellCenter', parent=pzone)
      PT.new_DataArray('Id', np.arange(PT.Zone.n_cell(pzone)), parent=mg)

    for array in PT.get_nodes_from_label(mg, 'DataArray_t'):
      array[1] = array[1].astype(float)

    for bc in PT.get_nodes_from_label(pzone, 'BC_t'):
      PT.set_value(bc, 'FamilySpecified')
      mg = PT.get_node_from_name(bc, 'MultiGridBCFaceInfo')
      if mg is None:
        mg = PT.new_BCDataSet('MultiGridBCFaceInfo', parent=bc)
        PT.new_BCData('DirichletData', {'Id' : np.arange(PT.Subset.n_elem(bc)).astype(float)}, parent=mg)
      else:
        for array in PT.get_nodes_from_label(mg, 'DataArray_t'):
          array[1] = array[1].astype(float)


  maia.io.part_tree_to_file(ptree, 'part.cgns', comm, True)