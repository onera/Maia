import pytest
import pytest_parallel
from mpi4py import MPI

import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.algo.dist import multigrid as MGA
from maia.algo.dist import agglomeration as AGL
from maia.factory import multigrid as MGF

from maia.algo.dist.test.test_multigrid_algo import make_bcs_vertex_located


@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("dim", [2,3])
def test_mg_partitioning(dim, comm):

  n_lvl = 3
  coeff = 2**n_lvl
  dims = [coeff*8+1,coeff*4+1]
  if dim == 3:
    dims.append(coeff*7 + 1)

  tree = maia.factory.generate_dist_block(dims, "S", comm)

  maia.algo.dist.agglomerate_cells(tree, n_lvl, comm)

  MGA.convert_s_to_ngon(tree, comm)
  MGA.merge_connected_zones(tree, comm)
  ptree = MGF.partition_dist_tree(tree, comm)

  for i in range(n_lvl):
    fine   = PT.deep_copy(AGL.single_level_tree(ptree, i))
    coarse = PT.deep_copy(AGL.single_level_tree(ptree, i+1))

    # Check sols
    for zone in PT.get_all_Zone_t(coarse):
      PT.new_FlowSolution('CurIdx', loc='CellCenter', fields={'Idx' : np.arange(MT.Zone.pn_cell(zone))}, parent=zone)
    maia.algo.interpolate(coarse, fine, comm, ['CurIdx'], 'CellCenter')

    for zone in PT.get_all_Zone_t(fine):
      assert (PT.get_np_value(PT.find_node_from_path(zone, 'MultiGridCellInfo/CoarseLocalIdx')) == \
              PT.get_np_value(PT.find_node_from_path(zone, 'CurIdx/Idx'))).all()

      for bc in PT.get_nodes_from_label(coarse, 'BC_t'):
        bcds = PT.new_BCDataSet(parent=bc)
        PT.new_BCData('DirichletData', fields={'Idx' : np.arange(MT.Subset.pn_elem(bc))}, parent=bcds)
        fine_ext   = maia.algo.part.extract_part_from_bc_name(fine, PT.get_name(bc), MPI.COMM_SELF)
        coarse_ext = maia.algo.part.extract_part_from_bc_name(coarse, PT.get_name(bc), MPI.COMM_SELF)

        PT.set_name(PT.find_node_from_label(fine_ext, 'FlowSolution_t'), 'MGFaceInfo')
        maia.algo.interpolate(coarse_ext, fine_ext, MPI.COMM_SELF, [PT.get_name(bc)], 'CellCenter')
          
        assert (PT.get_np_value(PT.find_node_from_name(fine_ext, 'CoarseLocalIdx')) == \
                PT.get_np_value(PT.find_node_from_name(fine_ext, 'Idx'))).all()


@pytest_parallel.mark.parallel(2)
def test_mg_partitioning_opts(comm):

  n_lvl = 2
  coeff = 2**n_lvl
  dims = [coeff*8+1,coeff*4+1]

  tree = maia.factory.generate_dist_block(dims, "S", comm)

  maia.algo.dist.agglomerate_cells(tree, n_lvl, comm)
  MGA.convert_s_to_ngon(tree, comm)

  # Create dummy field
  for zone in PT.get_all_Zone_t(tree):
    PT.new_DiscreteData(loc='CellCenter', fields={'Dummy' : np.ones(MT.Zone.dn_cell(zone))}, parent=zone)

  # Compute part id on coarsest mesh
  coarse = AGL.single_level_tree(tree, 2)
  zone = PT.get_node_from_label(coarse, 'Zone_t')
  partid = np.zeros(MT.Zone.dn_cell(zone), np.int32)
  partid[-1] = 1

  ptree = MGF.partition_dist_tree(tree, comm, target_part=[partid], data_transfer=['DiscreteData_t'])

  for zone in PT.get_all_Zone_t(ptree):
    assert PT.get_node_from_path(zone, 'DiscreteData/Dummy') is not None

  if comm.rank == 1:
    for i in range(n_lvl):
      zone = PT.find_node_from_path(ptree, f'Base.LV{i}/zone.P1.N0')
      assert MT.Zone.pn_cell(zone) == 2 * 4**(n_lvl-i)
  