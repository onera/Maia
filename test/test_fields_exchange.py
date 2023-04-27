import pytest
from   pytest_mpi_check._decorator import mark_mpi_test
import os
import numpy as np

import maia.pytree        as PT

from maia import io       as MIO
from maia import factory  as MF
from maia import transfer as MT
from maia import algo     as MA
from maia import utils    as MU

from maia.transfer import protocols as EP


def _load_dist_tree(sub_comm):
  # Load the distributed tree
  mesh_file = os.path.join(MU.test_utils.mesh_dir, 'U_ATB_45.yaml')
  dist_tree = MIO.file_to_dist_tree(mesh_file, sub_comm)
  return dist_tree

def _create_dist_sol(dist_tree, sub_comm):
  # Create artificial fields on the distributed zone
  dist_zone = PT.get_all_Zone_t(dist_tree)[0] #This mesh is single zone
  fs = PT.new_FlowSolution('FlowSolution', loc='CellCenter', parent=dist_zone)
  cell_distri = PT.maia.getDistribution(dist_zone, 'Cell')[1]
  n_cell_dist = cell_distri[1] - cell_distri[0]
  PT.new_DataArray('RankId', sub_comm.Get_rank() * np.ones(n_cell_dist), parent=fs)
  PT.new_DataArray('CellId', np.arange(cell_distri[0], cell_distri[1]) + 1, parent=fs)
  PT.new_DataArray('CstField', np.ones(n_cell_dist), parent=fs)

def _create_dist_dataset(dist_tree, sub_comm):
  bc_amont = PT.get_node_from_name_and_label(dist_tree, 'amont', 'BC_t')
  bc_aval  = PT.get_node_from_name_and_label(dist_tree, 'aval',  'BC_t')

  for i, bc in enumerate([bc_amont, bc_aval]):
    patch_distri = PT.maia.getDistribution(bc, "Index")[1]
    patch_size   = PT.Subset.n_elem(bc) #This is local
    bcds   = PT.new_child(bc, 'BCDataSet', 'BCDataSet_t')
    bcdata = PT.new_child(bcds, 'BCData', 'BCData_t')
    PT.new_DataArray('BCId', (i+1) * np.ones(patch_size), parent=bcdata)
    PT.new_DataArray('FaceId', np.arange(patch_distri[0], patch_distri[1])+1, parent=bcdata)

def _split(dist_tree, sub_comm):
  zone_to_parts = MF.partitioning.compute_regular_weights(dist_tree, sub_comm, 2)
  part_tree = MF.partition_dist_tree(dist_tree, sub_comm, zone_to_parts=zone_to_parts)
  return part_tree


@mark_mpi_test([3])
class Test_fields_exchange:

  # Shared code for each case, who create the configuration
  def get_trees(self, sub_comm):
    dist_tree = _load_dist_tree(sub_comm)   #Load tree
    _create_dist_sol(dist_tree, sub_comm)   # Create artificial fields on the distributed zone
    part_tree = _split(dist_tree, sub_comm) # Split to get the partitioned tree
  
    # For now, we have no solution on the partitioned tree : 
    assert PT.get_node_from_label(part_tree, 'FlowSolution_t') is None
    return dist_tree, part_tree


  def test_zone_level(self, sub_comm):
    dist_tree, part_tree = self.get_trees(sub_comm)
    # The lowest level api to exchange field works at zone level
    dist_zone  = PT.get_all_Zone_t(dist_tree)[0]
    part_zones = PT.get_all_Zone_t(part_tree)
    MT.dist_to_part.data_exchange.dist_sol_to_part_sol(dist_zone, part_zones, sub_comm)
    # We retrieve our fields in each partition:
    for part_zone in part_zones:
      assert PT.get_node_from_path(part_zone, 'FlowSolution/RankId')   is not None
      assert PT.get_node_from_path(part_zone, 'FlowSolution/CellId')   is not None
      assert PT.get_node_from_path(part_zone, 'FlowSolution/CstField') is not None

    # Modify fields on partitions
    for part_zone in part_zones:
      for field in PT.iter_nodes_from_predicates(part_zone, 'FlowSolution_t/DataArray_t'):
        field[1] = 2*field[1]

    #We have the opposite function to send back data to the distributed zone :
    bck_zone = PT.deep_copy(dist_zone)
    MT.part_to_dist.data_exchange.part_sol_to_dist_sol(dist_zone, part_zones, sub_comm)
    for field in PT.iter_nodes_from_predicates(dist_zone, 'FlowSolution_t/DataArray_t'):
      assert np.allclose(PT.get_node_from_name(bck_zone, PT.get_name(field))[1] * 2, field[1])

  def test_created_sol(self, sub_comm):
    dist_tree, part_tree = self.get_trees(sub_comm)
    dist_zone  = PT.get_all_Zone_t(dist_tree)[0]
    part_zones = PT.get_all_Zone_t(part_tree)
    MT.dist_to_part.data_exchange.dist_sol_to_part_sol(dist_zone, part_zones, sub_comm)

    # Note that if a FlowSolution or some fields are created on the partitioned zones, they will be transfered to the
    # distributed zone as well
    for part_zone in part_zones:
      part_fs = PT.get_child_from_name(part_zone, 'FlowSolution')
      PT.new_DataArray('PartRankId', sub_comm.Get_rank() * np.ones(PT.Zone.n_cell(part_zone)), parent=part_fs)
      part_fs_new = PT.new_FlowSolution('CreatedFS', loc='Vertex', parent=part_zone)
      PT.new_DataArray('PartRankId', sub_comm.Get_rank() * np.ones(PT.Zone.n_vtx(part_zone)), parent=part_fs_new)

    MT.part_to_dist.data_exchange.part_sol_to_dist_sol(dist_zone, part_zones, sub_comm)
    assert PT.get_node_from_path(dist_zone, 'FlowSolution/RankId')     is not None
    assert PT.get_node_from_path(dist_zone, 'FlowSolution/PartRankId') is not None
    assert PT.get_node_from_path(dist_zone, 'CreatedFS/PartRankId')    is not None

  def test_zone_level_with_filters(self, sub_comm):
    # Each low level function accepts an include or exclude argument, allowing a smoother control
    # of field to exchange :
    dist_tree, part_tree = self.get_trees(sub_comm)
    dist_zone  = PT.get_all_Zone_t(dist_tree)[0]
    part_zones = PT.get_all_Zone_t(part_tree)

    MT.dist_to_part.data_exchange.dist_sol_to_part_sol(dist_zone, part_zones, sub_comm, exclude=['FlowSolution/CellId'])
    for part_zone in part_zones:
      assert PT.get_node_from_path(part_zone, 'FlowSolution/RankId')   is not None
      assert PT.get_node_from_path(part_zone, 'FlowSolution/CellId')   is None
      assert PT.get_node_from_path(part_zone, 'FlowSolution/CstField') is not None

    # Modify fields on partitions (on part_zones we have RankId and CstField)
    for part_zone in part_zones:
      for field in PT.iter_nodes_from_predicates(part_zone, 'FlowSolution_t/DataArray_t'):
        field[1] = 2*field[1]

    # Wildcard are also accepted
    bck_zone = PT.deep_copy(dist_zone)
    MT.part_to_dist.data_exchange.part_sol_to_dist_sol(dist_zone, part_zones, sub_comm, include=['FlowSolution/*Id'])
    assert np.allclose(PT.get_node_from_name(bck_zone, "RankId")[1] * 2, PT.get_node_from_name(dist_zone, "RankId")[1]) #Only this one had a way-and-back transfert
    assert np.allclose(PT.get_node_from_name(bck_zone, "CellId")[1]    , PT.get_node_from_name(dist_zone, "CellId")[1])
    assert np.allclose(PT.get_node_from_name(bck_zone, "CstField")[1]  , PT.get_node_from_name(dist_zone, "CstField")[1])

  def test_tree_level(self, sub_comm):
    dist_tree, part_tree = self.get_trees(sub_comm)
    # We also provide a tree-level api who loop over all the zones to exchange data of all kind
    MT.dist_to_part.dist_tree_to_part_tree_all(dist_tree, part_tree, sub_comm)
    for part_zone in PT.iter_all_Zone_t(part_tree):
      assert PT.get_node_from_path(part_zone, 'FlowSolution/RankId')   is not None
      assert PT.get_node_from_path(part_zone, 'FlowSolution/CellId')   is not None
      assert PT.get_node_from_path(part_zone, 'FlowSolution/CstField') is not None
      
  def test_zone_level_and_partial_reduce_func(self, sub_comm):
    dist_tree, part_tree = self.get_trees(sub_comm)
    dist_zone  = PT.get_all_Zone_t(dist_tree)[0]
    part_zones = PT.get_all_Zone_t(part_tree)
    MT.dist_to_part.data_exchange.dist_sol_to_part_sol(dist_zone, part_zones, sub_comm)

    # Note that if a FlowSolution or some fields are created on the partitioned zones, they will be transfered to the
    # distributed zone as well
    for part_zone in part_zones:
      part_fs = PT.get_child_from_name(part_zone, 'FlowSolution')
      PT.new_DataArray('PartRankId', sub_comm.Get_rank() * np.ones(PT.Zone.n_cell(part_zone)), parent=part_fs)
      part_fs_new = PT.new_FlowSolution('FlowSolutionVtx', loc='Vertex', parent=part_zone)
      PT.new_DataArray('PartRankId', sub_comm.Get_rank() * np.ones(PT.Zone.n_vtx(part_zone)), parent=part_fs_new)

    MT.part_to_dist.data_exchange.part_sol_to_dist_sol(dist_zone, part_zones, sub_comm)
    assert PT.get_node_from_path(dist_zone, 'FlowSolution/RankId')        is not None
    assert PT.get_node_from_path(dist_zone, 'FlowSolution/PartRankId')    is not None
    assert PT.get_node_from_path(dist_zone, 'FlowSolutionVtx/PartRankId') is not None
    
    bck_zone = PT.deep_copy(dist_zone)
    MT.part_to_dist.data_exchange.part_sol_to_dist_sol(dist_zone, part_zones, sub_comm, include=['FlowSolutionVtx/PartRankId'],reduce_func=EP.reduce_sum)
    bck_fs_cc_rank_id  = PT.get_value(PT.get_node_from_path(bck_zone,  'FlowSolution/PartRankId'))
    fs_cc_rank_id      = PT.get_value(PT.get_node_from_path(dist_zone, 'FlowSolution/PartRankId'))
    assert np.all(bck_fs_cc_rank_id.shape == fs_cc_rank_id.shape)
    assert np.all(fs_cc_rank_id == bck_fs_cc_rank_id)
    bck_fs_vtx_rank_id = PT.get_value(PT.get_node_from_path(bck_zone,  'FlowSolutionVtx/PartRankId'))
    fs_vtx_rank_id     = PT.get_value(PT.get_node_from_path(dist_zone, 'FlowSolutionVtx/PartRankId'))
    assert np.all(bck_fs_vtx_rank_id.shape == fs_vtx_rank_id.shape)
    assert np.all(fs_vtx_rank_id >= bck_fs_vtx_rank_id)
    assert np.any(fs_vtx_rank_id > sub_comm.Get_size()*np.ones(len(fs_vtx_rank_id)))


@mark_mpi_test([2])
class Test_multiple_labels_exchange:

  # Shared code for each case, who create the configuration
  def get_trees(self, sub_comm):
    dist_tree = _load_dist_tree(sub_comm)     #Load tree
    _create_dist_sol(dist_tree, sub_comm)     # Create artificial fields on the distributed zone
    _create_dist_dataset(dist_tree, sub_comm) # Create artificial BCDataSet
    part_tree = _split(dist_tree, sub_comm)   # Split to get the partitioned tree
    return dist_tree, part_tree

  def _cleanup(self, part_tree):
    PT.rm_nodes_from_predicate(part_tree, lambda n : PT.get_label(n) in ['FlowSolution_t', 'BCDataSet_t'])

  def test_zone_level(self, sub_comm):
    dist_tree, part_tree = self.get_trees(sub_comm)
    dist_zone  = PT.get_all_Zone_t(dist_tree)[0]
    part_zones = PT.get_all_Zone_t(part_tree)
    # At tree level API, one can use the _all and _only versions to select only
    # some labels to exchange. Note that we can also filter the fields using the paths
    MT.dist_to_part.dist_zone_to_part_zones_only(dist_zone, part_zones, sub_comm, \
        include_dict = {'FlowSolution_t' : ['*/*Id']})
    for part_zone in part_zones:
      assert PT.get_node_from_path(part_zone, 'FlowSolution/RankId')   is not None
      assert PT.get_node_from_path(part_zone, 'FlowSolution/CstField') is None
      assert PT.get_node_from_label(part_zone, 'BCDataSet_t') is None
    self._cleanup(part_tree)

    MT.dist_to_part.dist_zone_to_part_zones_only(dist_zone, part_zones, sub_comm, \
        include_dict = {'FlowSolution_t' : ['FlowSolution/*Id'], 'BCDataSet_t' : ['amont/*/*/*', 'aval/*/*/FaceId']})
    for part_zone in part_zones:
      assert PT.get_node_from_path(part_zone, 'FlowSolution/RankId')   is not None
      assert PT.get_node_from_path(part_zone, 'FlowSolution/CstField') is None
      bc_amont = PT.get_node_from_name_and_label(part_zone, 'amont', 'BC_t')
      if bc_amont is not None:
        assert PT.get_node_from_name(bc_amont, 'BCId') is not None
        assert PT.get_node_from_name(bc_amont, 'FaceId') is not None
      bc_aval = PT.get_node_from_name_and_label(part_zone, 'aval', 'BC_t')
      if bc_aval is not None:
        assert PT.get_node_from_name(bc_aval, 'BCId') is None
        assert PT.get_node_from_name(bc_aval, 'FaceId') is not None
    self._cleanup(part_tree)

    MT.dist_to_part.dist_zone_to_part_zones_all(dist_zone, part_zones, sub_comm, \
        exclude_dict = {'FlowSolution_t' : ['*']}) #Everything, excepted FS
    for part_zone in part_zones:
      assert PT.get_node_from_path(part_zone, 'FlowSolution') is None
      bc_amont = PT.get_node_from_name_and_label(part_zone, 'amont', 'BC_t')
      bc_aval  = PT.get_node_from_name_and_label(part_zone, 'aval',  'BC_t')
      for bc in [bc_amont, bc_aval]:
        if bc is not None: #BC can be absent from partition
          assert PT.get_node_from_name(bc, 'BCId') is not None
          assert PT.get_node_from_name(bc, 'FaceId') is not None
    self._cleanup(part_tree)

  def test_tree_level(self, sub_comm):
    dist_tree, part_tree = self.get_trees(sub_comm)
    # At tree level API, one can select only some labels to exchange
    MT.dist_to_part.dist_tree_to_part_tree_only_labels(dist_tree, part_tree, ['BCDataSet_t'], sub_comm)
    assert PT.get_node_from_label(part_tree, 'FlowSolution_t') is None
    for part in PT.get_all_Zone_t(part_tree):
      bc_amont = PT.get_node_from_name_and_label(part, 'amont', 'BC_t')
      bc_aval  = PT.get_node_from_name_and_label(part, 'aval',  'BC_t')
      for bc in [bc_amont, bc_aval]:
        if bc is not None: #BC can be absent from partition
          assert PT.get_node_from_name(bc, 'BCId') is not None
          assert PT.get_node_from_name(bc, 'FaceId') is not None
