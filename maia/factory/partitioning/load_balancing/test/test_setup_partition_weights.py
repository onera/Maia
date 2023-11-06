import pytest_parallel
import numpy as np
from mpi4py import MPI

import maia.pytree as PT
from maia.pytree.yaml import parse_yaml_cgns
from maia.factory.partitioning.load_balancing import setup_partition_weights

@pytest_parallel.mark.parallel(3)
class Test_npart_per_zone_3p:
  yt = """
Base0 CGNSBase_t [3,3]:
  ZoneU1 Zone_t [[1331,1000,0]]:
    ZoneType ZoneType_t "Unstructured":
  ZoneU2 Zone_t [[216,125,0]]:
    ZoneType ZoneType_t "Unstructured":
  ZoneS Zone_t [[21,20,0],[21,20,0],[2,1,0]]:
    ZoneType ZoneType_t "Structured":
"""
  dist_tree = parse_yaml_cgns.to_cgns_tree(yt)
  def test_one_part(self, comm):
    zone_to_weights = setup_partition_weights.npart_per_zone(self.dist_tree, comm)
    for zone in PT.get_all_Zone_t(self.dist_tree):
      assert 'Base0/'+PT.get_name(zone) in zone_to_weights
    for zone, weights in zone_to_weights.items():
      assert len(weights) == 1
      assert abs(weights[0] - 1./3.) < 1E-2 #Bad precision due to remainder

  def test_multiple_part(self, comm):
    n_part = 2 if comm.Get_rank() == 1 else 1
    zone_to_weights = setup_partition_weights.npart_per_zone(self.dist_tree, comm, n_part)
    for zone, weights in zone_to_weights.items():
      assert len(weights) == n_part
      for weight in weights:
        assert abs(weight - 0.25) < 1E-2 #Bad precision due to remainder

@pytest_parallel.mark.parallel(2)
class Test_balance_multizone_tree_2p:
  yt = """
Base0 CGNSBase_t [3,3]:
  ZoneU1 Zone_t [[1331,1000,0]]:
    ZoneType ZoneType_t "Unstructured":
  ZoneU2 Zone_t [[216,125,0]]:
    ZoneType ZoneType_t "Unstructured":
  ZoneS Zone_t [[21,20,0],[21,20,0],[2,1,0]]:
    ZoneType ZoneType_t "Structured":
"""
  dist_tree = parse_yaml_cgns.to_cgns_tree(yt)

  def test_uniform(self, comm):
    zone_to_weights = setup_partition_weights.balance_multizone_tree(self.dist_tree,
        comm, only_uniform=True)
    if comm.Get_rank() == 0:
      assert zone_to_weights['Base0/ZoneU1'] == [1.0]
      assert zone_to_weights['Base0/ZoneU2'] == [1.0]
      assert 'Base0/ZoneS' not in zone_to_weights
    elif comm.Get_rank() == 1:
      assert 'Base0/ZoneU1' not in zone_to_weights
      assert 'Base0/ZoneU2' not in zone_to_weights
      assert zone_to_weights['Base0/ZoneS']  == [1.0]

  def test_non_uniform(self, comm):
    zone_to_weights = setup_partition_weights.balance_multizone_tree(self.dist_tree, comm)
    if comm.Get_rank() == 0:
      assert zone_to_weights['Base0/ZoneU1'] == [.238]
      assert zone_to_weights['Base0/ZoneU2'] == [1.0]
      assert zone_to_weights['Base0/ZoneS']  == [1.0]
    elif comm.Get_rank() == 1:
      assert zone_to_weights['Base0/ZoneU1'] == [.762]
      assert 'Base0/ZoneU2' not in zone_to_weights
      assert 'Base0/ZoneS'  not in zone_to_weights

@pytest_parallel.mark.parallel([2,4])
def test_compute_nosplit_weights(comm):
  yt = """
Base0 CGNSBase_t [3,3]:
  ZoneU1 Zone_t [[1331,1000,0]]:
    ZoneType ZoneType_t "Unstructured":
  ZoneU2 Zone_t [[216,125,0]]:
    ZoneType ZoneType_t "Unstructured":
  ZoneS Zone_t [[21,20,0],[21,20,0],[2,1,0]]:
    ZoneType ZoneType_t "Structured":
"""
  dist_tree = parse_yaml_cgns.to_cgns_tree(yt)
  zone_to_weights = setup_partition_weights.compute_nosplit_weights(dist_tree, comm)
  if comm.Get_size() == 2:
    if comm.Get_rank() == 0:
      assert len(zone_to_weights) == 2
      assert zone_to_weights['Base0/ZoneU2'] == [1.0]
      assert zone_to_weights['Base0/ZoneS']  == [1.0]
    if comm.Get_rank() == 1:
      assert len(zone_to_weights) == 1
      assert zone_to_weights['Base0/ZoneU1']  == [1.0]
  assert comm.allreduce(len(zone_to_weights), MPI.SUM) == 3
