from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np
from mpi4py import MPI

import Converter.Internal as I

from maia.utils import parse_yaml_cgns
from maia.partitioning.load_balancing import setup_partition_weights

@mark_mpi_test(3)
class Test_npart_per_zone_3p:
  yt = """
Base0 CGNSBase_t [3,3]:
  ZoneU1 Zone_t [[1331,1000,0]]:
  ZoneU2 Zone_t [[216,125,0]]:
  ZoneS Zone_t [[21,20,0],[21,20,0],[2,1,0]]:
"""
  dist_tree = parse_yaml_cgns.to_cgns_tree(yt)
  def test_one_part(self, sub_comm):
    zone_to_weights = setup_partition_weights.npart_per_zone(self.dist_tree, sub_comm)
    for zone in I.getZones(self.dist_tree):
      assert I.getName(zone) in zone_to_weights
    for zone, weights in zone_to_weights.items():
      assert len(weights) == 1
      assert abs(weights[0] - 1./3.) < 1E-2 #Bad precision due to remainder

  def test_multiple_part(self, sub_comm):
    n_part = 2 if sub_comm.Get_rank() == 1 else 1
    zone_to_weights = setup_partition_weights.npart_per_zone(self.dist_tree, sub_comm, n_part)
    for zone, weights in zone_to_weights.items():
      assert len(weights) == n_part
      for weight in weights:
        assert abs(weight - 0.25) < 1E-2 #Bad precision due to remainder

@mark_mpi_test(2)
class Test_balance_multizone_tree_2p:
  yt = """
Base0 CGNSBase_t [3,3]:
  ZoneU1 Zone_t [[1331,1000,0]]:
  ZoneU2 Zone_t [[216,125,0]]:
  ZoneS Zone_t [[21,20,0],[21,20,0],[2,1,0]]:
"""
  dist_tree = parse_yaml_cgns.to_cgns_tree(yt)

  def test_uniform(self, sub_comm):
    zone_to_weights = setup_partition_weights.balance_multizone_tree(self.dist_tree,
        sub_comm, only_uniform=True)
    if sub_comm.Get_rank() == 0:
      assert zone_to_weights['ZoneU1'] == [1.0]
      assert zone_to_weights['ZoneU2'] == [1.0]
      assert zone_to_weights['ZoneS']  == []
    elif sub_comm.Get_rank() == 1:
      assert zone_to_weights['ZoneU1'] == []
      assert zone_to_weights['ZoneU2'] == []
      assert zone_to_weights['ZoneS']  == [1.0]

  def test_non_uniform(self, sub_comm):
    zone_to_weights = setup_partition_weights.balance_multizone_tree(self.dist_tree, sub_comm)
    if sub_comm.Get_rank() == 0:
      assert zone_to_weights['ZoneU1'] == [.238]
      assert zone_to_weights['ZoneU2'] == [1.0]
      assert zone_to_weights['ZoneS']  == [1.0]
    elif sub_comm.Get_rank() == 1:
      assert zone_to_weights['ZoneU1'] == [.762]
      assert zone_to_weights['ZoneU2'] == []
      assert zone_to_weights['ZoneS']  == []

