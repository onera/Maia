import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import maia.transfer.protocols as EP


@mark_mpi_test(2)
def test_block_to_part(sub_comm):
  dist_data = dict()
  expected_part_data = dict()
  if sub_comm.Get_rank() == 0:
    partial_distri = np.array([0, 5, 10])
    ln_to_gn_list = [np.array([2,4,6,10])]
    dist_data["field"] = np.array([1., 2., 3., 4., 5.])
    expected_part_data["field"] = [np.array([2., 4., 6., 1000.])]
  else:
    partial_distri = np.array([5, 10, 10])
    ln_to_gn_list = [np.array([9,7,5,3,1]),
                     np.array([8]),
                     np.array([1])]
    dist_data["field"] = np.array([6., 7., 8., 9., 1000.])
    expected_part_data["field"] = [np.array([9., 7., 5., 3., 1.]), np.array([8.]), np.array([1.])]

  part_data = EP.block_to_part(dist_data, partial_distri, ln_to_gn_list, sub_comm)
  assert len(part_data["field"]) == len(ln_to_gn_list)
  for i_part in range(len(ln_to_gn_list)):
    assert part_data["field"][i_part].dtype == np.float64
    assert (part_data["field"][i_part] == expected_part_data["field"][i_part]).all()

@mark_mpi_test(2)
def test_block_to_part_with_void(sub_comm):
  dist_data = dict()
  expected_part_data = dict()
  if sub_comm.Get_rank() == 0:
    partial_distri = np.array([0, 5, 10])
    ln_to_gn_list = [np.array([10,8])]
    dist_data["field"] = np.array([1., 2., 3., 4., 5.])
    expected_part_data["field"] = [np.array([1000., 8.])]
  else:
    partial_distri = np.array([5, 10, 10])
    ln_to_gn_list = list()
    dist_data["field"] = np.array([6., 7., 8., 9., 1000.])
    expected_part_data["field"] = list()

  part_data = EP.block_to_part(dist_data, partial_distri, ln_to_gn_list, sub_comm)
  assert len(part_data["field"]) == len(ln_to_gn_list)
  for i_part in range(len(ln_to_gn_list)):
    assert part_data["field"][i_part].dtype == np.float64
    assert (part_data["field"][i_part] == expected_part_data["field"][i_part]).all()

@mark_mpi_test(2)
def test_part_to_block(sub_comm):
  part_data = dict()
  expected_dist_data = dict()
  if sub_comm.Get_rank() == 0:
    partial_distri = np.array([0, 5, 10])
    ln_to_gn_list = [np.array([2,4,6,10])]
    part_data["field"] = [np.array([2., 4., 6., 1000.])]
    expected_dist_data["field"] = np.array([1., 2., 3., 4., 5.])
  else:
    partial_distri = np.array([5, 10, 10])
    ln_to_gn_list = [np.array([9,7,5,3,1]),
                     np.array([8]),
                     np.array([1])]
    part_data["field"] = [np.array([9., 7., 5., 3., 1.]), np.array([8.]), np.array([1.])]
    expected_dist_data["field"] = np.array([6., 7., 8., 9., 1000.])

  dist_data = EP.part_to_block(part_data, partial_distri, ln_to_gn_list, sub_comm)
  assert dist_data["field"].dtype == np.float64
  assert (dist_data["field"] == expected_dist_data["field"]).all()

