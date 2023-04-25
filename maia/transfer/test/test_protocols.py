import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import maia.transfer.protocols as EP

class Test_auto_expand_distri:
  
  @mark_mpi_test(3)
  def test_straightforward(self, sub_comm):
    if sub_comm.Get_rank() == 0:
      distri_partial = np.array([0, 10, 40])
    if sub_comm.Get_rank() == 1:
      distri_partial = np.array([10, 20, 40])
    if sub_comm.Get_rank() == 2:
      distri_partial = np.array([20,40,40])
    assert np.array_equal(EP.auto_expand_distri(distri_partial, sub_comm), \
        np.array([0,10,20,40]))

    distri_full = np.array([0,10,20,40])
    assert EP.auto_expand_distri(distri_full, sub_comm) is distri_full

  @mark_mpi_test(2)
  def test_corner_cases(self, sub_comm):
    distri_partial = np.array([0, 10, 40]) if sub_comm.Get_rank() == 0 else np.array([10, 40, 40])
    assert np.array_equal(EP.auto_expand_distri(distri_partial, sub_comm), \
        np.array([0,10,40]))
    distri_partial = np.array([0, 40, 40]) if sub_comm.Get_rank() == 0 else np.array([40, 40, 40])
    assert np.array_equal(EP.auto_expand_distri(distri_partial, sub_comm), \
        np.array([0,40,40]))
    distri_partial = np.array([0, 0, 40]) if sub_comm.Get_rank() == 0 else np.array([0, 40, 40])
    assert np.array_equal(EP.auto_expand_distri(distri_partial, sub_comm), \
        np.array([0,0,40]))
    distri_partial = np.array([0, 0, 0]) if sub_comm.Get_rank() == 0 else np.array([0, 0, 0])
    assert np.array_equal(EP.auto_expand_distri(distri_partial, sub_comm), \
        np.array([0,0,0]))
    # Already full
    for distri_full in [[0,10,40], [0,0,40], [0,40,40], [0,0,0]]:
      _distri_full = np.array(distri_full)
      assert np.array_equal(EP.auto_expand_distri(_distri_full, sub_comm), _distri_full)

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

@mark_mpi_test(2)
def test_part_to_block_with_reduce_sum(sub_comm):
  part_data = dict()
  expected_dist_data = dict()
  if sub_comm.Get_rank() == 0:
    partial_distri = np.array([0, 5, 9])
    ln_to_gn_list = [np.array([2,4,6,9])]
    part_data["field"] = [np.array([2., 4., 6., 1000.])]
    expected_dist_data["field"] = np.array([1.+1., 2., 3., 4., 5.])
  else:
    partial_distri = np.array([5, 9, 9])
    ln_to_gn_list = [np.array([9,7,5,3,1]),
                     np.array([8]),
                     np.array([1])]
    part_data["field"] = [np.array([9., 7., 5., 3., 1.]), np.array([8.]), np.array([1.])]
    expected_dist_data["field"] = np.array([6., 7., 8., 9.+1000.])

  dist_data = EP.part_to_block(part_data, partial_distri, ln_to_gn_list, sub_comm,\
                               reduce_func=EP.reduce_sum)
  assert dist_data["field"].dtype == np.float64
  assert (dist_data["field"] == expected_dist_data["field"]).all()

@mark_mpi_test(2)
def test_part_to_block_with_reduce_min(sub_comm):
  part_data = dict()
  expected_dist_data = dict()
  if sub_comm.Get_rank() == 0:
    partial_distri = np.array([0, 5, 9])
    ln_to_gn_list = [np.array([2,4,6,9])]
    part_data["field"] = [np.array([2., 4., 6., 1000.])]
    expected_dist_data["field"] = np.array([min(1.,1.), 2., 3., 4., 5.])
  else:
    partial_distri = np.array([5, 9, 9])
    ln_to_gn_list = [np.array([9,7,5,3,1]),
                     np.array([8]),
                     np.array([1])]
    part_data["field"] = [np.array([9., 7., 5., 3., 1.]), np.array([8.]), np.array([1.])]
    expected_dist_data["field"] = np.array([6., 7., 8., min(9.,1000.)])

  dist_data = EP.part_to_block(part_data, partial_distri, ln_to_gn_list, sub_comm,\
                               reduce_func=EP.reduce_min)
  assert dist_data["field"].dtype == np.float64
  assert (dist_data["field"] == expected_dist_data["field"]).all()

@mark_mpi_test(2)
def test_part_to_block_with_reduce_max(sub_comm):
  part_data = dict()
  expected_dist_data = dict()
  if sub_comm.Get_rank() == 0:
    partial_distri = np.array([0, 5, 9])
    ln_to_gn_list = [np.array([2,4,6,9])]
    part_data["field"] = [np.array([2., 4., 6., 1000.])]
    expected_dist_data["field"] = np.array([max(1.,1.), 2., 3., 4., 5.])
  else:
    partial_distri = np.array([5, 9, 9])
    ln_to_gn_list = [np.array([9,7,5,3,1]),
                     np.array([8]),
                     np.array([1])]
    part_data["field"] = [np.array([9., 7., 5., 3., 1.]), np.array([8.]), np.array([1.])]
    expected_dist_data["field"] = np.array([6., 7., 8., max(9.,1000.)])

  dist_data = EP.part_to_block(part_data, partial_distri, ln_to_gn_list, sub_comm,\
                               reduce_func=EP.reduce_max)
  assert dist_data["field"].dtype == np.float64
  assert (dist_data["field"] == expected_dist_data["field"]).all()

@mark_mpi_test(2)
def test_part_to_block_with_reduce_mean(sub_comm):
  part_data = dict()
  expected_dist_data = dict()
  if sub_comm.Get_rank() == 0:
    partial_distri = np.array([0, 5, 9])
    ln_to_gn_list = [np.array([2,4,6,9])]
    part_data["field"] = [np.array([2., 4., 6., 1000.])]
    expected_dist_data["field"] = np.array([(1.+1.)/2., 2., 3., 4., 5.])
  else:
    partial_distri = np.array([5, 9, 9])
    ln_to_gn_list = [np.array([9,7,5,3,1]),
                     np.array([8]),
                     np.array([1])]
    part_data["field"] = [np.array([9., 7., 5., 3., 1.]), np.array([8.]), np.array([1.])]
    expected_dist_data["field"] = np.array([6., 7., 8., (9.+1000.)/2.])

  dist_data = EP.part_to_block(part_data, partial_distri, ln_to_gn_list, sub_comm,\
                               reduce_func=EP.reduce_mean)
  assert dist_data["field"].dtype == np.float64
  assert (dist_data["field"] == expected_dist_data["field"]).all()

