import pytest
import pytest_parallel
import numpy as np

import maia.transfer.protocols as EP

class Test_auto_expand_distri:
  
  @pytest_parallel.mark.parallel(3)
  def test_straightforward(self, comm):
    if comm.Get_rank() == 0:
      distri_partial = np.array([0, 10, 40])
    if comm.Get_rank() == 1:
      distri_partial = np.array([10, 20, 40])
    if comm.Get_rank() == 2:
      distri_partial = np.array([20,40,40])
    assert np.array_equal(EP.auto_expand_distri(distri_partial, comm), \
        np.array([0,10,20,40]))

    distri_full = np.array([0,10,20,40])
    assert EP.auto_expand_distri(distri_full, comm) is distri_full

  @pytest_parallel.mark.parallel(2)
  def test_corner_cases(self, comm):
    distri_partial = np.array([0, 10, 40]) if comm.Get_rank() == 0 else np.array([10, 40, 40])
    assert np.array_equal(EP.auto_expand_distri(distri_partial, comm), \
        np.array([0,10,40]))
    distri_partial = np.array([0, 40, 40]) if comm.Get_rank() == 0 else np.array([40, 40, 40])
    assert np.array_equal(EP.auto_expand_distri(distri_partial, comm), \
        np.array([0,40,40]))
    distri_partial = np.array([0, 0, 40]) if comm.Get_rank() == 0 else np.array([0, 40, 40])
    assert np.array_equal(EP.auto_expand_distri(distri_partial, comm), \
        np.array([0,0,40]))
    distri_partial = np.array([0, 0, 0]) if comm.Get_rank() == 0 else np.array([0, 0, 0])
    assert np.array_equal(EP.auto_expand_distri(distri_partial, comm), \
        np.array([0,0,0]))
    # Already full
    for distri_full in [[0,10,40], [0,0,40], [0,40,40], [0,0,0]]:
      _distri_full = np.array(distri_full)
      assert np.array_equal(EP.auto_expand_distri(_distri_full, comm), _distri_full)

@pytest_parallel.mark.parallel(2)
def test_block_to_part(comm):
  dist_data = dict()
  expected_part_data = dict()
  if comm.Get_rank() == 0:
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

  part_data = EP.block_to_part(dist_data, partial_distri, ln_to_gn_list, comm)
  assert len(part_data["field"]) == len(ln_to_gn_list)
  for i_part in range(len(ln_to_gn_list)):
    assert part_data["field"][i_part].dtype == np.float64
    assert (part_data["field"][i_part] == expected_part_data["field"][i_part]).all()

@pytest_parallel.mark.parallel(2)
def test_block_to_part_with_void(comm):
  dist_data = dict()
  expected_part_data = dict()
  if comm.Get_rank() == 0:
    partial_distri = np.array([0, 5, 10])
    ln_to_gn_list = [np.array([10,8])]
    dist_data["field"] = np.array([1., 2., 3., 4., 5.])
    expected_part_data["field"] = [np.array([1000., 8.])]
  else:
    partial_distri = np.array([5, 10, 10])
    ln_to_gn_list = list()
    dist_data["field"] = np.array([6., 7., 8., 9., 1000.])
    expected_part_data["field"] = list()

  part_data = EP.block_to_part(dist_data, partial_distri, ln_to_gn_list, comm)
  assert len(part_data["field"]) == len(ln_to_gn_list)
  for i_part in range(len(ln_to_gn_list)):
    assert part_data["field"][i_part].dtype == np.float64
    assert (part_data["field"][i_part] == expected_part_data["field"][i_part]).all()

@pytest_parallel.mark.parallel(2)
def test_part_to_block(comm):
  part_data = dict()
  expected_dist_data = dict()
  if comm.Get_rank() == 0:
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

  dist_data = EP.part_to_block(part_data, partial_distri, ln_to_gn_list, comm)
  assert dist_data["field"].dtype == np.float64
  assert (dist_data["field"] == expected_dist_data["field"]).all()

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("reduce_func", ["sum", "min", "max", "mean"])
def test_part_to_block_with_reduce(reduce_func, comm):
  part_data = dict()
  expected_dist_data = dict()
  if comm.Get_rank() == 0:
    partial_distri = np.array([0, 5, 9])
    ln_to_gn_list = [np.array([2,4,6,9])]
    part_data["field"] = [np.array([2., 4., 6., 1000.])]
    if reduce_func == "sum":
      expected_dist_data["field"] = np.array([1.+1., 2., 3., 4., 5.])
    elif reduce_func == "min":
      expected_dist_data["field"] = np.array([min(1.,1.), 2., 3., 4., 5.])
    elif reduce_func == "max":
      expected_dist_data["field"] = np.array([max(1.,1.), 2., 3., 4., 5.])
    elif reduce_func == "mean":
      expected_dist_data["field"] = np.array([(1.+1.)/2., 2., 3., 4., 5.])
  else:
    partial_distri = np.array([5, 9, 9])
    ln_to_gn_list = [np.array([9,7,5,3,1]),
                     np.array([8]),
                     np.array([1])]
    part_data["field"] = [np.array([9., 7., 5., 3., 1.]), np.array([8.]), np.array([1.])]
    if reduce_func == "sum":
      expected_dist_data["field"] = np.array([6., 7., 8., 9.+1000.])
    elif reduce_func == "min":
      expected_dist_data["field"] = np.array([6., 7., 8., min(9.,1000.)])
    elif reduce_func == "max":
      expected_dist_data["field"] = np.array([6., 7., 8., max(9.,1000.)])
    elif reduce_func == "mean":
      expected_dist_data["field"] = np.array([6., 7., 8., (9.+1000.)/2.])

  _reduce_func = {"sum" : EP.reduce_sum,
                  "min" : EP.reduce_min, 
                  "max" : EP.reduce_max, 
                  "mean": EP.reduce_mean}[reduce_func]

  dist_data = EP.part_to_block(part_data, partial_distri, ln_to_gn_list, comm, reduce_func=_reduce_func)
  assert dist_data["field"].dtype == np.float64
  assert (dist_data["field"] == expected_dist_data["field"]).all()

@pytest_parallel.mark.parallel(2)
def test_part_to_part(comm):

  #Test wo. stride
  if comm.Get_rank() == 0:
    gnum1 = [np.array([1,3,5]), np.array([11])]
    gnum2 = [np.array([9])]
  elif comm.Get_rank() == 1:
    gnum1 = [np.array([7,9])]
    gnum2 = [np.array([7,5,5,1,11])]

  send = [10.0 * t for t in gnum1]

  recv = EP.part_to_part(send, gnum1, gnum2, comm)
  for r,g in zip(recv, gnum2):
    assert (r == 10.0*g).all()

  #Test with stride
  if comm.Get_rank() == 0:
    gnum1 = [np.array([1,3,5]), np.array([11])]
    stride = [np.array([1,1,2], np.int32), np.array([1], np.int32)]
    send = [np.array([10,30,50,51.]), np.array([110.])]
    gnum2 = [np.array([9])]
  elif comm.Get_rank() == 1:
    gnum1 = [np.array([7,9])]
    stride = [np.array([2,1], np.int32)]
    send = [np.array([70., 71, 90.])]
    gnum2 = [np.array([7,5,5,1,11])]

  recv_stride, recv = EP.part_to_part_strided(stride, send, gnum1, gnum2, comm)
  if comm.Get_rank() == 0:
    assert (recv_stride[0] == [1]).all()
    assert (recv[0] == [90.]).all()
  elif comm.Get_rank() == 1:
    assert (recv_stride[0] == [2,2,2,1,1]).all()
    assert (recv[0] == [70.,71,50,51,50,51,10,110]).all()
