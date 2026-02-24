import pytest
import pytest_parallel
import numpy as np

import maia.transfer.protocols as EP

@pytest_parallel.mark.parallel(3)
def test_check_dict_keys(comm):
  dist_data = {'data1' : comm.rank, 'data2' : 42}
  EP._check_dict_keys(dist_data, comm)
  
  if comm.rank == 1:
    dist_data = {'data2' : 42, 'data1' : comm.rank} # Same key, but inverted order
  with pytest.raises(KeyError):
    EP._check_dict_keys(dist_data, comm)

  dist_data = {'data2' : 42, 'data1' : comm.rank}
  if comm.rank == 2:
    dist_data['data3'] = 'Will fail, because number of keys differs'
  with pytest.raises(KeyError):
    EP._check_dict_keys(dist_data, comm)


@pytest_parallel.mark.parallel(2)
def test_block_to_part(comm):
  dist_data = dict()
  expected_part_data = dict()
  if comm.Get_rank() == 0:
    partial_distri = np.array([0, 5, 10])
    ln_to_gn_list = [np.array([1,3,5,9])]
    dist_data["field"] = np.array([1., 2., 3., 4., 5.])
    expected_part_data["field"] = [np.array([2., 4., 6., 1000.])]
  else:
    partial_distri = np.array([5, 10, 10])
    ln_to_gn_list = [np.array([8,6,4,2,0]),
                     np.array([7]),
                     np.array([0])]
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
    ln_to_gn_list = [np.array([9,7])]
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
    ln_to_gn_list = [np.array([1,3,5,9])]
    part_data["field"] = [np.array([2., 4., 6., 1000.])]
    expected_dist_data["field"] = np.array([1., 2., 3., 4., 5.])
  else:
    partial_distri = np.array([5, 10, 10])
    ln_to_gn_list = [np.array([8,6,4,2,0]),
                     np.array([7]),
                     np.array([0])]
    part_data["field"] = [np.array([9., 7., 5., 3., 1.]), np.array([8.]), np.array([1.])]
    expected_dist_data["field"] = np.array([6., 7., 8., 9., 1000.])

  dist_data = EP.part_to_block(part_data, partial_distri, ln_to_gn_list, comm)
  assert dist_data["field"].dtype == np.float64
  assert (dist_data["field"] == expected_dist_data["field"]).all()

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("reduce_func", ["sum", "min", "max"])
def test_part_to_block_with_reduce(reduce_func, comm):
  part_data = dict()
  expected_dist_data = dict()
  if comm.Get_rank() == 0:
    partial_distri = np.array([0, 5, 9])
    ln_to_gn_list = [np.array([1,3,5,8])]
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
    ln_to_gn_list = [np.array([8,6,4,2,0]),
                     np.array([7]),
                     np.array([0])]
    part_data["field"] = [np.array([9., 7., 5., 3., 1.]), np.array([8.]), np.array([1.])]
    if reduce_func == "sum":
      expected_dist_data["field"] = np.array([6., 7., 8., 9.+1000.])
    elif reduce_func == "min":
      expected_dist_data["field"] = np.array([6., 7., 8., min(9.,1000.)])
    elif reduce_func == "max":
      expected_dist_data["field"] = np.array([6., 7., 8., max(9.,1000.)])
    elif reduce_func == "mean":
      expected_dist_data["field"] = np.array([6., 7., 8., (9.+1000.)/2.])

  _reduce_func = {"sum" : EP.ReduceOp.SUM,
                  "min" : EP.ReduceOp.MIN, 
                  "max" : EP.ReduceOp.MAX}[reduce_func]
                  #"mean": EP.reduce_mean}[reduce_func]

  dist_data = EP.part_to_block(part_data, partial_distri, ln_to_gn_list, comm, reduce_op=_reduce_func)
  assert dist_data["field"].dtype == np.float64
  assert (dist_data["field"] == expected_dist_data["field"]).all()

@pytest_parallel.mark.parallel(2)
def test_part_to_part(comm):

  #Test w/o stride
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

@pytest_parallel.mark.parallel(3)
def test_mblock_to_block(comm):
  # 3 ranks, 2 input data
  if comm.rank == 0:
    distri_in = [np.array([0, 5, 20]), np.array([0, 8, 15])]
    distri_out = np.array([0, 7, 35])
  elif comm.rank == 1:
    distri_in = [np.array([5, 5, 20]), np.array([8, 12, 15])]
    distri_out = np.array([7, 30, 35])
  else:
    distri_in = [np.array([5, 20, 20]), np.array([12, 15, 15])]
    distri_out = np.array([30, 35, 35])

  MBTB = EP.MultiBlockToBlock(distri_in, distri_out, comm)

  # Cste stride == 1
  data_in = [np.arange(distri[0], distri[1])+100*i for i,distri in enumerate(distri_in)]

  data_out = MBTB.exchange(data_in)
  expected = [np.arange(0, 7),
              np.concatenate([np.arange(7, 20), np.arange(0,10)+100]),
              np.arange(10, 15)+100][comm.rank]
  assert np.array_equal(expected, data_out)
  
  # Cste stride == 3
  data_in = [np.repeat(data, 3) + np.tile([.1, .2, .3], data.size) for data in data_in]
  
  data_out = MBTB.exchange(data_in, stride_in=3)
  expected = np.repeat(expected, 3) + np.tile([.1, .2, .3], expected.size)

  assert np.array_equal(expected, data_out)


  # Variable stride
  if comm.rank == 0:
    stride_in = [np.array([0,1,1,0,2]), np.array([0,0,0,0,1,0,0,3])]
    data_in = [np.array([1.1, 2.1 ,4.1,4.2]), np.array([104.1, 107.1,107.2,107.3])]
  elif comm.rank == 1:
    stride_in = [np.array([], int), np.array([1,3,3,1])]
    data_in = [np.array([]), np.array([108.1, 109.1,109.2,109.3, 110.1,110.2,110.3, 111.1])]
  elif comm.rank == 2:
    stride_in = [np.array([1,1,1,0,0,0,0,0,2,2,0,0,0,1,0], int), np.array([0,2,0])]
    data_in = [np.array([5.1, 6.1, 7.1, 13.1,13.2, 14.1,14.2, 18.1]), np.array([113.1,113.2])]

  stride_out, data_out = MBTB.exchange(data_in, stride_in)
  
  if comm.rank == 0:
    expt_stride = np.array([0,1,1,0,2,1,1])
    expt_data = np.array([1.1, 2.1, 4.1,4.2, 5.1, 6.1])
  elif comm.rank == 1:
    expt_stride = np.array([1,0,0,0,0,0,2,2,0,0,0,1,0, 0,0,0,0,1,0,0,3,1,3])
    expt_data = np.array([7.1, 13.1,13.2, 14.1,14.2, 18.1, 104.1, 107.1,107.2,107.3, 108.1, 109.1,109.2,109.3])
  elif comm.rank == 2:
    expt_stride = np.array([3,1,0,2,0])
    expt_data = np.array([110.1,110.2,110.3, 111.1, 113.1,113.2])

  assert np.array_equal(expt_stride, stride_out)
  assert np.array_equal(expt_data, data_out)