import pytest
import pytest_parallel

import numpy as np

from maia.transfer._protocols import block_to_block

def test_overlap_size():
  assert block_to_block._overlap_size(1, 10, 15, 20) == 0
  assert block_to_block._overlap_size(15, 20, 1, 3) == 0
  assert block_to_block._overlap_size(1, 100, 50, 60) == 60-50
  assert block_to_block._overlap_size(6, 8, 1, 12) == 8-6
  assert block_to_block._overlap_size(0, 10, 10, 20) == 0
  assert block_to_block._overlap_size(0, 10, 5, 30) == 5

@pytest_parallel.mark.parallel(2)
def test_BlockToBlock(comm):
    distri_in  = np.array([0, 8, 10])
    distri_out = np.array([0, 5, 10])
    rank = comm.Get_rank()

    BTB = block_to_block.BlockToBlock(distri_in, distri_out, comm)
    if rank == 0:
      assert (BTB.send_counts == [5,3]).all()
      assert (BTB.recv_counts == [5,0]).all()
    if rank == 1:
      assert (BTB.send_counts == [0,2]).all()
      assert (BTB.recv_counts == [3,2]).all()

    # Cst stride exchange
    data_in = 10. * (np.arange(distri_in[rank], distri_in[rank+1]) + 1)
    data_out = BTB.exchange(data_in)
    assert (data_out == 10 * (np.arange(distri_out[rank], distri_out[rank+1]) + 1)).all()

    data_in = 10. * (np.arange(4*distri_in[rank], 4*distri_in[rank+1]) + 1)
    data_out = BTB.exchange(data_in, 4)
    assert (data_out == 10 * (np.arange(4*distri_out[rank], 4*distri_out[rank+1]) + 1)).all()

    # Var stride exchange
    if rank == 0:
      stride_in = np.array([0,0,3,0,0,2,0,1])
      data_in = np.array([110, 111, 112,   121, 122,  131])
      expt_stride_out = np.array([0,0,3,0,0])
      expt_data_out = np.array([110, 111, 112])

      
    elif rank == 1:
      stride_in = np.array([1,2])
      data_in = np.array([141,  151, 152])
      expt_stride_out = np.array([2,0,1,1,2])
      expt_data_out = np.array([121,122, 131, 141, 151,152])

    stride_out, data_out = BTB.exchange_field(data_in, stride_in)
    assert (stride_out == expt_stride_out).all()
    assert (data_out == expt_data_out).all()
    

