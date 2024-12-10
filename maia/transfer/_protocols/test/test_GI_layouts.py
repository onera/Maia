from cmaia.utils import layouts

import numpy as np
def test_take_strided():
    idx = np.array([1,0,2,1])

    counts = np.array([3,1,2])
    data_in = np.array([10.,11.,12.,   13,  100,102], float)

    data_out = np.empty(7, float)
    layouts.take_stridedDI(counts, data_in, idx, data_out)
    assert (data_out == [13., 10,11,12, 100,102, 13]).all()


def test_put_strided():
    idx = np.array([1,0,2,1])
    counts_out = np.array([3,2,1], int)
    data_out   = np.empty(6, float)

    counts_in = np.array([2,3,1,1], int)  # => Only one compatible stride for idx 1
    data_in = np.array([1.1, 1.2,   2.1, 2.2, 2.3,   3.1,   4.1])
    data_out.fill(-1)

    layouts.put_strided(data_out, counts_out, idx, counts_in, data_in)
    assert (data_out == np.array([2.1,2.2,2.3,  1.1,1.2,  3.1])).all()
    

    counts_in = np.array([2,4,1,1], int) # => No compatible stride for idx 0
    data_in = np.array([1.1, 1.2,   2.1, 2.2, 2.3, 2.4,   3.1,   4.1])
    data_out.fill(-1)

    layouts.put_strided(data_out, counts_out, idx, counts_in, data_in)
    assert (data_out == np.array([-1,-1.,-1.,  1.1,1.2,  3.1])).all()


    counts_in = np.array([2,3,1,2], int) # => Two compatible stride for idx 1 (last is keep)
    data_in = np.array([1.1, 1.2,   2.1, 2.2, 2.3,   3.1,   4.1, 4.2]) 
    data_out.fill(-1)

    layouts.put_strided(data_out, counts_out, idx, counts_in, data_in)
    assert (data_out == np.array([2.1,2.2,2.3,  4.1,4.2,  3.1])).all()