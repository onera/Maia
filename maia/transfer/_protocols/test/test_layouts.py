from cmaia.utils import layouts

import numpy as np
def test_put_strided():
    idx = np.array([1,0,2,1])

    counts = np.array([3,1,2])
    data_in = np.array([10.,11.,12.,   13,  100,102], float)

    data_out = np.empty(7, float)
    layouts.take_stridedDI(counts, data_in, idx, data_out)
    assert (data_out == [13., 10,11,12, 100,102, 13]).all()
