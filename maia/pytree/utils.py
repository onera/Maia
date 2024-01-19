import numpy as np

import maia.pytree as PT

def _gc_transform_point(index_1, start_1, start_2, tr):
    return np.matmul(tr, (index_1 - start_1)) + start_2

def _gc_transform_window(window_1, start_1, start_2, tr):
    window_2 = np.empty_like(window_1)
    window_2[:,0] = _gc_transform_point(window_1[:,0], start_1, start_2, tr)
    window_2[:,1] = _gc_transform_point(window_1[:,1], start_1, start_2, tr)
    return window_2

def gc_transform_point(gc, idx):
    """
    Compute the indices of a given interface point in the opposite zone
    axis system 
    """
    return _gc_transform_point(idx,
                               PT.get_child_from_name(gc, 'PointRange')[1][:,0],
                               PT.get_child_from_name(gc, 'PointRangeDonor')[1][:,0],
                               PT.GridConnectivity.Transform(gc, True))
def gc_transform_window(gc, window):
    """
    Compute the indices of a given interface window in the opposite zone
    axis system 
    """
    return _gc_transform_window(window,
                                PT.get_child_from_name(gc, 'PointRange')[1][:,0],
                                PT.get_child_from_name(gc, 'PointRangeDonor')[1][:,0],
                                PT.GridConnectivity.Transform(gc, True))


