# This module contains API functions deprecated in v1.5. 
# It has to be removed when maia 1.6 is released

import warnings
def compute_cell_center(zone):
    from .geometry import _compute_elements_center
    msg = "This function is deprecated and will be removed in next release. "\
          "Consider using ``maia.algo.compute_elements_center(zone, 3)`` instead."
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return _compute_elements_center(zone, 3)

def compute_face_center(zone):
    from .geometry import _compute_elements_center
    msg = "This function is deprecated and will be removed in next release. "\
          "Consider using ``maia.algo.compute_elements_center(zone, 2)`` instead."
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return _compute_elements_center(zone, 2)

def compute_edge_center(zone):
    from .geometry import _compute_elements_center
    msg = "This function is deprecated and will be removed in next release. "\
          "Consider using ``maia.algo.compute_elements_center(zone, 1)`` instead."
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return _compute_elements_center(zone, 1)