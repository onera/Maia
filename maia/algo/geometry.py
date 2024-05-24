
import maia.pytree        as PT
import maia.pytree.maia   as MT
from maia.algo.apply_function_to_nodes import zones_iterator

from .dist import geometry as dist_geometry
from .part import geometry as part_geometry

def _compute_vol_center(zone, comm=None):
  if MT.getDistribution(zone) is not None:
    assert comm is not None
    return dist_geometry.compute_cell_center(zone, comm)
  else:
    return part_geometry.compute_cell_center(zone)

def _compute_face_center(zone, comm=None):
  if MT.getDistribution(zone) is not None:
    assert comm is not None
    return dist_geometry.compute_face_center(zone, comm)
  else:
    return part_geometry.compute_face_center(zone)
