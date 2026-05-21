import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.utils import np_utils

from .geometry import _compute_elements_center

def get_point_cloud(zone, comm, location):
  """
  If location == Vertex, return the (interlaced) coordinates of vertices 
  and vertex global numbering of a partitioned zone
  If location == Center, compute and return the (interlaced) coordinates of
  cell centers and cell global numbering of a partitioned zone
  """

  if location == 'Vertex':
    distri   = MT.Zone.vtx_distribution(zone)
    _coords = PT.Zone.coordinates(zone)
    coords = [c if c is not None else np.zeros_like(_coords[0]) for c in _coords]
    coords   = np_utils.interweave_arrays(coords)

  elif location == 'CellCenter':
    distri   = MT.Zone.cell_distribution(zone)
    coords = _compute_elements_center(zone, 'CellCenter', comm)
  
  #Try to catch a container with the given name
  elif (container := PT.get_child_from_name(zone, location)) is not None:
    assert not PT.pred.IS_SUBSET(container)
    distri = MT.Container.distribution(container, zone)
    coords = [PT.get_value(c).reshape(-1, order='F') for c in PT.get_children_from_name(container, 'Coordinate*')]
    coords = np_utils.interweave_arrays(coords)

  else:
    raise RuntimeError("Unknow location or node")

  ln_to_gn = np.arange(distri[0]+1, distri[1]+1, dtype=distri.dtype)
  return coords, ln_to_gn

def extract_sub_cloud(coords, lngn, indices):
  """
  Extract coordinates and lngn from a list of indices, starting at 0.
  """
  # Reexport func
  from maia.algo.part import point_cloud_utils as PCUp
  return PCUp.extract_sub_cloud(coords, lngn, indices)