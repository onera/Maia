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
  vtx_distri   = MT.get_distribution(zone, 'Vertex')[1]
  cell_distri  = MT.get_distribution(zone, 'Cell')[1]

  if location == 'Vertex':
    vtx_ln_to_gn = np.arange(vtx_distri[0], vtx_distri[1], dtype=vtx_distri.dtype) + 1
    coords = [c.reshape(-1, order='F') for c in PT.Zone.coordinates(zone)]
    vtx_coords   = np_utils.interweave_arrays(coords)
    return vtx_coords, vtx_ln_to_gn

  elif location == 'CellCenter':
    cell_distri   = MT.get_distribution(zone, 'Cell')[1]
    cell_ln_to_gn = np.arange(cell_distri[0], cell_distri[1], dtype=cell_distri.dtype) + 1
    center_cell = _compute_elements_center(zone, 'CellCenter', comm)
    return center_cell, cell_ln_to_gn
  
  else: #Try to catch a container with the given name
    container = PT.get_child_from_name(zone, location)
    if container:
      assert PT.get_child_from_name(container, 'PointList') is None
      assert PT.get_child_from_name(container, 'PointRange') is None
      coords = [PT.get_value(c).reshape(-1, order='F') for c in PT.get_children_from_name(container, 'Coordinate*')]
      int_coords = np_utils.interweave_arrays(coords)
      if PT.Subset.GridLocation(container) == 'Vertex':
        ln_to_gn = np.arange(vtx_distri[0], vtx_distri[1], dtype=vtx_distri.dtype) + 1
      elif PT.Subset.GridLocation(container) == 'CellCenter':
        ln_to_gn = np.arange(cell_distri[0], cell_distri[1], dtype=cell_distri.dtype) + 1
      return int_coords, ln_to_gn

  raise RuntimeError("Unknow location or node")


def extract_sub_cloud(coords, lngn, indices):
  """
  Extract coordinates and lngn from a list of indices, starting at 0.
  """
  # Reexport func
  from maia.algo.part import point_cloud_utils as PCUp
  return PCUp.extract_sub_cloud(coords, lngn, indices)