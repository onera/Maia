import Converter.Internal as I
import numpy as np
from maia.utils.py_utils import list_or_only_elt

def VertexSize(zone):
  assert I.getType(zone) == "Zone_t"
  z_sizes = I.getValue(zone)
  return list_or_only_elt(z_sizes[:,0])

def CellSize(zone):
  assert I.getType(zone) == "Zone_t"
  z_sizes = I.getValue(zone)
  return list_or_only_elt(z_sizes[:,1])

def VertexBoundarySize(zone):
  assert I.getType(zone) == "Zone_t"
  z_sizes = I.getValue(zone)
  return list_or_only_elt(z_sizes[:,2])

def point_range_size(pr_n):
  assert I.getType(pr_n) == "IndexRange_t"
  pr_values = pr_n[1]
  return (pr_values[:,1] - pr_values[:,0] + 1)


def zone_n_vtx( zone ):
  return np.prod(VertexSize(zone))

def zone_n_cell( zone ):
  return np.prod(CellSize(zone))

def zone_n_vtx_bnd( zone ):
  return np.prod(VertexBoundarySize(zone))

def point_range_lenght(pr_n):
  return np.prod(point_range_size(pr_n))
