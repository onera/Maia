import Converter.Internal as I
import numpy as np
from maia.utils.py_utils import list_or_only_elt
from maia.sids.elements_utils import *

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

def ZoneType(zone):
  assert I.getType(zone) == "Zone_t"
  zone_type_n = I.getNodeFromType1(zone, 'ZoneType_t')
  return I.getValue(zone_type_n)


def ElementRange(elements):
  assert I.getType(elements) == "Elements_t"
  return I.getNodeFromName(elements,"ElementRange")[1]

def ElementType(elements):
  assert I.getType(elements) == "Elements_t"
  return elements[1][0]

def GridLocation(node):
  grid_loc_n = I.getNodeFromType1(node, 'GridLocation_t')
  return I.getValue(grid_loc_n) if grid_loc_n else 'Vertex'

def point_range_sizes(pr_n):
  """Allow point_range to be inverted (PR[:,1] < PR[:,0])
  as it can occurs in struct GCs
  """
  assert I.getType(pr_n) == "IndexRange_t"
  pr_values = pr_n[1]
  return np.abs(pr_values[:,1] - pr_values[:,0]) + 1


def zone_n_vtx( zone ):
  return np.prod(VertexSize(zone))

def zone_n_cell( zone ):
  return np.prod(CellSize(zone))

def zone_n_vtx_bnd( zone ):
  return np.prod(VertexBoundarySize(zone))

def point_range_n_elt(pr_n):
  return np.prod(point_range_sizes(pr_n))
