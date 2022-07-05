import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia                        import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.utils                  import np_utils

from maia.algo.part.geometry  import compute_cell_center

def get_point_cloud(zone, location='CellCenter'):
  """
  If location == Vertex, return the (interlaced) coordinates of vertices 
  and vertex global numbering of a partitioned zone
  If location == Center, compute and return the (interlaced) coordinates of
  cell centers and cell global numbering of a partitioned zone
  """
  if location == 'Vertex':
    coords = [c.reshape(-1, order='F') for c in PT.Zone.coordinates(zone)]
    vtx_coords   = np_utils.interweave_arrays(coords)
    vtx_ln_to_gn = I.getVal(MT.getGlobalNumbering(zone, 'Vertex')).astype(pdm_gnum_dtype)
    return vtx_coords, vtx_ln_to_gn

  elif location == 'CellCenter':
    cell_ln_to_gn = I.getVal(MT.getGlobalNumbering(zone, 'Cell')).astype(pdm_gnum_dtype)
    center_cell = compute_cell_center(zone)
    return center_cell, cell_ln_to_gn
  
  elif PT.is_valid_name(location):
    container = I.getNodeFromName1(zone, location)
    if container:
      coords = [I.getVal(c).reshape(-1, order='F') for c in PT.get_children_from_name(container, 'Coordinate*')]
      loc = PT.Subset.GridLocation(container)
      _loc = loc.replace('Center', '')
      int_coords = np_utils.interweave_arrays(coords)
      ln_to_gn = I.getVal(MT.getGlobalNumbering(zone, _loc)).astype(pdm_gnum_dtype)
      return int_coords, ln_to_gn

  raise RuntimeError("Unknow location or node")

