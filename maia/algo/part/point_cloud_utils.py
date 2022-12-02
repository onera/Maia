import Pypdm.Pypdm as PDM
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia                        import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.utils                  import np_utils, as_pdm_gnum, layouts

from maia.algo.part.geometry  import compute_cell_center

def _get_zone_ln_to_gn_from_loc(zone, location):
  """ Wrapper to get the expected lngn value """
  _loc = location.replace('Center', '')
  ln_to_gn = as_pdm_gnum(PT.get_value(MT.getGlobalNumbering(zone, _loc)))
  return ln_to_gn

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
    vtx_ln_to_gn = _get_zone_ln_to_gn_from_loc(zone, location)
    return vtx_coords, vtx_ln_to_gn

  elif location == 'CellCenter':
    cell_ln_to_gn = _get_zone_ln_to_gn_from_loc(zone, location)
    center_cell = compute_cell_center(zone)
    return center_cell, cell_ln_to_gn
  
  else: #Try to catch a container with the given name
    container = PT.get_child_from_name(zone, location)
    if container:
      coords = [PT.get_value(c).reshape(-1, order='F') for c in PT.get_children_from_name(container, 'Coordinate*')]
      int_coords = np_utils.interweave_arrays(coords)
      ln_to_gn = _get_zone_ln_to_gn_from_loc(zone, PT.Subset.GridLocation(container))
      return int_coords, ln_to_gn

  raise RuntimeError("Unknow location or node")

def extract_sub_cloud(coords, lngn, indices):
  """
  Extract coordinates and lngn from a list of indices, starting at 0.
  """
  sub_lngn   = layouts.extract_from_indices(lngn  , indices, 1, 0)
  sub_coords = layouts.extract_from_indices(coords, indices, 3, 0)
  return sub_coords, sub_lngn

def create_sub_numbering(lngn_l, comm):
  """
  Create a new compact, starting at 1 numbering from a list of
  gnums.
  """
  n_part = len(lngn_l)
  gen_gnum = PDM.GlobalNumbering(3, n_part, 0, 0., comm)

  for i_part, lngn in enumerate(lngn_l):
    gen_gnum.gnum_set_from_parent(i_part, lngn.size, lngn)

  gen_gnum.gnum_compute()

  return [gen_gnum.gnum_get(i_part)["gnum"] for i_part in range(n_part)]
