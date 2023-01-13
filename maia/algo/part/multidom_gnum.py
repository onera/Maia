import numpy as np

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.utils import par_utils


def _get_shifted_arrays(arrays_per_dom, comm):
  shifted_per_dom = []
  offset = np.zeros(len(arrays_per_dom)+1, dtype=pdm_gnum_dtype)
  for i_dom, arrays in enumerate(arrays_per_dom):
    offset[i_dom+1] = offset[i_dom] + par_utils.arrays_max(arrays, comm)
    shifted_per_dom.append([array + offset[i_dom] for array in arrays]) # Shift (with copy)
  return offset, shifted_per_dom

def get_shifted_ln_to_gn_from_loc(parts_per_dom, location, comm):
  """ Wraps _get_zone_ln_to_gn_from_loc around multiple domains,
  shifting lngn with previous values"""
  from .point_cloud_utils import _get_zone_ln_to_gn_from_loc
  lngns_per_dom = []
  for part_zones in parts_per_dom:
    lngns_per_dom.append([_get_zone_ln_to_gn_from_loc(part, location) for part in part_zones])
  return _get_shifted_arrays(lngns_per_dom, comm)


