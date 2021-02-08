import Converter.Internal as I
import numpy as np

import maia.sids.sids               as SIDS
from maia.utils import py_utils
from maia.distribution.distribution_function import uniform_distribution
from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype

def get_cgns_distribution(dist_zone, path):
  """
  Return the (partial) distribution array of a distributed zone from
  its path. Array is converted to pdm gnum_dtype.
  """
  return I.getNodeFromPath(dist_zone, path)[1].astype(pdm_gnum_dtype)

def create_all_elt_distribution(dist_elts, comm):
  """
  Create the :CGNS#Distribution-like distribution array we would
  have if all the Element_t nodes were concatenated
  """
  elt_sections_dn  = [SIDS.ElementSize(elt) for elt in dist_elts]
  elt_sections_idx = py_utils.nb_to_offset(elt_sections_dn, dtype=pdm_gnum_dtype)
  return uniform_distribution(elt_sections_idx[-1], comm)

def collect_cgns_g_numbering(part_zones, path):
  """
  Return the list of the CGNS:GlobalNumbering array found for each
  partitioned zone following path path.
  An empty array is returned for zone having no global numbering.
  """
  return [I.getNodeFromPath(part_zone, path)[1] if I.getNodeFromPath(part_zone, path)
      is not None else np.empty(0, pdm_gnum_dtype) for part_zone in part_zones]

def create_all_elt_g_numbering(p_zone, dist_elts):
  """
  Create for the partitioned zone p_zone the global numbering array
  that would correspond to all the Elements_t of the mesh.
  """
  sorting_idx = np.argsort([SIDS.ElementRange(elt)[0] for elt in dist_elts])
  sorted_dist_elts  = [dist_elts[k] for k in sorting_idx]
  elt_sections_dn   = [SIDS.ElementSize(elt) for elt in sorted_dist_elts]
  elt_sections_idx  = py_utils.nb_to_offset(elt_sections_dn, dtype=np.int32)
  p_elts = [I.getNodeFromName(p_zone, I.getName(elt)) for elt in sorted_dist_elts]
  elt_sections_pn = [SIDS.ElementSize(elt) if elt else 0 for elt in p_elts]
  offset = 0
  np_elt_ln_to_gn = np.empty(sum(elt_sections_pn), dtype=pdm_gnum_dtype)
  for i_elt, p_elt in enumerate(p_elts):
    if p_elt:
      local_ln_gn = I.getNodeFromPath(p_elt, ':CGNS#GlobalNumbering/Element')[1]
      np_elt_ln_to_gn[offset:offset+elt_sections_pn[i_elt]] = local_ln_gn + elt_sections_idx[i_elt]
      offset += elt_sections_pn[i_elt]
  return np_elt_ln_to_gn

# def collect_lntogn_from_splited_path(part_zones, prefix, suffix):
  # lngn_list = list()
  # for p_zone in part_zones:
    # extension  = '.'.join(I.getName(p_zone).split('.')[-2:])
    # ln_gn_path = '{0}.{1}/{2}'.format(prefix, extension, suffix)
    # ln_gn_node = I.getNodeFromPath(p_zone, ln_gn_path)
    # if ln_gn_node:
      # lngn_list.append(I.getValue(ln_gn_path))
  # return lngn_list

