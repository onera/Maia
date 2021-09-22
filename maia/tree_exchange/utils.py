import Converter.Internal as I
import numpy as np

import maia.sids.sids               as SIDS
import maia.sids.Internal_ext as IE
from maia.sids  import conventions as conv
from maia.utils import py_utils
from maia.distribution.distribution_function import uniform_distribution
from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype

def get_partitioned_zones(part_tree, dist_zone_path):
  """
  Return a list of the partitioned zones created from a distributed zone name
  found in part_tree
  """
  base_name, zone_name = I.getPathAncestor(dist_zone_path), I.getPathLeaf(dist_zone_path)
  part_base = I.getNodeFromPath(part_tree, base_name)
  if part_base:
    return [part for part in I.getZones(part_base) if \
        conv.get_part_prefix(I.getName(part)) == zone_name]
  else:
    return []

def get_cgns_distribution(dist_node, name):
  """
  Return the (partial) distribution array of a distributed zone from
  its path. Array is converted to pdm gnum_dtype.
  """
  return I.getVal(IE.getDistribution(dist_node, name)).astype(pdm_gnum_dtype)

def create_all_elt_distribution(dist_elts, comm):
  """
  Create the :CGNS#Distribution-like distribution array we would
  have if all the Element_t nodes were concatenated
  """
  elt_sections_dn  = [SIDS.ElementSize(elt) for elt in dist_elts]
  elt_sections_idx = py_utils.sizes_to_indices(elt_sections_dn, dtype=pdm_gnum_dtype)
  return uniform_distribution(elt_sections_idx[-1], comm)

def collect_cgns_g_numbering(part_nodes, name, prefix=''):
  """
  Return the list of the CGNS:GlobalNumbering array of name name found for each
  partition, stating from part_node and searching under the (optional) prefix path
  An empty array is returned if the partitioned node does not exists
  """
  prefixed = lambda node : I.getNodeFromPath(node, prefix)
  return [np.empty(0, pdm_gnum_dtype) if prefixed(part_node) is None else \
      I.getVal(IE.getGlobalNumbering(prefixed(part_node), name)).astype(pdm_gnum_dtype) for part_node in part_nodes]

def create_all_elt_g_numbering(p_zone, dist_elts):
  """
  Create for the partitioned zone p_zone the global numbering array
  that would correspond to all the Elements_t of the mesh.
  """
  sorting_idx = np.argsort([SIDS.ElementRange(elt)[0] for elt in dist_elts])
  sorted_dist_elts  = [dist_elts[k] for k in sorting_idx]
  elt_sections_dn   = [SIDS.ElementSize(elt) for elt in sorted_dist_elts]
  elt_sections_idx  = py_utils.sizes_to_indices(elt_sections_dn, dtype=np.int32)
  p_elts = [I.getNodeFromName(p_zone, I.getName(elt)) for elt in sorted_dist_elts]
  elt_sections_pn = [SIDS.ElementSize(elt) if elt else 0 for elt in p_elts]
  offset = 0
  np_elt_ln_to_gn = np.empty(sum(elt_sections_pn), dtype=pdm_gnum_dtype)
  for i_elt, p_elt in enumerate(p_elts):
    if p_elt:
      local_ln_gn = I.getVal(IE.getGlobalNumbering(p_elt, 'Element'))
      np_elt_ln_to_gn[offset:offset+elt_sections_pn[i_elt]] = local_ln_gn + elt_sections_idx[i_elt]
      offset += elt_sections_pn[i_elt]
  return np_elt_ln_to_gn

@IE.check_is_label('Zone_t')
def get_entities_numbering(part_zone, as_pdm=True):
  """
  Shortcut to return vertex, face and cell global numbering of a partitioned
  (structured or unstructured/ngon) zone.
  If as_pdm is True, force output to have PDM_gnum_t data type
  """
  vtx_ln_to_gn   = I.getVal(IE.getGlobalNumbering(part_zone, 'Vertex'))
  cell_ln_to_gn  = I.getVal(IE.getGlobalNumbering(part_zone, 'Cell'))
  if SIDS.Zone.Type(part_zone) == "Structured":
    face_ln_to_gn = I.getVal(IE.getGlobalNumbering(part_zone, 'Face'))
  else:
    ngons = [e for e in I.getNodesFromType1(part_zone, 'Elements_t') if SIDS.ElementCGNSName(e) == 'NGON_n']
    assert len(ngons) == 1, "For unstructured zones, only NGon connectivity is supported"
    face_ln_to_gn = I.getVal(IE.getGlobalNumbering(ngons[0], 'Element'))
  if as_pdm:
    vtx_ln_to_gn  = vtx_ln_to_gn.astype(pdm_gnum_dtype)
    face_ln_to_gn = face_ln_to_gn.astype(pdm_gnum_dtype)
    cell_ln_to_gn = cell_ln_to_gn.astype(pdm_gnum_dtype)
  return vtx_ln_to_gn, face_ln_to_gn, cell_ln_to_gn


