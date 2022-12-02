import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.utils import np_utils, par_utils
from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype

def get_partitioned_zones(part_tree, dist_zone_path):
  """
  Return a list of the partitioned zones created from a distributed zone name
  found in part_tree
  """
  base_name, zone_name = PT.path_head(dist_zone_path), PT.path_tail(dist_zone_path)
  part_base = PT.get_node_from_path(part_tree, base_name)
  if part_base:
    return [part for part in PT.iter_all_Zone_t(part_base) if \
        MT.conv.get_part_prefix(PT.get_name(part)) == zone_name]
  else:
    return []

def get_cgns_distribution(dist_node, name):
  """
  Return the (partial) distribution array of a distributed zone from
  its path.
  """
  return PT.get_value(MT.getDistribution(dist_node, name))

def create_all_elt_distribution(dist_elts, comm):
  """
  Create the :CGNS#Distribution-like distribution array we would
  have if all the Element_t nodes were concatenated
  """
  elt_sections_dn  = [PT.Element.Size(elt) for elt in dist_elts]
  elt_sections_idx = np_utils.sizes_to_indices(elt_sections_dn)
  return par_utils.uniform_distribution(elt_sections_idx[-1], comm)

def collect_cgns_g_numbering(part_nodes, name, prefix=''):
  """
  Return the list of the CGNS:GlobalNumbering array of name name found for each
  partition, stating from part_node and searching under the (optional) prefix path
  An empty array is returned if the partitioned node does not exists
  """
  prefixed = lambda node : PT.get_node_from_path(node, prefix)
  return [np.empty(0, pdm_gnum_dtype) if prefixed(part_node) is None else \
      PT.get_value(MT.getGlobalNumbering(prefixed(part_node), name)).astype(pdm_gnum_dtype) for part_node in part_nodes]

def create_all_elt_g_numbering(p_zone, dist_elts):
  """
  Create for the partitioned zone p_zone the global numbering array
  that would correspond to all the Elements_t of the mesh.
  """
  sorting_idx = np.argsort([PT.Element.Range(elt)[0] for elt in dist_elts])
  sorted_dist_elts  = [dist_elts[k] for k in sorting_idx]
  elt_sections_dn   = [PT.Element.Size(elt) for elt in sorted_dist_elts]
  elt_sections_idx  = np_utils.sizes_to_indices(elt_sections_dn, dtype=np.int32)
  p_elts = [PT.get_node_from_name(p_zone, PT.get_name(elt)) for elt in sorted_dist_elts]
  elt_sections_pn = [PT.Element.Size(elt) if elt else 0 for elt in p_elts]
  offset = 0
  np_elt_ln_to_gn = np.empty(sum(elt_sections_pn), dtype=pdm_gnum_dtype)
  for i_elt, p_elt in enumerate(p_elts):
    if p_elt:
      local_ln_gn = PT.get_value(MT.getGlobalNumbering(p_elt, 'Element'))
      np_elt_ln_to_gn[offset:offset+elt_sections_pn[i_elt]] = local_ln_gn + elt_sections_idx[i_elt]
      offset += elt_sections_pn[i_elt]
  return np_elt_ln_to_gn

@PT.check_is_label('Zone_t')
def get_entities_numbering(part_zone, as_pdm=True):
  """
  Shortcut to return vertex, face and cell global numbering of a partitioned
  (structured or unstructured/ngon) zone.
  If as_pdm is True, force output to have PDM_gnum_t data type
  """
  vtx_ln_to_gn   = PT.get_value(MT.getGlobalNumbering(part_zone, 'Vertex'))
  cell_ln_to_gn  = PT.get_value(MT.getGlobalNumbering(part_zone, 'Cell'))
  if PT.Zone.Type(part_zone) == "Structured":
    face_ln_to_gn = PT.get_value(MT.getGlobalNumbering(part_zone, 'Face'))
  else:
    ngon = PT.Zone.NGonNode(part_zone)
    face_ln_to_gn = PT.get_value(MT.getGlobalNumbering(ngon, 'Element'))
  if as_pdm:
    vtx_ln_to_gn  = vtx_ln_to_gn.astype(pdm_gnum_dtype, casting='same_kind', copy=False)
    face_ln_to_gn = face_ln_to_gn.astype(pdm_gnum_dtype, casting='same_kind', copy=False)
    cell_ln_to_gn = cell_ln_to_gn.astype(pdm_gnum_dtype, casting='same_kind', copy=False)
  return vtx_ln_to_gn, face_ln_to_gn, cell_ln_to_gn

def create_mask_tree(root, labels, include, exclude):
  """
  Create a mask tree from root using either the include or exclude list + hints on searched labels
  """
  if len(include) * len(exclude) != 0:
    raise ValueError("`include` and `exclude` args are mutually exclusive")

  if len(include) > 0:
    to_include = PT.concretize_paths(root, include, labels)
  elif len(exclude) > 0:
    #In exclusion mode, we get all the paths matching labels and exclude the one founded
    all_paths = PT.predicates_to_paths(root, labels)
    to_exclude = PT.concretize_paths(root, exclude, labels)
    to_include = [p for p in all_paths if not p in to_exclude]
  else:
    to_include = PT.predicates_to_paths(root, labels)

  return PT.paths_to_tree(to_include, PT.get_name(root))

