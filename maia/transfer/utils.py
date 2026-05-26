import numpy as np

from maia.typing import *
import maia.pytree       as PT
import maia.pytree.maia  as MT
import maia.pytree.utils as PTu

from maia.utils import np_utils, par_utils
from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype

def create_all_elt_distribution(dist_elts: List[CGNSTree], comm: MPIComm) -> NDArray:
  """
  Create the :CGNS#Distribution-like distribution array we would
  have if all the Element_t nodes were concatenated
  """
  elt_sections_dn  = [PT.Element.Size(elt) for elt in dist_elts]
  return par_utils.uniform_distribution(sum(elt_sections_dn), comm)

 
def create_all_elt_g_numbering(p_zone: CGNSPartTree, dist_elts: List[CGNSTree]) -> NDArray:
  """
  Create for the partitioned zone p_zone the global numbering array
  that would correspond to all the Elements_t of the mesh.
  """
  sorting_idx = np.argsort([PT.Element.Range(elt)[0] for elt in dist_elts])
  sorted_dist_elts  = [dist_elts[k] for k in sorting_idx]
  elt_sections_dn   = [PT.Element.Size(elt) for elt in sorted_dist_elts]
  elt_sections_idx  = np_utils.sizes_to_indices(elt_sections_dn, dtype=pdm_gnum_dtype)
  p_elts = [PT.get_node_from_name(p_zone, PT.get_name(elt)) for elt in sorted_dist_elts]
  elt_sections_pn = [MT.Element.globalnumbering(elt).size if elt else 0 for elt in p_elts]
  offset = 0
  np_elt_ln_to_gn = np.empty(sum(elt_sections_pn), dtype=pdm_gnum_dtype)
  for i_elt, p_elt in enumerate(p_elts):
    if p_elt:
      local_ln_gn = MT.Element.globalnumbering(p_elt)
      np_elt_ln_to_gn[offset:offset+elt_sections_pn[i_elt]] = local_ln_gn + elt_sections_idx[i_elt]
      offset += elt_sections_pn[i_elt]
  return np_elt_ln_to_gn

def get_entities_numbering(part_zone: CGNSTree) -> \
     Tuple[NDArray, Optional[NDArray], Optional[NDArray], NDArray]:
  """
  Shortcut to return vertex, edge, face and cell global numbering of a partitioned
  (structured or unstructured) zone. Arrays can be None if numbering does not exists.
  """
  vtx_ln_to_gn  = MT.Zone.vtx_globalnumbering(part_zone)
  edge_ln_to_gn = None
  face_ln_to_gn = None
  cell_ln_to_gn = MT.Zone.cell_globalnumbering(part_zone)

  edge_ln_to_gn_n = MT.get_GlobalNumbering(part_zone, 'Edge')
  if edge_ln_to_gn_n is not None:
    edge_ln_to_gn = PT.get_np_value(edge_ln_to_gn_n)
  elif PT.Zone.has_ngon_elements(part_zone) and PT.Zone.CellDimension(part_zone) == 2:
    try:
      edge = MT.Zone.EdgeNode(part_zone)
    except: # In some 2D Poly meshes, edges are not defined
      pass
    else:
      edge_ln_to_gn = MT.Element.globalnumbering(edge)

  face_ln_to_gn_n = MT.get_GlobalNumbering(part_zone, 'Face')
  if face_ln_to_gn_n is not None:
    face_ln_to_gn = PT.get_np_value(face_ln_to_gn_n)
  elif PT.Zone.has_ngon_elements(part_zone):
    # Face can be recovered from ngon global numbering
    ngon = PT.Zone.NGonNode(part_zone)
    face_ln_to_gn = MT.Element.globalnumbering(ngon)

  return vtx_ln_to_gn, edge_ln_to_gn, face_ln_to_gn, cell_ln_to_gn

def create_mask_tree(root: CGNSTree, labels: List[str], include: List[str], exclude: List[str]) -> CGNSTree:
  """
  Create a mask tree from root using either the include or exclude list + hints on searched labels
  """
  if len(include) * len(exclude) != 0:
    raise ValueError("`include` and `exclude` args are mutually exclusive")

  if len(include) > 0:
    to_include = PTu.concretize_paths(root, include, labels)
  elif len(exclude) > 0:
    #In exclusion mode, we get all the paths matching labels and exclude the one founded
    all_paths = PT.predicates_to_paths(root, labels)
    to_exclude = PTu.concretize_paths(root, exclude, labels)
    to_include = [p for p in all_paths if not p in to_exclude]
  else:
    to_include = PT.predicates_to_paths(root, labels)

  return PTu.paths_to_tree(to_include, PT.get_name(root))

