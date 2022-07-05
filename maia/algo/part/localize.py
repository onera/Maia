import numpy as np

import Pypdm.Pypdm as PDM

import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia                        import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.utils                  import py_utils, np_utils
from maia.transfer               import utils as te_utils
from maia.factory.dist_from_part import discover_nodes_from_matching

from .point_cloud_utils import get_point_cloud

# ------------------------------------------------------------------------
def register_src_part(mesh_loc, i_part, part_zone, keep_alive):
  """
  Get connectivity of a partitioned zone and register it a mesh_location
  pdm object
  """

  cx, cy, cz = PT.Zone.coordinates(part_zone)
  vtx_coords = np_utils.interweave_arrays([cx,cy,cz])
  keep_alive.append(vtx_coords)

  ngon  = PT.Zone.NGonNode(part_zone)
  nface = PT.Zone.NFaceNode(part_zone)

  cell_face_idx = I.getNodeFromName1(nface, "ElementStartOffset")[1]
  cell_face     = I.getNodeFromName1(nface, "ElementConnectivity")[1]
  face_vtx_idx  = I.getNodeFromName1(ngon,  "ElementStartOffset")[1]
  face_vtx      = I.getNodeFromName1(ngon,  "ElementConnectivity")[1]

  vtx_ln_to_gn, face_ln_to_gn, cell_ln_to_gn = te_utils.get_entities_numbering(part_zone)
  keep_alive.append(cell_ln_to_gn)

  n_cell = cell_ln_to_gn.shape[0]
  n_face = face_ln_to_gn.shape[0]
  n_vtx  = vtx_ln_to_gn .shape[0]

  mesh_loc.part_set(i_part, n_cell, cell_face_idx, cell_face, cell_ln_to_gn,
                            n_face, face_vtx_idx, face_vtx, face_ln_to_gn,
                            n_vtx, vtx_coords, vtx_ln_to_gn)


# --------------------------------------------------------------------------
def _localize_points(src_parts_per_dom, tgt_parts_per_dom, location, comm, \
    reverse=False, loc_tolerance=1E-6):
  """
  """
  n_dom_src = len(src_parts_per_dom)
  n_dom_tgt = len(tgt_parts_per_dom)

  assert n_dom_src == n_dom_tgt == 1
  n_part_per_dom_src = [len(parts) for parts in src_parts_per_dom]
  n_part_per_dom_tgt = [len(parts) for parts in tgt_parts_per_dom]
  n_part_src = sum(n_part_per_dom_src)
  n_part_tgt = sum(n_part_per_dom_tgt)

  # > Create and setup global data
  mesh_loc = PDM.MeshLocation(mesh_nature=1, n_point_cloud=1, comm=comm, enable_reverse=reverse)
  mesh_loc.mesh_global_data_set(n_part_src) # For now only one domain is supported
  mesh_loc.n_part_cloud_set(0, n_part_tgt)  # For now only one domain is supported

  # > Register source
  keep_alive = list()
  for i_domain, src_part_zones in enumerate(src_parts_per_dom):
    for i_part, src_part in enumerate(src_part_zones):
      register_src_part(mesh_loc, i_part, src_part, keep_alive)

  # > Compute points to be localized and register target
  for i_domain, tgt_part_zones in enumerate(tgt_parts_per_dom):
    for i_part, tgt_part in enumerate(tgt_part_zones):
      coords, ln_to_gn = get_point_cloud(tgt_part, location)
      keep_alive.append(coords)
      keep_alive.append(ln_to_gn)
      mesh_loc.cloud_set(0, i_part, ln_to_gn.shape[0], coords, ln_to_gn)

  mesh_loc.tolerance_set(loc_tolerance)
  mesh_loc.compute()

  all_located_id   = [mesh_loc.located_get  (0,i_part) for i_part in range(n_part_tgt)]
  all_unlocated_id = [mesh_loc.unlocated_get(0,i_part) for i_part in range(n_part_tgt)]

  #This is result from the target perspective (api : (i_pt_cloud, i_part))
  all_target_data = [mesh_loc.location_get(0, i_tgt_part) for i_tgt_part in range(n_part_tgt)]
  # Add ids in dict
  for i_part, data in enumerate(all_target_data):
    data['located_ids']   = all_located_id[i_part]
    data['unlocated_ids'] = all_unlocated_id[i_part]
  # Reshape output to list of lists (as input domains)
  located_per_dom = py_utils.to_nested_list(all_target_data, n_part_per_dom_tgt)

  #This is result from the source perspective (api : ((i_part, i_pt_cloud))
  if reverse:
    all_located_inv = [mesh_loc.points_in_elt_get(i_src_part, 0) for i_src_part in range(n_part_src)]
    located_inv_per_dom = py_utils.to_nested_list(all_located_inv, n_part_per_dom_src)

  if reverse:
    return located_per_dom, located_inv_per_dom
  else:
    return located_per_dom

def localize_points(src_tree, tgt_tree, location, comm, **options):
  """Localize points between two partitioned trees.

  For all the points of the target tree matching the given location,
  search the cell of the source tree in which it is enclosed.
  The result, i.e. the gnum of the source cell (or -1 if the point is not localized),
  is stored in a DiscreteData_t container called "Localization" on the target zones.

  - Source tree must be unstructured and have a ngon connectivity.
  - Partitions must come from a single initial domain on both source and target tree.

  Localization can be parametred thought the options kwargs:

  - ``loc_tolerance`` (default = 1E-6) -- Geometric tolerance for the method.

  Args:
    src_tree (CGNSTree): Source tree, partitionned. Only U-NGon connectivities are managed.
    tgt_tree (CGNSTree): Target tree, partitionned. Structured or U-NGon connectivities are managed.
    location ({'CellCenter', 'Vertex'}) : Target points to localize
    comm       (MPIComm): MPI communicator
    **options: Additional options related to location strategy
  """
  dist_src_doms = I.newCGNSTree()
  discover_nodes_from_matching(dist_src_doms, [src_tree], 'CGNSBase_t/Zone_t', comm,
                                    merge_rule=lambda zpath : MT.conv.get_part_prefix(zpath))
  src_parts_per_dom = list()
  for zone_path in PT.predicates_to_paths(dist_src_doms, 'CGNSBase_t/Zone_t'):
    src_parts_per_dom.append(te_utils.get_partitioned_zones(src_tree, zone_path))

  dist_tgt_doms = I.newCGNSTree()
  discover_nodes_from_matching(dist_tgt_doms, [tgt_tree], 'CGNSBase_t/Zone_t', comm,
                                    merge_rule=lambda zpath : MT.conv.get_part_prefix(zpath))

  tgt_parts_per_dom = list()
  for zone_path in PT.predicates_to_paths(dist_tgt_doms, 'CGNSBase_t/Zone_t'):
    tgt_parts_per_dom.append(te_utils.get_partitioned_zones(tgt_tree, zone_path))

  located_data = _localize_points(src_parts_per_dom, tgt_parts_per_dom, location, comm)

  for i_dom, tgt_parts in enumerate(tgt_parts_per_dom):
    for i_part, tgt_part in enumerate(tgt_parts):
      sol = I.createUniqueChild(tgt_part, "Localization", "DiscreteData_t")
      I.newGridLocation(location, sol)
      data = located_data[i_dom][i_part]
      n_tgts = data['located_ids'].size + data['unlocated_ids'].size,
      src_gnum = -np.ones(n_tgts, dtype=pdm_gnum_dtype) #Init with -1 to carry unlocated points
      src_gnum[data['located_ids']-1] = data['location']
      I.newDataArray("SrcId", src_gnum, parent=sol)

