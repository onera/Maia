import numpy as np

import Pypdm.Pypdm as PDM

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia                        import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.utils                  import py_utils, np_utils, par_utils
from maia.transfer               import utils as te_utils
from maia.factory.dist_from_part import get_parts_per_blocks

from .point_cloud_utils import get_shifted_point_clouds

def _get_part_data(part_zone):
  cx, cy, cz = PT.Zone.coordinates(part_zone)
  vtx_coords = np_utils.interweave_arrays([cx,cy,cz])

  ngon  = PT.Zone.NGonNode(part_zone)
  nface = PT.Zone.NFaceNode(part_zone)

  cell_face_idx = PT.get_child_from_name(nface, "ElementStartOffset")[1]
  cell_face     = PT.get_child_from_name(nface, "ElementConnectivity")[1]
  face_vtx_idx  = PT.get_child_from_name(ngon,  "ElementStartOffset")[1]
  face_vtx      = PT.get_child_from_name(ngon,  "ElementConnectivity")[1]

  vtx_ln_to_gn, face_ln_to_gn, cell_ln_to_gn = te_utils.get_entities_numbering(part_zone)

  return [cell_face_idx, cell_face, cell_ln_to_gn, \
      face_vtx_idx, face_vtx, face_ln_to_gn, vtx_coords, vtx_ln_to_gn]

def _mesh_location(src_parts, tgt_clouds, comm, reverse=False, loc_tolerance=1E-6):
  """ Wrapper of PDM mesh location
  For now, only 1 domain is supported so we expect source parts and target clouds
  as flat lists :
  Parts are tuple (cell_face_idx, cell_face, cell_lngn,
   face_vtx_idx, face_vtx, face_lngn, vtx_coords, vtx_lngn)
  Cloud are tuple (coords, lngn)
  """

  n_part_src = len(src_parts)
  n_part_tgt = len(tgt_clouds)
  # > Create and setup global data
  mesh_loc = PDM.MeshLocation(mesh_nature=1, n_point_cloud=1, comm=comm, enable_reverse=True) #PDM crash if reverse=False
  mesh_loc.mesh_global_data_set(n_part_src)  # For now only one domain is supported
  mesh_loc.n_part_cloud_set(0, n_part_tgt)   # For now only one domain is supported

  # > Register source
  for i_part, part_data in enumerate(src_parts):
    cell_face_idx, cell_face, cell_ln_to_gn, face_vtx_idx, face_vtx, face_ln_to_gn, \
      vtx_coords, vtx_ln_to_gn = part_data
    mesh_loc.part_set(i_part, cell_ln_to_gn.size, cell_face_idx, cell_face, cell_ln_to_gn,
                              face_ln_to_gn.size, face_vtx_idx, face_vtx, face_ln_to_gn,
                              vtx_ln_to_gn.size, vtx_coords, vtx_ln_to_gn)
  # > Setup target
  for i_part, (coords, lngn) in enumerate(tgt_clouds):
    mesh_loc.cloud_set(0, i_part, lngn.shape[0], coords, lngn)

  mesh_loc.tolerance_set(loc_tolerance)
  mesh_loc.compute()

  # This is located and unlocated indices
  all_located_id   = [mesh_loc.located_get  (0,i_part) for i_part in range(n_part_tgt)]
  all_unlocated_id = [mesh_loc.unlocated_get(0,i_part) for i_part in range(n_part_tgt)]

  #This is result from the target perspective (api : (i_pt_cloud, i_part))
  all_target_data = [mesh_loc.location_get(0, i_tgt_part) for i_tgt_part in range(n_part_tgt)]
  # Add ids in dict
  for i_part, data in enumerate(all_target_data):
    data.pop('g_num')
    data['located_ids']   = all_located_id[i_part] - 1
    data['unlocated_ids'] = all_unlocated_id[i_part] - 1

  #This is result from the source perspective (api : ((i_part, i_pt_cloud))
  if reverse:
    all_located_inv = [mesh_loc.points_in_elt_get(i_src_part, 0) for i_src_part in range(n_part_src)]
    return all_target_data, all_located_inv
  else:
    return all_target_data

def _localize_points(src_parts_per_dom, tgt_parts_per_dom, location, comm, \
    reverse=False, loc_tolerance=1E-6):
  """
  """
  locs = ['Cell', 'Face', 'Vtx']
  n_dom_src = len(src_parts_per_dom)
  n_dom_tgt = len(tgt_parts_per_dom)

  n_part_per_dom_src = [len(parts) for parts in src_parts_per_dom]
  n_part_per_dom_tgt = [len(parts) for parts in tgt_parts_per_dom]
  n_part_src = sum(n_part_per_dom_src)
  n_part_tgt = sum(n_part_per_dom_tgt)

  # > Register source
  src_parts = []
  src_offsets = {loc : np.zeros(n_dom_src+1, dtype=pdm_gnum_dtype) for loc in locs}
  for i_domain, src_part_zones in enumerate(src_parts_per_dom):
    src_parts_domain = [_get_part_data(src_part) for src_part in src_part_zones]
    # Compute global offsets for this domain
    for array_idx, loc in zip([2,5,7], locs):
      dom_max = par_utils.arrays_max([src_part[array_idx] for src_part in src_parts_domain], comm)
      src_offsets[loc][i_domain+1] = src_offsets[loc][i_domain] + dom_max
    # Shift source arrays (inplace)
    for src_part in src_parts_domain:
      src_part[2] += src_offsets['Cell'][i_domain] #cell_ln_to_gn
      src_part[5] += src_offsets['Face'][i_domain] #face_ln_to_gn
      src_part[7] += src_offsets['Vtx' ][i_domain] #vtx_ln_to_gn
    src_parts.extend(src_parts_domain)

  tgt_offset, tgt_clouds = get_shifted_point_clouds(tgt_parts_per_dom, location, comm)
  tgt_clouds = py_utils.to_flat_list(tgt_clouds)

  result = _mesh_location(src_parts, tgt_clouds, comm, reverse, loc_tolerance)

  # Shift back source data
  for i_domain, src_parts_domain in enumerate(py_utils.to_nested_list(src_parts, n_part_per_dom_src)):
    for src_part in src_parts_domain:
      src_part[2] -= src_offsets['Cell'][i_domain] #cell_ln_to_gn
      src_part[5] -= src_offsets['Face'][i_domain] #face_ln_to_gn
      src_part[7] -= src_offsets['Vtx' ][i_domain] #vtx_ln_to_gn

  # Shift results and get domain ids
  direct_result = result[0] if reverse else result
  for tgt_result in direct_result:
    tgt_result['location_shifted'] = tgt_result.pop('location') #Rename key
    tgt_result['location'], tgt_result['domain'] = np_utils.shifted_to_local(
        tgt_result['location_shifted'], src_offsets['Cell'])
  if reverse:
    for src_result in result[1]:
      src_result['points_gnum_shifted'] = src_result.pop('points_gnum') #Rename key
      src_result['points_gnum'], src_result['domain'] = np_utils.shifted_to_local(
          src_result['points_gnum_shifted'], tgt_offset)

  # Reshape output to list of lists (as input domains)
  if reverse:
    return py_utils.to_nested_list(result[0], n_part_per_dom_tgt),\
           py_utils.to_nested_list(result[1], n_part_per_dom_src)
  else:
    return py_utils.to_nested_list(result, n_part_per_dom_tgt)

def localize_points(src_tree, tgt_tree, location, comm, **options):
  """Localize points between two partitioned trees.

  For all the points of the target tree matching the given location,
  search the cell of the source tree in which it is enclosed.
  The result, i.e. the gnum & domain number of the source cell (or -1 if the point is not localized),
  are stored in a ``DiscreteData_t`` container called "Localization" on the target zones.

  Source tree must be unstructured and have a NGon connectivity.

  Localization can be parametred thought the options kwargs:

  - ``loc_tolerance`` (default = 1E-6) -- Geometric tolerance for the method.

  Args:
    src_tree (CGNSTree): Source tree, partitionned. Only U-NGon connectivities are managed.
    tgt_tree (CGNSTree): Target tree, partitionned. Structured or U-NGon connectivities are managed.
    location ({'CellCenter', 'Vertex'}) : Target points to localize
    comm       (MPIComm): MPI communicator
    **options: Additional options related to location strategy

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #localize_points@start
        :end-before: #localize_points@end
        :dedent: 2
  """
  _src_parts_per_dom = get_parts_per_blocks(src_tree, comm)
  src_parts_per_dom = list(_src_parts_per_dom.values())
  tgt_parts_per_dom = list(get_parts_per_blocks(tgt_tree, comm).values())

  located_data = _localize_points(src_parts_per_dom, tgt_parts_per_dom, location, comm, **options)

  dom_list = '\n'.join(_src_parts_per_dom.keys())
  for i_dom, tgt_parts in enumerate(tgt_parts_per_dom):
    for i_part, tgt_part in enumerate(tgt_parts):
      sol = PT.update_child(tgt_part, "Localization", "DiscreteData_t")
      PT.new_GridLocation(location, sol)
      data = located_data[i_dom][i_part]
      n_tgts = data['located_ids'].size + data['unlocated_ids'].size,
      src_gnum = -np.ones(n_tgts, dtype=pdm_gnum_dtype) #Init with -1 to carry unlocated points
      src_dom  = -np.ones(n_tgts, dtype=np.int32)
      src_gnum[data['located_ids']] = data['location']
      src_dom [data['located_ids']] = data['domain']
      PT.new_DataArray("SrcId", src_gnum, parent=sol)
      PT.new_DataArray("DomId", src_dom,  parent=sol)
      PT.new_node("DomainList", "Descriptor_t", dom_list, parent=sol)

