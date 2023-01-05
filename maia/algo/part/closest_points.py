import numpy as np

import Pypdm.Pypdm as PDM

import maia.pytree        as PT

from maia                        import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.utils                  import py_utils, par_utils
from maia.factory.dist_from_part import get_parts_per_blocks

from .point_cloud_utils import get_point_cloud


def _closest_points(src_clouds, tgt_clouds, comm, n_pts=1, reverse=False):
  """ Wrapper of PDM mesh location
  For now, only 1 domain is supported so we expect source parts and target clouds
  as flat lists of tuples (coords, lngn)
  """

  # > Create and setup global data
  closest_point = PDM.ClosestPoints(comm, n_closest=n_pts)
  closest_point.n_part_cloud_set(len(src_clouds), len(tgt_clouds))

  # > Setup source
  for i_part, (coords, lngn) in enumerate(src_clouds):
    closest_point.src_cloud_set(i_part, lngn.shape[0], coords, lngn)

  # > Setup target
  for i_part, (coords, lngn) in enumerate(tgt_clouds):
    closest_point.tgt_cloud_set(i_part, lngn.shape[0], coords, lngn)

  closest_point.compute()

  all_closest = [closest_point.points_get(i_part_tgt) for i_part_tgt in range(len(tgt_clouds))]

  if reverse:
    all_closest_inv = [closest_point.tgt_in_src_get(i_src_part) for i_src_part in range(len(src_clouds))]
    return all_closest, all_closest_inv
  else:
    return all_closest

def _get_shifted_clouds(parts_per_dom, location, comm):
  """ Wrapts get_point_cloud and shift the LNGN according
  to the domain """
  n_dom = len(parts_per_dom)
  clouds = []
  offset = np.zeros(n_dom+1, dtype=pdm_gnum_dtype)
  for i_domain, part_zones in enumerate(parts_per_dom):
    clouds_domain = [get_point_cloud(part, location) for part in part_zones]
    dom_max = par_utils.arrays_max([cloud[1] for cloud in clouds_domain], comm)
    offset[i_domain+1] = offset[i_domain] + dom_max
    # Shift (with copy for this one -- only one array)
    clouds_domain = [(coords, lngn + offset[i_domain]) for (coords, lngn) in clouds_domain]
    clouds.extend(clouds_domain)
  return offset, clouds


def _find_closest_points(src_parts_per_dom, tgt_parts_per_dom, src_location, tgt_location, comm, reverse=False):
  n_dom_src = len(src_parts_per_dom)
  n_dom_tgt = len(tgt_parts_per_dom)

  n_part_per_dom_src = [len(parts) for parts in src_parts_per_dom]
  n_part_per_dom_tgt = [len(parts) for parts in tgt_parts_per_dom]
  n_part_src = sum(n_part_per_dom_src)
  n_part_tgt = sum(n_part_per_dom_tgt)

  # > Setup source
  src_offset, src_clouds = _get_shifted_clouds(src_parts_per_dom, src_location, comm)

  # > Setup target
  tgt_offset, tgt_clouds = _get_shifted_clouds(tgt_parts_per_dom, tgt_location, comm)

  result = _closest_points(src_clouds, tgt_clouds, comm, 1, reverse)

  # Shift back result
  direct_result = result[0] if reverse else result
  for tgt_result in direct_result:
    gnum = tgt_result['closest_src_gnum']
    dom_id = np.searchsorted(src_offset, gnum)
    gnum -= src_offset[dom_id-1]
    tgt_result['domain'] = dom_id.astype(np.int32)
  if reverse:
    for src_result in result[1]:
      gnum = src_result['tgt_in_src']
      dom_id = np.searchsorted(tgt_offset, gnum)
      gnum -= tgt_offset[dom_id-1]
      src_result['domain'] = dom_id.astype(np.int32)
  # Reshape output to list of lists (as input domains)
  if reverse:
    return py_utils.to_nested_list(result[0], n_part_per_dom_tgt),\
           py_utils.to_nested_list(result[1], n_part_per_dom_src) 
  else:
    return py_utils.to_nested_list(result, n_part_per_dom_tgt)

def find_closest_points(src_tree, tgt_tree, location, comm):
  """Find the closest points between two partitioned trees.

  For all points of the target tree matching the given location,
  search the closest point of same location in the source tree.
  The result, i.e. the gnum & domain number of the source point, are stored in a ``DiscreteData_t``
  container called "ClosestPoint" on the target zones.
  The ids of source points refers to cells or vertices depending on the chosen location.

  Args:
    src_tree (CGNSTree): Source tree, partitionned
    tgt_tree (CGNSTree): Target tree, partitionned
    location ({'CellCenter', 'Vertex'}) : Entity to use to compute closest points
    comm       (MPIComm): MPI communicator

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #find_closest_points@start
        :end-before: #find_closest_points@end
        :dedent: 2
  """
  _src_parts_per_dom = get_parts_per_blocks(src_tree, comm)
  src_parts_per_dom = list(_src_parts_per_dom.values())
  tgt_parts_per_dom = list(get_parts_per_blocks(tgt_tree, comm).values())

  closest_data = _find_closest_points(src_parts_per_dom, tgt_parts_per_dom, location, location, comm)

  dom_list = '\n'.join(_src_parts_per_dom.keys())
  for i_dom, tgt_parts in enumerate(tgt_parts_per_dom):
    for i_part, tgt_part in enumerate(tgt_parts):
      shape = PT.Zone.CellSize(tgt_part) if location == 'CellCenter' else PT.Zone.VertexSize(tgt_part)
      data = closest_data[i_dom][i_part]
      sol = PT.update_child(tgt_part, "ClosestPoint", "DiscreteData_t")
      PT.new_GridLocation(location, sol)
      PT.new_DataArray("SrcId", data['closest_src_gnum'].reshape(shape, order='F'), parent=sol)
      PT.new_DataArray("DomId", data['domain'].reshape(shape, order='F'), parent=sol)
      PT.new_node("DomainList", "Descriptor_t", dom_list, parent=sol)
