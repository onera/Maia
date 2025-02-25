import numpy as np

import Pypdm.Pypdm as PDM

import maia.pytree        as PT

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype

from maia.utils                  import py_utils, np_utils, par_utils
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

def _mdom_closest_points(src_clouds_per_dom, tgt_clouds_per_dom, comm, reverse):

  n_clouds_per_dom_src = [len(parts) for parts in src_clouds_per_dom]
  n_clouds_per_dom_tgt = [len(parts) for parts in tgt_clouds_per_dom]

  # Shift data; we dont do it inplace since src and target data may share the same memory
  src_offset = np.zeros(len(src_clouds_per_dom)+1, dtype=pdm_gnum_dtype)
  tgt_offset = np.zeros(len(tgt_clouds_per_dom)+1, dtype=pdm_gnum_dtype)
  for i_domain, clouds in enumerate(src_clouds_per_dom):
    # Compute global offsets for this domain
    dom_max = par_utils.arrays_max([cloud[1] for cloud in clouds], comm)
    src_offset[i_domain+1] = src_offset[i_domain] + dom_max
    # Shift source arrays (copy)
    src_clouds_per_dom[i_domain] = [(c[0], c[1] + src_offset[i_domain]) for c in clouds]
  for i_domain, clouds in enumerate(tgt_clouds_per_dom):
    # Compute global offsets for this domain
    dom_max = par_utils.arrays_max([cloud[1] for cloud in clouds], comm)
    tgt_offset[i_domain+1] = tgt_offset[i_domain] + dom_max
    # Shift source arrays (copy)
    tgt_clouds_per_dom[i_domain] = [(c[0], c[1] + tgt_offset[i_domain]) for c in clouds]

  tgt_clouds = py_utils.to_flat_list(tgt_clouds_per_dom)
  src_clouds = py_utils.to_flat_list(src_clouds_per_dom)

  result = _closest_points(src_clouds, tgt_clouds, comm, 1, reverse)

  # Shift back result
  direct_result = result[0] if reverse else result
  for tgt_result in direct_result:
    gnum_shifted = tgt_result.pop('closest_src_gnum')
    tgt_result['closest_src_gnum'], tgt_result['domain'] = np_utils.shifted_to_local(gnum_shifted, src_offset)
  if reverse:
    for src_result in result[1]:
      gnum_shifted = src_result.pop('tgt_in_src')
      src_result['tgt_in_src'], src_result['domain'] = np_utils.shifted_to_local(gnum_shifted, tgt_offset)

  # Reshape output to list of lists (as input domains)
  if reverse:
    return py_utils.to_nested_list(result[0], n_clouds_per_dom_tgt),\
           py_utils.to_nested_list(result[1], n_clouds_per_dom_src) 
  else:
    return py_utils.to_nested_list(result, n_clouds_per_dom_tgt)

def _find_closest_points(src_parts_per_dom, tgt_parts_per_dom, src_location, tgt_location, comm, reverse=False):

  src_clouds = [[get_point_cloud(part, src_location) for part in src_parts] \
          for src_parts in src_parts_per_dom]
  tgt_clouds = [[get_point_cloud(part, tgt_location) for part in tgt_parts] \
          for tgt_parts in tgt_parts_per_dom]

  return _mdom_closest_points(src_clouds, tgt_clouds, comm, reverse)


def find_closest_points(src_tree, tgt_tree, location, comm):
  """
  Partitionned implementation of maia.algo.find_closest_points
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
