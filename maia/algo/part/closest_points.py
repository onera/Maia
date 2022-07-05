import numpy as np

import Pypdm.Pypdm as PDM

import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils                  import py_utils
from maia.transfer               import utils as te_utils
from maia.factory.dist_from_part import discover_nodes_from_matching

from .point_cloud_utils import get_point_cloud


def _closest_points(src_clouds, tgt_clouds, comm, reverse=False):
  # For now, only 1 domain is supported so we expect source and target clouds
  # as flat lists of tuples (coords, lngn)

  # > Create and setup global data
  closest_point = PDM.ClosestPoints(comm, n_closest=1)
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


def _find_closest_points(src_parts_per_dom, tgt_parts_per_dom, src_location, tgt_location, comm, reverse=False):
  n_dom_src = len(src_parts_per_dom)
  n_dom_tgt = len(tgt_parts_per_dom)

  assert n_dom_src == n_dom_tgt == 1
  n_part_per_dom_src = [len(parts) for parts in src_parts_per_dom]
  n_part_per_dom_tgt = [len(parts) for parts in tgt_parts_per_dom]
  n_part_src = sum(n_part_per_dom_src)
  n_part_tgt = sum(n_part_per_dom_tgt)

  # > Setup source
  src_clouds = []
  for i_domain, src_part_zones in enumerate(src_parts_per_dom):
    for i_part, src_part in enumerate(src_part_zones):
      src_clouds.append(get_point_cloud(src_part, src_location))

  # > Setup target
  tgt_clouds = []
  for i_domain, tgt_part_zones in enumerate(tgt_parts_per_dom):
    for i_part, tgt_part in enumerate(tgt_part_zones):
      tgt_clouds.append(get_point_cloud(tgt_part, tgt_location))

  result = _closest_points(src_clouds, tgt_clouds, comm, reverse)

  # Reshape output to list of lists (as input domains)
  if reverse:
    return py_utils.to_nested_list(result[0], n_part_per_dom_tgt),\
           py_utils.to_nested_list(result[1], n_part_per_dom_src) 
  else:
    return py_utils.to_nested_list(result, n_part_per_dom_tgt)

def find_closest_points(src_tree, tgt_tree, location, comm):
  """Find the closest points between two partitioned trees.

  For all the points of the target tree matching the given location,
  search the closest point of same location in the source tree.
  The result, i.e. the gnum of the source point, is stored in a DiscreteData_t
  container called "ClosestPoint" on the target zones.
  The ids of source points refers to cells or vertices depending on the chosen location.

  Partitions must come from a single initial domain on both source and target tree.

  Args:
    src_tree (CGNSTree): Source tree, partitionned
    tgt_tree (CGNSTree): Target tree, partitionned
    location ({'CellCenter', 'Vertex'}) : Entity to use to compute closest points
    comm       (MPIComm): MPI communicator
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

  closest_data = _find_closest_points(src_parts_per_dom, tgt_parts_per_dom, location, location, comm)
  for i_dom, tgt_parts in enumerate(tgt_parts_per_dom):
    for i_part, tgt_part in enumerate(tgt_parts):
      shape = PT.Zone.CellSize(tgt_part) if location == 'CellCenter' else PT.Zone.VertexSize(tgt_part)
      data = closest_data[i_dom][i_part]
      sol = I.createUniqueChild(tgt_part, "ClosestPoint", "DiscreteData_t")
      I.newGridLocation(location, sol)
      I.newDataArray("SrcId", data['closest_src_gnum'].reshape(shape, order='F'), parent=sol)
