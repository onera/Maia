import maia.pytree as PT

from maia.utils import py_utils

from maia.algo.dist.localize import get_point_cloud

from maia.algo.part.closest_points import _mdom_closest_points as _mdom_closest_points_part

def _mdom_closest_points(src_clouds, tgt_clouds, comm, reverse):

  # Add a level in list to mimic partitions
  tgt_clouds_per_dom = [[c] for c in tgt_clouds]
  src_clouds_per_dom = [[c] for c in src_clouds]

  result = _mdom_closest_points_part(src_clouds_per_dom, tgt_clouds_per_dom, comm, reverse)

  # Remove intermediate level
  if reverse:
    return py_utils.to_flat_list(result[0]), py_utils.to_flat_list(result[1])
  else:
    return py_utils.to_flat_list(result)


def _find_closest_points(src_dom, tgt_dom, src_location, tgt_location, comm, reverse=False):

  src_clouds = [get_point_cloud(zone, comm, src_location) for zone in src_dom]
  tgt_clouds = [get_point_cloud(zone, comm, tgt_location) for zone in tgt_dom]
  return _mdom_closest_points(src_clouds, tgt_clouds, comm, reverse)


def find_closest_points(src_tree, tgt_tree, location, comm):
  """
  Distributed implementation of maia.algo.find_closest_points
  """

  src_dom = PT.get_children_from_predicates(src_tree, 'CGNSBase_t/Zone_t')
  tgt_dom = PT.get_children_from_predicates(tgt_tree, 'CGNSBase_t/Zone_t')

  closest_data = _find_closest_points(src_dom, tgt_dom, location, location, comm)

  dom_list = '\n'.join(PT.predicates_to_paths(src_tree, 'CGNSBase_t/Zone_t'))

  for i_dom, tgt_part in enumerate(tgt_dom):
    data = closest_data[i_dom]
    sol = PT.update_child(tgt_part, "ClosestPoint", "DiscreteData_t")
    PT.new_GridLocation(location, sol)
    PT.new_DataArray("SrcId", data['closest_src_gnum'], parent=sol)
    PT.new_DataArray("DomId", data['domain'], parent=sol)
    PT.new_node("DomainList", "Descriptor_t", dom_list, parent=sol)
