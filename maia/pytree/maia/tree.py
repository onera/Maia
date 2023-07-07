import numpy as np

import maia.pytree      as PT

from maia.transfer import protocols as EP

from maia.utils          import np_utils, py_utils
from maia.utils.parallel import algo as par_algo

# Note : these two will probably go elsewhere in maia or directly in PDM
def _encode(strings):
  bstrings = [s.encode() for s in strings]
  stride = np.array([len(bs) for bs in bstrings], np.int32)

  stride_idx = np_utils.sizes_to_indices(stride)
  buff = np.empty(stride_idx[-1], np.int8)
  for i, bs in enumerate(bstrings):
    for j in range(len(bs)):
      buff[stride_idx[i]+j] = bs[j]

  return stride, buff

def _decode(stride, buff):
  stride_idx = np_utils.sizes_to_indices(stride)
  return [bytes(buff[stride_idx[i]:stride_idx[i+1]]).decode() for i in range(stride.size)]



def rename_zones(part_tree, old_to_new_path, comm):
  """ Rename the zones in a partitioned context.

  This mainly consists in sending the new names to the other ranks in
  ordre to have them renaming their GridConnectivity_t nodes

  New names must be a list of size nb. zones in the parttree, giving the
  new path of each zone (note that base name is not allowed to change)
  """

  zones_path_ini = list(old_to_new_path.keys())
  new_names = list(old_to_new_path.values())

  PT.enforceDonorAsPath(part_tree)
  is_gc = lambda n : PT.get_label(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t']
  gc_predicates = ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', is_gc]
  gcs = PT.get_children_from_predicates(part_tree, gc_predicates)
  zones_path_wanted = [PT.get_value(gc) for gc in gcs]

  zone_gnum = par_algo.compute_gnum(zones_path_ini + zones_path_wanted, comm)
  cur_zone_gnum, wanted_zone_gnum = py_utils.to_nested_list(zone_gnum, (len(zones_path_ini), len(zones_path_wanted)))
  
  send_stride, encoded_names = _encode(new_names)
  recv_stride, recv_encoded_names = EP.part_to_part_strided([send_stride], [encoded_names], [cur_zone_gnum], [wanted_zone_gnum], comm)
  recv_names = _decode(recv_stride[0], recv_encoded_names[0]) #0 because only one part

  # Update tree
  for i, path in enumerate(zones_path_ini):
    zone = PT.get_node_from_path(part_tree, path)
    PT.set_name(zone, PT.path_tail(new_names[i]))
  for gc, new_name in zip(gcs, recv_names):
    PT.set_value(gc, new_name)

