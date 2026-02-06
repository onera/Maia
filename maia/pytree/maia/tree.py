import numpy as np

import maia.pytree       as PT

from maia.pytree.typing import *
from maia.typing        import *

from maia.utils import np_utils, py_utils

__all__ = ['rename_zones']

# Note : these two will probably go elsewhere in maia or directly in PDM
def _encode(strings:List[str]) -> Tuple[NDArray[np.int32], NDArray[np.int8]]:
  bstrings = [s.encode() for s in strings]
  stride = np.array([len(bs) for bs in bstrings], np.int32)

  stride_idx = np_utils.sizes_to_indices(stride)
  buff = np.empty(stride_idx[-1], np.int8) # np.int8 because bytes characters are 1-byte long
  for i, bs in enumerate(bstrings):
    for j in range(len(bs)):
      buff[stride_idx[i]+j] = bs[j]

  return stride, buff

def _decode(stride:NDArray[np.int32], buff:NDArray[np.int8]) -> List[str]:
  stride_idx = np_utils.sizes_to_indices(stride)
  return [bytes(buff[stride_idx[i]:stride_idx[i+1]]).decode() for i in range(stride.size)]


def rename_zones(part_tree:CGNSPartTree, old_to_new_path:Mapping[str,str], comm:MPIComm) -> None: 
  """ Rename the Zone_t nodes of a partitioned tree.

  New names are provided through ``old_to_new`` parameter, which maps zone pathes
  to their new value. Note that:

  - the parent base of each zone is not allowed to change,
  - **you are responsible** of preserving maia :ref:`naming_conv`,
  - zones that keep their original names may be omitted from mapping.

  The GridConnectivity_t values are updated by this function.

  Args:
    part_tree (CGNSPartTree) : Partitioned tree, starting at top level  
    old_to_new_path (dict) : azezae
    comm (MPIComm) : MPI communicator
  Example:
    >>> part_tree = PT.yaml.to_cgns_tree('''
    ... Base CGNSBase_t:
    ...   ZoneA.P2.N0 Zone_t:
    ...   ZoneA.P2.N1 Zone_t:
    ...     ZoneGridConnectivity ZoneGridConnectivity_t:
    ...       match GridConnectivity_t "ZoneB.P2.N0":
    ...   ZoneB.P2.N0 Zone_t:
    ... ''')
    >>> pzones = MT.rename_zones(part_tree, {'Base/ZoneB.P2.N0' : 'Base/ZoneC.P2.N0'}, comm)
    >>> PT.print_tree(PT.get_child_from_label(part_tree, 'CGNSBase_t'))
    Base CGNSBase_t 
    ├───ZoneA.P2.N0 Zone_t 
    ├───ZoneA.P2.N1 Zone_t 
    │   └───ZoneGridConnectivity ZoneGridConnectivity_t 
    │       └───match GridConnectivity_t "Base/ZoneC.P2.N0"
    └───ZoneC.P2.N0 Zone_t 
  """
  from maia.transfer import protocols as EP
  from maia.utils.parallel import algo as par_algo

  zones_path_ini = list(old_to_new_path.keys())
  new_names = list(old_to_new_path.values())

  PT.enforceDonorAsPath(part_tree)
  gc_predicates:Predicates = ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', PT.pred.IS_GC]
  gcs = PT.get_children_from_predicates(part_tree, gc_predicates)
  zones_path_wanted = [PT.get_str_value(gc) for gc in gcs]

  zone_gnum = par_algo.compute_gnum(zones_path_ini + zones_path_wanted, comm)
  cur_zone_gnum, wanted_zone_gnum = py_utils.to_nested_list(zone_gnum, (len(zones_path_ini), len(zones_path_wanted)))
  
  send_stride, encoded_names = _encode(new_names)
  recv_stride, recv_encoded_names = EP.part_to_part_strided([send_stride], [encoded_names], [cur_zone_gnum], [wanted_zone_gnum], comm)
  assert len(recv_stride[0]) == len(wanted_zone_gnum)
  recv_names = _decode(recv_stride[0], recv_encoded_names[0]) #0 because only one part

  # Update tree
  for i, path in enumerate(zones_path_ini):
    zone = PT.find_node_from_path(part_tree, path)
    PT.set_name(zone, PT.utils.path_tail(new_names[i]))
  for gc, new_name in zip(gcs, recv_names):
    PT.set_value(gc, new_name)

