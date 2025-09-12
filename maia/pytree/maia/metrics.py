import sys
from maia.pytree.typing import *
import maia.pytree as PT

def _is_distributed(node, parent):
  if node[3] == 'IndexArray_t':
    return True
  if node[3] != 'DataArray_t':
    return False
  if parent[3] in ['GridCoordinates_t', 'Elements_t', \
          'FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t', 'BCData_t']:
    return True
  if node[0] in ['PointList', 'PointListDonor']:
    return True

def dtree_nbytes(tree:CGNSTree) -> Tuple[int,int,int]:
  """Compute the approximate size (in bytes) of a distributed tree.

  Size is returned as a 3-tuple providing
  - metadata size (size of names, labels and pytree structure ~= everything but node values)
  - global data size (size of undistributed data; distribution arrays [start,end,tot] also counted here)
  - distributed data size (local size of distributed arrays)
  """
  
  sizes = {key:0 for key in ['meta', 'glob', 'dist']}

  def size_recorder(nodes):
    node = nodes[-1]
    sizes['meta'] += sys.getsizeof(node) + sys.getsizeof(node[0]) + sys.getsizeof(node[2]) + sys.getsizeof(node[3])
    if node[1] is not None:
      parent = nodes[-2]
      if _is_distributed(node, parent):
        sizes['dist'] += node[1].nbytes
      else:
        sizes['glob'] += node[1].nbytes

  PT.scan(tree, size_recorder, ancestors=True)

  return (sizes['meta'], sizes['glob'], sizes['dist'])

