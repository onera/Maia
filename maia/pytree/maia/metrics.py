import sys
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

def dtree_nbytes(tree):
  """Compute the approximate size (in bytes) of a distributed tree.

  Size is returned as a 3-tuple providing
  - metadata size (size of names, labels and pytree structure ~= everything but node values)
  - global data size (size of undistributed data; distribution arrays [start,end,tot] also counted here)
  - distributed data size (local size of distributed arrays)
  """
  class size_recorder:
    def __init__(self):
      self.meta_size = 0
      self.glob_size = 0
      self.dist_size = 0

    def pre(self, parent, node):
      self.meta_size += sys.getsizeof(node) + sys.getsizeof(node[0]) + sys.getsizeof(node[2]) + sys.getsizeof(node[3])
      if node[1] is not None:
        if _is_distributed(node, parent):
          self.dist_size += node[1].nbytes
        else:
          self.glob_size += node[1].nbytes

  v = size_recorder()
  PT.graph.cgns.depth_first_search(tree, v, depth='parent')

  return (v.meta_size, v.glob_size, v.dist_size)

