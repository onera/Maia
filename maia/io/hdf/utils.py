import numpy as np

import Converter.Internal as I
import maia.pytree.sids   as sids
import maia.pytree.maia   as MT

def pl_or_pr_size(node):
  """
  For a given node, search for a PointList of a PointRange children and return the
  size of the corresponding region:
   - directly from the value of PointRange for the PointRange case
   - using PointList#Size, if existing, or the :CGNS#Distribution otherwise for the
     PointList case.
  If both node are present, size of PointList is returned (but this should not happens!)
  Raises if both are absent.
  """
  patch = sids.Subset.getPatch(node)
  if I.getType(patch) == 'IndexArray_t':
    try:
      return I.getNodeFromName1(node, 'PointList#Size')[1]
    except TypeError: #No PL#Size, try to get info from :CGNS#Distribution and suppose size = 1,N
      distri = I.getVal(MT.getDistribution(node, 'Index'))
      return np.array([1, distri[2]])
  elif I.getType(patch) == 'IndexRange_t':
    return sids.PointRange.SizePerIndex(patch)

def apply_dataspace_to_arrays(node, node_path, data_space, hdf_filter):
  """
  Fill the hdf_filter with the specified data_space for all the DataArray_t nodes
  below the parent node node
  """
  for data_array in I.getNodesFromType1(node, 'DataArray_t'):
    path = node_path+"/"+data_array[0]
    hdf_filter[path] = data_space

def apply_dataspace_to_pointlist(node, node_path, data_space, hdf_filter):
  """
  Fill the hdf_filter with the specified data_space for PointList and PointListDonor nodes
  (if existing) below the parent node node
  """
  if I.getNodeFromName1(node, 'PointList') is not None:
    hdf_filter[node_path + "/PointList"] = data_space
  if I.getNodeFromName1(node, 'PointListDonor') is not None:
    hdf_filter[node_path + "/PointListDonor"] = data_space


