import maia.sids.sids as SIDS
import Converter.Internal as I
import maia.sids.Internal_ext as IE
import numpy as np

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
  pl_n = I.getNodeFromName1(node, 'PointList')
  pr_n = I.getNodeFromName1(node, 'PointRange')
  assert not(pl_n is None and pr_n is None)
  if pl_n:
    try:
      return I.getNodeFromName1(node, 'PointList#Size')[1]
    except TypeError: #No PL#Size, try to get info from :CGNS#Distribution and suppose size = 1,N
      distri = I.getVal(IE.getDistribution(node, 'Index'))
      return np.array([1, distri[2]])
  if pr_n:
    return SIDS.PointRange.VertexSize(pr_n)

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


