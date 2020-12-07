import maia.sids.sids as SIDS
import Converter.Internal as I

def pl_or_pr_size(node):
  pl_n = I.getNodeFromName1(node, 'PointList')
  pr_n = I.getNodeFromName1(node, 'PointRange')
  assert not(pl_n is None and pr_n is None)
  if pl_n:
    try:
      return I.getNodeFromName1(node, 'PointList#Size')[1]
    except TypeError: #No PL#Size, try to get info from :CGNS#Distribution and suppose size = 1,N
      distri = I.getNodeFromPath(node, ':CGNS#Distribution/Distribution')[1]
      return [1, distri[2]]
  if pr_n:
    return SIDS.point_range_size(pr_n)

def apply_dataspace_to_arrays(node, node_path, data_space, hdf_filter):
  for data_array in I.getNodesFromType1(node, 'DataArray_t'):
    path = node_path+"/"+data_array[0]
    hdf_filter[path] = data_space

def apply_dataspace_to_pointlist(node, node_path, data_space, hdf_filter):
  if I.getNodeFromName1(node, 'PointList') is not None:
    hdf_filter[node_path + "/PointList"] = data_space
  if I.getNodeFromName1(node, 'PointListDonor') is not None:
    hdf_filter[node_path + "/PointListDonor"] = data_space
    

