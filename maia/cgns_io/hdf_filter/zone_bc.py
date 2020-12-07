import Converter.Internal as I
import maia.sids.sids as SIDS
# from .index_array         import create_index_array_filter
from .data_array          import create_data_array_filter

def create_zone_bc_filter(zone, zone_path, hdf_filter):
  """
  """
  for zone_bc in I.getNodesFromType1(zone, 'ZoneBC_t'):
    zone_bc_path = zone_path+"/"+zone_bc[0]
    for bc in I.getNodesFromType1(zone_bc, 'BC_t'):
      bc_path = zone_bc_path+"/"+bc[0]

      distrib_bc_n = I.getNodeFromName1(bc          , ':CGNS#Distribution')
      distrib_bc   = I.getNodeFromName1(distrib_bc_n, 'Distribution')[1]

      if I.getNodeFromName1(bc, 'PointList') is not None:
        bc_shape = I.getNodeFromName1(bc, 'PointList#Size')[1]
        data_space = create_data_array_filter(distrib_bc, bc_shape)
        hdf_filter[bc_path + "/PointList"] = data_space

      pr_n = I.getNodeFromName1(bc, 'PointRange')
      if pr_n is not None:
        bc_shape = SIDS.point_range_size(pr_n)

      for bcds in I.getNodesFromType1(bc, "BCDataSet_t"):
        bcds_path = bc_path + "/" + bcds[0]
        distrib_bcds_n = I.getNodeFromName1(bcds, ':CGNS#Distribution')

        if distrib_bcds_n is None: #BCDS uses BC distribution
          distrib_data = distrib_bc
          data_shape   = bc_shape
        else: #BCDS has its own distribution
          distrib_data = I.getNodeFromName1(distrib_bcds_n, 'Distribution')[1]
          if I.getNodeFromName1(bcds, 'PointList') is not None:
            data_shape = I.getNodeFromName1(bcds, 'PointList#Size')[1]
          pr_n = I.getNodeFromName1(bcds, 'PointRange')
          if pr_n is not None:
            data_shape = SIDS.point_range_size(pr_n)

        data_space = create_data_array_filter(distrib_data, data_shape)
        if distrib_bcds_n is not None and pr_n is None:
          hdf_filter[bcds_path + "/PointList"] = data_space

        for bcdata in I.getNodesFromType1(bcds, 'BCData_t'):
          bcdata_path = bcds_path + "/" + bcdata[0]
          for data_array in I.getNodesFromType1(bcdata, 'DataArray_t'):
            path = bcdata_path+"/"+data_array[0]
            hdf_filter[path] = data_space

