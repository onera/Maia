import Converter.Internal as I
import maia.sids.sids as SIDS
# from .index_array         import create_index_array_filter
from .point_list          import create_point_list_filter
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
        create_point_list_filter(bc, bc_path, "PointList", distrib_bc, hdf_filter)

      bc_shape = None
      pr_n = I.getNodeFromName1(bc, 'PointRange')
      if pr_n is not None:
        bc_shape = SIDS.point_range_size(pr_n)

      for bcds in I.getNodesFromType1(bc, "BCDataSet_t"):
        bcds_path = bc_path + "/" + bcds[0]
        distrib_bcds_n = I.getNodeFromName1(bcds, ':CGNS#Distribution')

        if distrib_bcds_n is None: #BCDS uses BC distribution
          distrib_data = distrib_bc
          data_shape   = bc_shape
        else: #BCDS has its own distribution, we also have to check if a PointRange is given
          distrib_data = I.getNodeFromName1(distrib_bcds_n, 'Distribution')[1]
          data_shape   = None
          pr_n = I.getNodeFromName1(bcds, 'PointRange')
          if pr_n is not None:
            data_shape = SIDS.point_range_size(pr_n)

        if data_shape is not None: #Structured, we have 3d arrays
          for bcdata in I.getNodesFromType1(bcds, 'BCData_t'):
            bcdata_path = bcds_path + "/" + bcdata[0]
            create_data_array_filter(bcdata, bcdata_path, distrib_data, hdf_filter, data_shape)
        else: #Unstructured, we have fake 1d array shaped (1,N)
          for bcdata in I.getNodesFromType1(bcds, 'BCData_t'):
            bcdata_path = bcds_path + "/" + bcdata[0]
            for data in I.getNodesFromType1(bcdata,  'DataArray_t'):
              create_point_list_filter(bcdata, bcdata_path, data[0], distrib_data, hdf_filter)

