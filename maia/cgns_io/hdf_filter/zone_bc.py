import Converter.Internal as I
from .              import utils
from .hdf_dataspace import create_data_array_filter

def create_zone_bc_filter(zone, zone_path, hdf_filter):
  """
  """
  for zone_bc in I.getNodesFromType1(zone, 'ZoneBC_t'):
    zone_bc_path = zone_path+"/"+zone_bc[0]
    for bc in I.getNodesFromType1(zone_bc, 'BC_t'):
      bc_path = zone_bc_path+"/"+bc[0]

      distrib_bc_n = I.getNodeFromName1(bc          , ':CGNS#Distribution')
      distrib_bc   = I.getNodeFromName1(distrib_bc_n, 'Distribution')[1]

      bc_shape = utils.pl_or_pr_size(bc)
      data_space = create_data_array_filter(distrib_bc, bc_shape)
      utils.apply_dataspace_to_pointlist(bc, bc_path, data_space, hdf_filter)


      for bcds in I.getNodesFromType1(bc, "BCDataSet_t"):
        bcds_path = bc_path + "/" + bcds[0]
        distrib_bcds_n = I.getNodeFromName1(bcds, ':CGNS#Distribution')

        if distrib_bcds_n is None: #BCDS uses BC distribution
          distrib_data = distrib_bc
          data_shape   = bc_shape
        else: #BCDS has its own distribution
          distrib_data = I.getNodeFromName1(distrib_bcds_n, 'Distribution')[1]
          data_shape = utils.pl_or_pr_size(bcds)

        data_space = create_data_array_filter(distrib_data, data_shape)
        utils.apply_dataspace_to_pointlist(bcds, bcds_path, data_space, hdf_filter)
        for bcdata in I.getNodesFromType1(bcds, 'BCData_t'):
          bcdata_path = bcds_path + "/" + bcdata[0]
          utils.apply_dataspace_to_arrays(bcdata, bcdata_path, data_space, hdf_filter)

