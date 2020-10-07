import Converter.PyTree   as C
import Converter.Internal as I

def compute_distribution_bc(bc, comm):
  """
  """
  pass


def compute_distribution_grid_connectivity(join, comm):
  """
  """
  distrib_ud_n = I.createUniqueChild(join, ':CGNS#Distribution', 'UserDefinedData_t')

  grid_location_n = I.getNodeFromType1(join, 'GridLocation_t')
  grid_location   = grid_location_n[1].to_string()

  if(grid_location == 'FaceCenter'):
    distrib_jn   = NPY.zeros(3, order='C', dtype='int32') # TODO remove int32
    I.newDataArray('distribution_face', distrib_jn, parent=distrib_ud_n)
  elif(grid_location == 'Vertex'):
    distrib_jn   = NPY.zeros(3, order='C', dtype='int32') # TODO remove int32
    I.newDataArray('distribution_vtx', distrib_jn, parent=distrib_ud_n)
