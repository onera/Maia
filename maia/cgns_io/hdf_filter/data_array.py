import Converter.Internal as I

def create_data_array_filter(cgns_node, cgns_path, distrib):
  """
  TODO : Structured grid
  """
  for data_array in I.getNodesFromType1(cgns_node, 'DataArray_t'):

    dn_da    = distrib[1] - distrib[0]
    DSMMRYDA = [[0         ], [1], [dn_da], [1]]
    DSFILEDA = [[distrib[0]], [1], [dn_da], [1]]
    DSGLOBDA = [[distrib[2]]]
    DSFORMDA = [[0]]

    path = cgns_path+"/"+data_array[0]
    hdf_filter[path] = DSMMRYDA + DSFILEDA + DSGLOBDA + DSFORMDA
