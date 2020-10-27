import Converter.Internal as I

def create_index_array_filter(cgns_node, cgns_path, distrib, hdf_filter):
  """
  TODO : Structured grid
  """
  for index_array in I.getNodesFromType1(cgns_node, 'IndexArray_t'):
    if(index_array[1] is not None): # Prevent reload
      continue

    dn_da    = distrib[1] - distrib[0]
    DSMMRYDA = [[0         ], [1], [dn_da], [1]]
    DSFILEDA = [[distrib[0]], [1], [dn_da], [1]]
    DSGLOBDA = [[distrib[2]]]
    DSFORMDA = [[0]]

    path = cgns_path+"/"+index_array[0]
    hdf_filter[path] = DSMMRYDA + DSFILEDA + DSGLOBDA + DSFORMDA
