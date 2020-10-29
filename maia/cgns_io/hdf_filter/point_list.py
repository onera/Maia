import Converter.Internal as I

def create_point_list_filter(cgns_node, cgns_path, pl_name, distrib, hdf_filter):
  """
  TODO : Structured grid
  """
  # pl_size_n = I.getNodeFromName1(cgns_node, pl_name+"#Size")
  # assert(pl_size_n[1][0] == 1)

  dn_pl    = distrib[1] - distrib[0]
  DSMMRYPL = [[0,0          ], [1, 1], [1, dn_pl], [1, 1]]
  DSFILEPL = [[0, distrib[0]], [1, 1], [1, dn_pl], [1, 1]]
  DSGLOBPL = [[1, distrib[2]]]
  DSFORMPL = [[0]]

  pl_path = cgns_path+"/"+pl_name
  hdf_filter[pl_path] = DSMMRYPL + DSFILEPL + DSGLOBPL + DSFORMPL
