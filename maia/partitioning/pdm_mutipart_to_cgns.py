import Converter.Internal as I
import numpy              as np

from .split_U.pdm_part_to_cgns_zone import pdm_part_to_cgns_zone
from .utils                 import compute_idx_from_color

def pdm_mutipart_to_cgns(multi_part, dist_tree, n_part_per_zone, part_base, comm):
  """
  """
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  part_dims_list = []
  part_data_list = []
  zoneg_id = 0
  for dist_zone in I.getZones(dist_tree):
    # zoneg_id = I.getNodeFromName1(dist_zone, ':CGNS#Registry')[1][0] - 1
    for i_part in range(n_part_per_zone[zoneg_id]):
      part_dims_list.append(multi_part.multipart_dim_get(i_part, zoneg_id))
      #Concatenate all dicts
      all_data_dict = {**multi_part.multipart_val_get               (i_part, zoneg_id),
                       **multi_part.multipart_graph_comm_vtx_val_get(i_part, zoneg_id),
                       **multi_part.multipart_ghost_information_get (i_part, zoneg_id)}
      part_data_list.append(all_data_dict)
      # print "Got part #{0} on global zone #{1}".format(i_part, zoneg_id+1)
    zoneg_id += 1

  part_path_nodes = I.createNode(':Ppart#ZonePaths',
                                 'UserDefinedData_t',
                                 parent=part_base)

  index    = 0
  zoneg_id = 0
  for dist_zone in I.getZones(dist_tree):
    # zoneg_id = I.getNodeFromName1(dist_zone, ':CGNS#Registry')[1][0] - 1
    # > TODO : join
    # add_paths_to_ghost_zone(dist_zone, part_path_nodes)

    n_parts = n_part_per_zone[zoneg_id]
    l_dims  = part_dims_list[index:index+n_parts]
    l_data  = part_data_list[index:index+n_parts]

    parts = pdm_part_to_cgns_zone(dist_zone, l_dims, l_data, comm)

    index    += n_parts
    zoneg_id += 1

    for part in parts:
      I._addChild(part_base, part)

