import Converter.Internal as I
import numpy              as np

from .split_U.pdm_part_to_cgns_zone import pdm_part_to_cgns_zone
from .utils                 import compute_idx_from_color

def pdm_mutipart_to_cgns(multi_part, d_u_zones, n_part_per_zone, part_base, comm):
  """
  """
  concat_pdm_data = lambda i_part, i_zone : {**multi_part.multipart_val_get               (i_part, i_zone),
                                             **multi_part.multipart_graph_comm_vtx_val_get(i_part, i_zone),
                                             **multi_part.multipart_ghost_information_get (i_part, i_zone)}

  part_path_nodes = I.createNode(':Ppart#ZonePaths', 'UserDefinedData_t', parent=part_base)

  for i_zone, d_zone in enumerate(d_u_zones):
    # > TODO : join
    # add_paths_to_ghost_zone(d_zone, part_path_nodes)

    n_part = n_part_per_zone[i_zone]
    l_dims = [multi_part.multipart_dim_get(i_part, i_zone) for i_part in range(n_part)]
    l_data = [concat_pdm_data(i_part, i_zone)              for i_part in range(n_part)]

    parts = pdm_part_to_cgns_zone(d_zone, l_dims, l_data, comm)

    for part in parts:
      I._addChild(part_base, part)

