import Converter.Internal as I
import numpy              as np
import copy               as CPY

from .pdm_part_to_cgns_zone import pdm_part_to_cgns_zone
from .utils                 import compute_idx_from_color

def pdm_mutipart_to_cgns(multi_part, dist_tree, n_part_per_zone, comm):
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
      part_data_list.append(multi_part.multipart_val_get(i_part, zoneg_id))
      # graph comm
      tmp = multi_part.multipart_graph_comm_vtx_val_get(i_part, zoneg_id)
      for fld in tmp:
        part_data_list[-1][fld] = tmp[fld]
      tmp = multi_part.multipart_ghost_information_get(i_part, zoneg_id)
      for fld in tmp:
        part_data_list[-1][fld] = tmp[fld]
      # print "Got part #{0} on global zone #{1}".format(i_part, zoneg_id+1)
    zoneg_id += 1

  dist_base = I.getNodeFromType1(dist_tree, 'CGNSBase_t')
  base_name = dist_base[0]
  part_tree = I.newCGNSTree()
  part_base = I.newCGNSBase(base_name, 3, 3, parent=part_tree)
  part_path_nodes = I.createNode(':Ppart#ZonePaths',
                                 'UserDefinedData_t',
                                 parent=part_base)

  for fam in I.getNodesFromType1(dist_base, 'Family_t'):
    I.addChild(part_base, fam)

  index    = 0
  zoneg_id = 0
  for dist_zone in I.getZones(dist_tree):
    # zoneg_id = I.getNodeFromName1(dist_zone, ':CGNS#Registry')[1][0] - 1

    # > TODO : join
    # add_paths_to_ghost_zone(dist_zone, part_path_nodes)

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    for i_part in range(n_part_per_zone[zoneg_id]):
      dims = part_dims_list[index]
      data = part_data_list[index]

      part_zone = I.newZone(name   = dist_zone[0]+'.P{0}.N{1}'.format(i_rank, i_part),
                            zsize  = [[dims['n_vtx'],dims['n_cell'],0]],
                            ztype  = 'Unstructured',
                            parent = part_base)

      pdm_part_to_cgns_zone(part_zone, dist_zone, dims, data, comm)

      if(data['np_vtx_ghost_information'] is not None):
        vtx_ghost_info = data['np_vtx_ghost_information']
        first_ghost_idx = np.searchsorted(vtx_ghost_info, 2) 
        n_ghost_node = len(vtx_ghost_info) - first_ghost_idx
        coord_node = I.getNodeFromName(part_zone,"GridCoordinates")

        I.newRind([0,n_ghost_node],coord_node)


      index    += 1

    zoneg_id += 1

  return part_tree

