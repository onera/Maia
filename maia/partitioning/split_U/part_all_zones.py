import Converter.Internal as I
import numpy              as np

import Pypdm.Pypdm        as PDM

import maia.sids.sids as SIDS
from .cgns_to_pdm_dmesh       import cgns_dist_zone_to_pdm_dmesh
from .cgns_to_pdm_dmesh_nodal import cgns_dist_zone_to_pdm_dmesh_nodal
from .pdm_part_to_cgns_zone   import pdm_part_to_cgns_zone

def prepare_part_weight(zones, n_part_per_zone, dzone_to_weighted_parts):
  part_weight = np.empty(sum(n_part_per_zone), dtype='float64')
  offset = 0
  for i_zone, zone in enumerate(zones):
    part_weight[offset:offset+n_part_per_zone[i_zone]] = dzone_to_weighted_parts[I.getName(zone)]
    offset += n_part_per_zone[i_zone]
  return part_weight

def set_mpart_reordering(multipart, reorder_options, keep_alive):
  renum_cell_method = "PDM_PART_RENUM_CELL_" + reorder_options['cell_renum_method']
  renum_face_method = "PDM_PART_RENUM_FACE_" + reorder_options['face_renum_method']
  if "CACHEBLOCKING" in reorder_options['cell_renum_method']:
    pdm_part_tool     = 1 if reorder_options['graph_part_tool'] == 'parmetis' else 2
    cacheblocking_props = np.array([reorder_options['n_cell_per_cache'],
                                    1,
                                    1,
                                    reorder_options['n_face_per_pack'],
                                    pdm_part_tool],
                                    dtype='int32', order='c')
  else:
    cacheblocking_props = None
  multipart.multipart_set_reordering(-1,
                                      renum_cell_method.encode('utf-8'),
                                      renum_face_method.encode('utf-8'),
                                      cacheblocking_props)
  keep_alive.append(cacheblocking_props)

def set_mpart_dmeshes(multi_part, u_zones, comm, keep_alive):
  for i_zone, zone in enumerate(u_zones):
    #Determine NGON or ELMT
    elmt_types = [SIDS.ElementType(elmt) for elmt in I.getNodesFromType1(zone, 'Elements_t')]
    is_ngon = 22 in elmt_types
    if is_ngon:
      dmesh    = cgns_dist_zone_to_pdm_dmesh(zone, comm)
      keep_alive.append(dmesh)
      multi_part.multipart_register_block(i_zone, dmesh)
    else:
      dmesh_nodal = cgns_dist_zone_to_pdm_dmesh_nodal(zone, comm, needs_bc=False)
      keep_alive.append(dmesh_nodal)
      multi_part.multipart_register_dmesh_nodal(i_zone, dmesh_nodal)

def collect_mpart_partitions(multi_part, d_zones, n_part_per_zone, comm, post_options):
  """
  """
  concat_pdm_data = lambda i_part, i_zone : {**multi_part.multipart_val_get               (i_part, i_zone),
                                             **multi_part.multipart_graph_comm_vtx_val_get(i_part, i_zone),
                                             **multi_part.multipart_ghost_information_get (i_part, i_zone),
                                             **multi_part.multipart_color_get             (i_part, i_zone)}

  #part_path_nodes = I.createNode(':Ppart#ZonePaths', 'UserDefinedData_t', parent=part_base)

  all_parts = list()
  for i_zone, d_zone in enumerate(d_zones):
    # > TODO : join
    # add_paths_to_ghost_zone(d_zone, part_path_nodes)

    n_part = n_part_per_zone[i_zone]
    l_dims = [multi_part.multipart_dim_get(i_part, i_zone) for i_part in range(n_part)]
    l_data = [concat_pdm_data(i_part, i_zone)              for i_part in range(n_part)]

    parts = pdm_part_to_cgns_zone(d_zone, l_dims, l_data, comm, post_options)
    all_parts.extend(parts)

  return all_parts

def part_U_zones(u_zones, dzone_to_weighted_parts, comm, part_options):

  # Careful ! Some object must be deleted at the very end of the function,
  # since they are usefull for pdm
  keep_alive = list()

  # Deduce the number of parts for each zone from dzone->weighted_parts dict
  n_part_per_zone = np.array([len(dzone_to_weighted_parts[I.getName(zone)]) for zone in u_zones],
                             dtype=np.int32)
  keep_alive.append(n_part_per_zone)

  # Init multipart object
  part_weight = prepare_part_weight(u_zones, n_part_per_zone, dzone_to_weighted_parts)

  pdm_part_tool     = 1 if part_options['graph_part_tool'] == 'parmetis' else 2
  pdm_weight_method = 2
  multi_part = PDM.MultiPart(len(u_zones), n_part_per_zone, 0, pdm_part_tool, pdm_weight_method, part_weight, comm)

  # Setup
  set_mpart_dmeshes(multi_part, u_zones, comm, keep_alive)
  set_mpart_reordering(multi_part, part_options['reordering'], keep_alive)

  #Run and return parts
  multi_part.multipart_run_ppart()

  post_options = {k:part_options[k] for k in ['part_interface_loc', 'dump_pdm_output']}
  u_parts = collect_mpart_partitions(multi_part, u_zones, n_part_per_zone, comm, post_options)

  del(multi_part) # Force multi_part object to be deleted before n_part_per_zone array
  del(keep_alive)
  for zone in u_zones:
    I._rmNodesByName1(zone, ':CGNS#MultiPart')

  return u_parts


