import numpy              as np
import Pypdm.Pypdm        as PDM

import maia.pytree        as PT

from . import cgns_to_pdm_dmesh
from .pdm_part_to_cgns_zone   import pdm_part_to_cgns_zone

from maia.utils import py_utils

maia_to_pdm_entity = {"cell"   : PDM._PDM_MESH_ENTITY_CELL,
                      "face"   : PDM._PDM_MESH_ENTITY_FACE,
                      "edge"   : PDM._PDM_MESH_ENTITY_EDGE,
                      "vtx"    : PDM._PDM_MESH_ENTITY_VERTEX}

maia_to_pdm_split_tool = {'parmetis' : PDM._PDM_SPLIT_DUAL_WITH_PARMETIS,
                          'ptscotch' : PDM._PDM_SPLIT_DUAL_WITH_PTSCOTCH,
                          'hilbert'  : PDM._PDM_SPLIT_DUAL_WITH_HILBERT,
                          'gnum'     : PDM._PDM_SPLIT_DUAL_WITH_IMPLICIT}

maia_to_pdm_connectivity = {"cell_elmt" : PDM._PDM_CONNECTIVITY_TYPE_CELL_ELMT,
                            "cell_cell" : PDM._PDM_CONNECTIVITY_TYPE_CELL_CELL,
                            "cell_face" : PDM._PDM_CONNECTIVITY_TYPE_CELL_FACE,
                            "cell_edge" : PDM._PDM_CONNECTIVITY_TYPE_CELL_EDGE,
                            "cell_vtx"  : PDM._PDM_CONNECTIVITY_TYPE_CELL_VTX,
                            "face_elmt" : PDM._PDM_CONNECTIVITY_TYPE_FACE_ELMT,
                            "face_cell" : PDM._PDM_CONNECTIVITY_TYPE_FACE_CELL,
                            "face_face" : PDM._PDM_CONNECTIVITY_TYPE_FACE_FACE,
                            "face_edge" : PDM._PDM_CONNECTIVITY_TYPE_FACE_EDGE,
                            "face_vtx"  : PDM._PDM_CONNECTIVITY_TYPE_FACE_VTX,
                            "edge_elmt" : PDM._PDM_CONNECTIVITY_TYPE_EDGE_ELMT,
                            "edge_cell" : PDM._PDM_CONNECTIVITY_TYPE_EDGE_CELL,
                            "edge_face" : PDM._PDM_CONNECTIVITY_TYPE_EDGE_FACE,
                            "edge_edge" : PDM._PDM_CONNECTIVITY_TYPE_EDGE_EDGE,
                            "edge_vtx"  : PDM._PDM_CONNECTIVITY_TYPE_EDGE_VTX,
                            "vtx_elmt"  : PDM._PDM_CONNECTIVITY_TYPE_VTX_ELMT,
                            "vtx_cell"  : PDM._PDM_CONNECTIVITY_TYPE_VTX_CELL,
                            "vtx_face"  : PDM._PDM_CONNECTIVITY_TYPE_VTX_FACE,
                            "vtx_edge"  : PDM._PDM_CONNECTIVITY_TYPE_VTX_EDGE,
                            "vtx_vtx"   : PDM._PDM_CONNECTIVITY_TYPE_VTX_VTX,
                            "elmt_cell" : PDM._PDM_CONNECTIVITY_TYPE_ELMT_CELL,
                            "elmt_face" : PDM._PDM_CONNECTIVITY_TYPE_ELMT_FACE,
                            "elmt_edge" : PDM._PDM_CONNECTIVITY_TYPE_ELMT_EDGE,
                            "elmt_vtx " : PDM._PDM_CONNECTIVITY_TYPE_ELMT_VTX}

pdm_geometry_kinds = [PDM._PDM_GEOMETRY_KIND_CORNER, PDM._PDM_GEOMETRY_KIND_RIDGE, 
                      PDM._PDM_GEOMETRY_KIND_SURFACIC, PDM._PDM_GEOMETRY_KIND_VOLUMIC]

def prepare_part_weight(bases_to_block, zone_to_weights):
  n_zones = sum([len(zones) for zones in bases_to_block.values()])
  n_parts = sum([len(weights) for weights in zone_to_weights.values()])
  n_part_per_zone = np.empty(n_zones, np.int32)
  part_weight     = np.empty(n_parts, np.float64)
  i = 0
  j = 0
  for base, zones in bases_to_block.items():
    for zone in zones:
      zone_path = f"{base}/{PT.get_name(zone)}"
      weights   = zone_to_weights.get(zone_path, [])
      n_part_per_zone[i] = len(weights)
      part_weight[j:j+n_part_per_zone[i]] = weights
      j += n_part_per_zone[i]
      i += 1
  return n_part_per_zone, part_weight

def set_mpart_reordering(multipart, reorder_options, keep_alive):
  renum_cell_method = "PDM_PART_RENUM_CELL_" + reorder_options['cell_renum_method']
  renum_face_method = "PDM_PART_RENUM_FACE_" + reorder_options['face_renum_method']
  renum_vtx_method  = "PDM_PART_RENUM_VTX_"  + reorder_options['vtx_renum_method']
  part_tool_dic = {'parmetis': 1, 'ptscotch': 2, 'hyperplane': 3}
  if "CACHEBLOCKING" in reorder_options['cell_renum_method']:
    pdm_part_tool     = part_tool_dic[reorder_options['graph_part_tool']]
    assert pdm_part_tool <= 2
    cacheblocking_props = np.array([reorder_options['n_cell_per_cache'],
                                    1,
                                    1,
                                    reorder_options['n_face_per_pack'],
                                    pdm_part_tool],
                                    dtype='int32', order='c')
  elif "HPC" in reorder_options['cell_renum_method']:
    pdm_part_tool     = part_tool_dic[reorder_options['graph_part_tool']]
    cacheblocking_props = np.array([reorder_options['n_cell_per_cache'],
                                    0, # is_asynchrone
                                    1,
                                    reorder_options['n_face_per_pack'],
                                    pdm_part_tool,
                                    1], # n_depth
                                    dtype='int32', order='c')
    # cacheblocking_props = np.array([reorder_options['n_cell_per_cache'],  # n_cell_per_cache_wanted
    #                                 0,                                    # is_asynchrone
    #                                 1,                                    # is_vectorisation
    #                                 reorder_options['n_face_per_pack'],   # n_vect_face
    #                                 pdm_part_tool,                        # split_method
    #                                 1,                                    # n_depth
    #                                 1,                                    # is_gauss_seidel
    #                                 1,                                    # is_gauss_seidel_cell_face
    #                                 1,                                    # enforce_rank_in_subdomain
    #                                 1],
    #                                   dtype='int32', order='c')
  else:
    cacheblocking_props = None
  multipart.multipart_set_reordering(-1,
                                      renum_cell_method.encode('utf-8'),
                                      renum_face_method.encode('utf-8'),
                                      cacheblocking_props)
  multipart.multipart_set_reordering_vtx(-1,
                                         renum_vtx_method.encode('utf-8'))
  keep_alive.append(cacheblocking_props)

def set_mpart_dmeshes(multi_part, u_zones, comm, keep_alive):

  is_ngon_3d = lambda z: PT.Zone.has_nface_elements(z) or \
          (PT.Zone.has_ngon_elements(z) and PT.get_child_from_name(PT.Zone.NGonNode(z), 'ParentElements') is not None)

  for i_zone, zone in enumerate(u_zones):
    if PT.Zone.n_cell(zone) == 0: # Zone has only vertex
      dmesh = cgns_to_pdm_dmesh.cgns_dist_zone_to_pdm_dmesh_vtx(zone, comm)
      keep_alive.append(dmesh)
      multi_part.multipart_register_block(i_zone, dmesh)
    #Determine NGON or ELMT
    elif PT.Zone.has_ngon_elements(zone):
      if is_ngon_3d(zone):
        dmesh    = cgns_to_pdm_dmesh.cgns_dist_zone_to_pdm_dmesh(zone, comm)
        keep_alive.append(dmesh)
        multi_part.multipart_register_block(i_zone, dmesh)
      else:
        dmesh_nodal    = cgns_to_pdm_dmesh.cgns_dist_zone_to_pdm_dmesh_poly2d(zone, comm)
        keep_alive.append(dmesh_nodal)
        multi_part.multipart_register_dmesh_nodal(i_zone, dmesh_nodal)
    else:
      dmesh_nodal = cgns_to_pdm_dmesh.cgns_dist_zone_to_pdm_dmesh_nodal(zone, comm, needs_bc=False)
      keep_alive.append(dmesh_nodal)
      multi_part.multipart_register_dmesh_nodal(i_zone, dmesh_nodal)


def _add_connectivity(multi_part, l_data, i_zone, n_part):
  """
  Enrich dictionnary with additional query of user
  """
  wanted_connectivities = ["cell_face", "face_cell", "face_vtx", "face_edge", "edge_vtx"]
  for key in wanted_connectivities:
    connectivity_type = maia_to_pdm_connectivity[key]
    for i_part in range(n_part):
      dict_res = multi_part.multipart_connectivity_get(i_part, i_zone, connectivity_type)
      l_data[i_part]["np_"+key] = dict_res["np_entity1_entity2"]
      if(dict_res["np_entity1_entity2_idx"] is not None):
        l_data[i_part]["np_"+key+'_idx'] = dict_res["np_entity1_entity2_idx"]


def _add_ln_to_gn(multi_part, l_data, i_zone, n_part):
  """
  Enrich dictionnary with additional query of user
  """
  wanted_lngn = ["cell", "face", "vtx", "edge"]
  for key in wanted_lngn:
    entity_type = maia_to_pdm_entity[key]
    for i_part in range(n_part):
      dict_res = multi_part.multipart_ln_to_gn_get(i_part, i_zone, entity_type)
      l_data[i_part]["np_"+key+'_ln_to_gn'] = dict_res["np_entity_ln_to_gn"]

def _add_color(multi_part, l_data, i_zone, n_part):
  """
  """
  for i_part in range(n_part):
    for key in ['cell', 'face', 'edge', 'vtx']:
      dict_res = multi_part.multipart_part_color_get(i_part, i_zone, maia_to_pdm_entity[key])
      l_data[i_part]["np_"+key+'_color'] = dict_res["np_entity_color"]
    dict_res = multi_part.multipart_thread_color_get(i_part, i_zone)
    l_data[i_part]["np_thread_color"] = dict_res["np_thread_color"]
    dict_res = multi_part.multipart_hyper_plane_color_get(i_part, i_zone)
    l_data[i_part]["np_hyperplane_color"] = dict_res["np_hyper_plane_color"]

def _add_graph_comm(multi_part, l_data, i_zone, n_part):
  """
  """
  wanted_graph = {'face' : PDM._PDM_BOUND_TYPE_FACE, 'vtx' : PDM._PDM_BOUND_TYPE_VTX, 'edge': PDM._PDM_BOUND_TYPE_EDGE}
  for i_part in range(n_part):
    for kind, pdm_kind in wanted_graph.items():
      for key, val in multi_part.multipart_graph_comm_get(i_part, i_zone, pdm_kind).items():
        l_data[i_part][key.replace('entity', kind)] = val

def collect_mpart_partitions(multi_part, d_zones, n_part_per_zone, comm, post_options):
  """
  """
  concat_pdm_data = lambda i_part, i_zone : {**multi_part.multipart_vtx_coord_get         (i_part, i_zone),
                                             **multi_part.multipart_ghost_information_get (i_part, i_zone)}

  all_parts = list()
  for i_zone, d_zone in enumerate(d_zones):

    n_part = n_part_per_zone[i_zone]
    l_dims = list()
    for i_part in range(n_part):
      l_dims.append({f'n_{key}': multi_part.multipart_n_entity_get(i_part, i_zone, entity) \
              for key,entity in maia_to_pdm_entity.items()})
    l_data = [concat_pdm_data(i_part, i_zone)              for i_part in range(n_part)]
    _add_connectivity(multi_part, l_data, i_zone, n_part)
    _add_ln_to_gn    (multi_part, l_data, i_zone, n_part)
    _add_color       (multi_part, l_data, i_zone, n_part)
    _add_graph_comm  (multi_part, l_data, i_zone, n_part)

    #For element : additional conversion step to retrieve part elements
    if PT.Zone.n_cell(d_zone) > 0 and not PT.Zone.has_ngon_elements(d_zone): # pmesh_nodal has not been computed if NGON were present
      pmesh_nodal = multi_part.multipart_part_mesh_nodal_get(i_zone)
      if pmesh_nodal is not None:
        for i_part in range(n_part):
          zone_dim = pmesh_nodal.dim_get()
          for j, kind in enumerate(pdm_geometry_kinds):
            if j <= zone_dim:
              l_data[i_part][f"{j}dsections"] = pmesh_nodal.part_mesh_nodal_get_sections(kind, i_part)
            else: # Section of higher dim than mesh dimension can not be getted (assert in pdm)
              l_data[i_part][f"{j}dsections"] = []

    parts = pdm_part_to_cgns_zone(d_zone, l_dims, l_data, comm, post_options)
    all_parts.extend(parts)

  return all_parts

def part_U_zones(bases_to_block_u, dzone_to_weighted_parts, comm, part_options):

  # Careful ! Some object must be deleted at the very end of the function,
  # since they are usefull for pdm
  keep_alive = list()

  # Bases_to_block_u is the same for each process, but dzone_to_weighted_parts contains
  # partial data
  n_part_per_zone, part_weight = prepare_part_weight(bases_to_block_u, dzone_to_weighted_parts)
  n_zones = n_part_per_zone.size

  keep_alive.append(n_part_per_zone)

  # Init multipart object
  requested_tool = part_options['graph_part_tool']
  if min([PT.Zone.n_cell(z) for zones in bases_to_block_u.values() for z in zones]) == 0:
    requested_tool = 'hilbert'
  pdm_part_tool = maia_to_pdm_split_tool[requested_tool]
  pdm_weight_method = 2
  multi_part = PDM.MultiPart(n_zones, n_part_per_zone, 0, pdm_part_tool, pdm_weight_method, part_weight, comm)

  # Setup
  u_zones = [zone for zones in bases_to_block_u.values() for zone in zones]
  set_mpart_dmeshes(multi_part, u_zones, comm, keep_alive)
  set_mpart_reordering(multi_part, part_options['reordering'], keep_alive)

  #Run and return parts
  multi_part.multipart_run_ppart()

  post_options = {k:part_options[k] for k in ['part_interface_loc', 'dump_pdm_output', 'output_connectivity',
                                              'save_all_connectivities', 'additional_ln_to_gn',
                                              'keep_empty_sections']}
  u_parts = collect_mpart_partitions(multi_part, u_zones, n_part_per_zone, comm, post_options)

  del(multi_part) # Force multi_part object to be deleted before n_part_per_zone array
  for zone in u_zones:
    PT.rm_children_from_name(zone, ':CGNS#MultiPart')

  i = 0
  j = 0
  bases_to_part_u = {}
  for base, zones in bases_to_block_u.items():
    bases_to_part_u[base] = []
    for zone in zones:
      bases_to_part_u[base].extend(u_parts[j:j+n_part_per_zone[i]])
      j += n_part_per_zone[i]
      i += 1

  del(keep_alive)
  return bases_to_part_u


