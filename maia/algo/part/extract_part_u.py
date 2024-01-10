import time
import mpi4py.MPI as MPI

import maia
import maia.pytree        as PT
import maia.pytree.maia   as MT
from   maia.factory  import dist_from_part
from   maia.transfer import utils                as TEU
from   maia.utils    import np_utils, layouts
from   .extraction_utils   import local_pl_offset, LOC_TO_DIM, DIMM_TO_DIMF,\
                                  get_partial_container_stride_and_order, discover_containers
from   .point_cloud_utils  import create_sub_numbering
from   maia import npy_pdm_gnum_dtype as pdm_gnum_dtype

import numpy as np

import Pypdm.Pypdm as PDM


def exchange_field_one_domain(part_zones, extract_zone, mesh_dim, exch_tool_box, container_name, comm) :

  # > Retrieve fields name + GridLocation + PointList if container
  #   is not know by every partition
  mask_container, grid_location, partial_field = discover_containers(part_zones, container_name, 'PointList', 'IndexArray_t', comm)
  if mask_container is None:
    return
  assert grid_location in ['Vertex', 'FaceCenter', 'CellCenter']


  # > FlowSolution node def by zone
  if extract_zone is not None :
    if PT.get_label(mask_container) == 'FlowSolution_t':
      FS_ep = PT.new_FlowSolution(container_name, loc=DIMM_TO_DIMF[mesh_dim][grid_location], parent=extract_zone)
    elif PT.get_label(mask_container) == 'ZoneSubRegion_t':
      FS_ep = PT.new_ZoneSubRegion(container_name, loc=DIMM_TO_DIMF[mesh_dim][grid_location], parent=extract_zone)
    else:
      raise TypeError
  

  # > Get PTP and parentElement for the good location
  ptp        = exch_tool_box['part_to_part'][grid_location]
  
  # LN_TO_GN
  _grid_location    = {"Vertex" : "Vertex", "FaceCenter" : "Element", "CellCenter" : "Cell"}
  
  if extract_zone is not None:
    elt_n            = extract_zone if grid_location!='FaceCenter' else PT.Zone.NGonNode(extract_zone)
    if elt_n is None :return
    part1_elt_gnum_n = PT.maia.getGlobalNumbering(elt_n, _grid_location[grid_location])
    part1_ln_to_gn   = [PT.get_value(part1_elt_gnum_n)]

  # Get reordering informations if point_list
  # https://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array
  if partial_field:
    pl_gnum1, stride = get_partial_container_stride_and_order(part_zones, container_name, grid_location, ptp, comm)

  # > Field exchange
  for fld_node in PT.get_children_from_label(mask_container, 'DataArray_t'):
    fld_name = PT.get_name(fld_node)
    fld_path = f"{container_name}/{fld_name}"
    
    if partial_field:
      # Get field and organize it according to the gnum1_come_from arrays order
      fld_data = list()
      for i_part, part_zone in enumerate(part_zones) :
        fld_n = PT.get_node_from_path(part_zone,fld_path)
        fld_data_tmp = PT.get_value(fld_n) if fld_n is not None else np.empty(0, dtype=np.float64)
        fld_data.append(fld_data_tmp[pl_gnum1[i_part]])
      p2p_type = PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_GNUM1_COME_FROM
    
    else :
      fld_data = [PT.get_node_from_path(part_zone,fld_path)[1] for part_zone in part_zones]
      stride   = 1
      p2p_type = PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART2

    # Reverse iexch
    req_id = ptp.reverse_iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                               p2p_type,
                               fld_data,
                               part2_stride=stride)
    part1_stride, part1_data = ptp.reverse_wait(req_id)

    # Interpolation and placement
    if extract_zone is not None:
      i_part = 0
      if part1_data[i_part].size!=0:
        PT.new_DataArray(fld_name, part1_data[i_part], parent=FS_ep)
  
  # Build PL with the last exchange stride
  if partial_field:
    if len(part1_data)!=0 and part1_data[0].size!=0:
      new_point_list = np.where(part1_stride[0]==1)[0] if part1_data[0].size!=0 else np.empty(0, dtype=np.int32)
      point_list = new_point_list + local_pl_offset(extract_zone, LOC_TO_DIM[grid_location])+1
      new_pl_node = PT.new_PointList(name='PointList', value=point_list.reshape((1,-1), order='F'), parent=FS_ep)
      partial_part1_lngn = [part1_ln_to_gn[0][new_point_list]]
    else:
      partial_part1_lngn = []

    # Update global numbering in FS
    partial_gnum = create_sub_numbering(partial_part1_lngn, comm)
    if extract_zone is not None and len(partial_gnum)!=0:
      PT.maia.newGlobalNumbering({'Index' : partial_gnum[0]}, parent=FS_ep)

  if part1_data[0].size==0:
    PT.rm_child(extract_zone, FS_ep)


def exchange_field_u(part_tree, extract_part_tree, mesh_dim, exch_tool_box, container_names, comm) :
  # Get zones by domains (only one domain for now)
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm)

  # Get zone from extractpart
  extract_zones = PT.get_all_Zone_t(extract_part_tree)
  assert len(extract_zones) <= 1
  extract_zone = extract_zones[0] if len(extract_zones)!=0 else None

  for container_name in container_names:
    for i_domain, dom_part_zones in enumerate(part_tree_per_dom.items()):
      dom_path   = dom_part_zones[0]
      part_zones = dom_part_zones[1]
      exchange_field_one_domain(part_zones, extract_zone, mesh_dim, exch_tool_box[dom_path], \
          container_name, comm)


def extract_part_one_domain_u(part_zones, point_list, location, comm,
                            # equilibrate=True,
                            graph_part_tool="hilbert"):
  """
  Prepare PDM extract_part object and perform the extraction of one domain.
  
  TODO : AJOUTER LE CHOIX PARTIONNEMENT
  """
  equilibrate=True
  
  dim = LOC_TO_DIM[location]

  n_part_in  = len(part_zones)
  n_part_out = 1 if equilibrate else n_part_in
  
  pdm_ep = PDM.ExtractPart(dim, # face/cells
                           n_part_in,
                           n_part_out,
                           equilibrate,
                           eval(f"PDM._PDM_SPLIT_DUAL_WITH_{graph_part_tool.upper()}"),
                           True,
                           comm)

  # > Discover BCs
  dist_zone = PT.new_Zone('Zone')
  gdom_bcs_path_per_dim = {"CellCenter":None, "FaceCenter":None, "EdgeCenter":None, "Vertex":None}
  for bc_type, dim_name in enumerate(gdom_bcs_path_per_dim):
    if LOC_TO_DIM[dim_name]<=dim:
      is_dim_bc = lambda n: PT.get_label(n)=="BC_t" and\
                            PT.Subset.GridLocation(n)==dim_name
      dist_from_part.discover_nodes_from_matching(dist_zone, part_zones, ["ZoneBC_t", is_dim_bc], comm, child_list=['GridLocation'])
      gdom_bcs_path_per_dim[dim_name] = PT.predicates_to_paths(dist_zone, ['ZoneBC_t',is_dim_bc])
      n_gdom_bcs = len(gdom_bcs_path_per_dim[dim_name])
      pdm_ep.part_n_group_set(bc_type+1, n_gdom_bcs)

  # Loop over domain zone : preparing extract part
  for i_part, part_zone in enumerate(part_zones):
    # Get NGon + NFac
    cx, cy, cz = PT.Zone.coordinates(part_zone)
    vtx_coords = np_utils.interweave_arrays([cx,cy,cz])
    
    ngon  = PT.Zone.NGonNode(part_zone)
    nface = PT.Zone.NFaceNode(part_zone)

    cell_face_idx = PT.get_child_from_name(nface, "ElementStartOffset" )[1]
    cell_face     = PT.get_child_from_name(nface, "ElementConnectivity")[1]
    face_vtx_idx  = PT.get_child_from_name(ngon,  "ElementStartOffset" )[1]
    face_vtx      = PT.get_child_from_name(ngon,  "ElementConnectivity")[1]

    vtx_ln_to_gn, _, face_ln_to_gn, cell_ln_to_gn = TEU.get_entities_numbering(part_zone)

    n_cell = cell_ln_to_gn.shape[0]
    n_face = face_ln_to_gn.shape[0]
    n_edge = 0
    n_vtx  = vtx_ln_to_gn .shape[0]

    pdm_ep.part_set(i_part,
                    n_cell, n_face, n_edge, n_vtx,
                    cell_face_idx, cell_face    ,
                    None, None, None,
                    face_vtx_idx , face_vtx     ,
                    cell_ln_to_gn, face_ln_to_gn,
                    None,
                    vtx_ln_to_gn , vtx_coords)

    pdm_ep.selected_lnum_set(i_part, point_list[i_part][0] - local_pl_offset(part_zone, dim) - 1)


    # Add BCs info
    bc_type = 1
    for dim_name, gdom_bcs_path in gdom_bcs_path_per_dim.items():
      if LOC_TO_DIM[dim_name]<=dim:
        for i_bc, bc_path in enumerate(gdom_bcs_path):
          bc_n  = PT.get_node_from_path(part_zone, bc_path)
          bc_pl = PT.get_value(PT.get_child_from_name(bc_n, 'PointList'))[0] \
                    if bc_n is not None else np.empty(0, np.int32)
          bc_gn = PT.get_value(MT.getGlobalNumbering(bc_n, 'Index')) if bc_n is not None else np.empty(0, pdm_gnum_dtype)
          pdm_ep.part_group_set(i_part, i_bc, bc_type, bc_pl-local_pl_offset(part_zone, LOC_TO_DIM[dim_name]) -1 , bc_gn)
      bc_type +=1

  pdm_ep.compute()

  # > Reconstruction du maillage de l'extract part
  n_extract_cell = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_CELL  )
  n_extract_face = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_FACE  )
  n_extract_edge = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_EDGE  )
  n_extract_vtx  = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_VERTEX)
  
  size_by_dim = {0: [[n_extract_vtx, 0             , 0]], # not yet implemented
                 1:   None                              , # not yet implemented
                 2: [[n_extract_vtx, n_extract_face, 0]],
                 3: [[n_extract_vtx, n_extract_cell, 0]] }

  # > ExtractPart zone construction
  extract_zone = PT.new_Zone(PT.maia.conv.add_part_suffix('Zone', comm.Get_rank(), 0),
                               size=size_by_dim[dim],
                               type='Unstructured')

  ep_vtx_ln_to_gn  = pdm_ep.ln_to_gn_get(0,PDM._PDM_MESH_ENTITY_VERTEX)
  PT.maia.newGlobalNumbering({"Vertex" : ep_vtx_ln_to_gn}, parent=extract_zone)

  # > Grid coordinates
  cx, cy, cz = layouts.interlaced_to_tuple_coords(pdm_ep.vtx_coord_get(0))
  extract_grid_coord = PT.new_GridCoordinates(parent=extract_zone)
  PT.new_DataArray('CoordinateX', cx, parent=extract_grid_coord)
  PT.new_DataArray('CoordinateY', cy, parent=extract_grid_coord)
  PT.new_DataArray('CoordinateZ', cz, parent=extract_grid_coord)

  if dim == 0:
    PT.maia.newGlobalNumbering({'Cell' : np.empty(0, dtype=ep_vtx_ln_to_gn.dtype)}, parent=extract_zone)

  # > NGON
  if dim >= 2:
    ep_face_vtx_idx, ep_face_vtx  = pdm_ep.connectivity_get(0, PDM._PDM_CONNECTIVITY_TYPE_FACE_VTX)
    ngon_n = PT.new_NGonElements( 'NGonElements',
                                  erange  = [1, n_extract_face],
                                  ec      = ep_face_vtx,
                                  eso     = ep_face_vtx_idx,
                                  parent  = extract_zone)

    ep_face_ln_to_gn = pdm_ep.ln_to_gn_get(0, PDM._PDM_MESH_ENTITY_FACE)
    PT.maia.newGlobalNumbering({'Element' : ep_face_ln_to_gn}, parent=ngon_n)
    if dim == 2:
      PT.maia.newGlobalNumbering({'Cell' : ep_face_ln_to_gn}, parent=extract_zone)

  # > NFACES
  if dim == 3:
    ep_cell_face_idx, ep_cell_face = pdm_ep.connectivity_get(0, PDM._PDM_CONNECTIVITY_TYPE_CELL_FACE)
    nface_n = PT.new_NFaceElements('NFaceElements',
                                    erange  = [n_extract_face+1, n_extract_face+n_extract_cell],
                                    ec      = ep_cell_face,
                                    eso     = ep_cell_face_idx,
                                    parent  = extract_zone)

    ep_cell_ln_to_gn = pdm_ep.ln_to_gn_get(0, PDM._PDM_MESH_ENTITY_CELL)
    PT.maia.newGlobalNumbering({'Element' : ep_cell_ln_to_gn}, parent=nface_n)
    PT.maia.newGlobalNumbering({'Cell' : ep_cell_ln_to_gn}, parent=extract_zone)

    maia.algo.nface_to_pe(extract_zone, comm)

  # - Get BCs
  zonebc_n = PT.new_ZoneBC(parent=extract_zone)
  bc_type = 1
  for dim_name, gdom_bcs_path in gdom_bcs_path_per_dim.items():
    if LOC_TO_DIM[dim_name]<=dim:
      for i_bc, bc_path in enumerate(gdom_bcs_path):
        bc_info = pdm_ep.extract_part_group_get(0, i_bc, bc_type)
        bc_pl = bc_info['group_entity'] +1
        bc_gn = bc_info['group_entity_ln_to_gn']
        if bc_pl.size != 0:
          bc_name = bc_path.split('/')[-1]
          bc_n = PT.new_BC(bc_name, point_list=bc_pl.reshape((1,-1), order='F'), loc=dim_name, parent=zonebc_n)
          PT.maia.newGlobalNumbering({'Index':bc_gn}, parent=bc_n)
    bc_type +=1 

  # - Get PTP by vertex and cell
  ptp = dict()
  if equilibrate:
    ptp['Vertex']       = pdm_ep.part_to_part_get(PDM._PDM_MESH_ENTITY_VERTEX)
    if dim >= 2: # NGON
      ptp['FaceCenter'] = pdm_ep.part_to_part_get(PDM._PDM_MESH_ENTITY_FACE)
    if dim == 3: # NFACE
      ptp['CellCenter'] = pdm_ep.part_to_part_get(PDM._PDM_MESH_ENTITY_CELL)
    
  # - Get parent elt
  parent_elt = dict()
  parent_elt['Vertex']       = pdm_ep.parent_ln_to_gn_get(0,PDM._PDM_MESH_ENTITY_VERTEX)
  if dim >= 2: # NGON
    parent_elt['FaceCenter'] = pdm_ep.parent_ln_to_gn_get(0,PDM._PDM_MESH_ENTITY_FACE)
  if dim == 3: # NFACE
    parent_elt['CellCenter'] = pdm_ep.parent_ln_to_gn_get(0,PDM._PDM_MESH_ENTITY_CELL)
  
  exch_tool_box = {'part_to_part' : ptp, 'parent_elt' : parent_elt}

  return [extract_zone], exch_tool_box


