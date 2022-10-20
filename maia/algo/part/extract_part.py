# =======================================================================================
# ---------------------------------------------------------------------------------------
from    mpi4py import MPI
import  numpy as np

import  maia.pytree as PT
import  maia
from    maia.transfer import utils                as TEU
from    maia.factory  import dist_from_part
from    maia.utils    import np_utils, layouts, py_utils
# from    maia.pytree.sids            import node_inspect       as sids
# from    maia.pytree.maia            import conventions        as conv
# from    maia.transfer.part_to_dist  import data_exchange      as PTD

import Pypdm.Pypdm as PDM
# ---------------------------------------------------------------------------------------
# =======================================================================================







# =======================================================================================
def exchange_field_one_domain(part_zones, part_tree_ep, ptp, exchange, comm) :
  
  # Get Isosurf zones (just one normally)
  part_tree_ep_zones = PT.get_all_Zone_t(part_tree_ep)
  assert(len(part_tree_ep_zones)==1)
  ep_part_zone = part_tree_ep_zones[0]


  for container_name in exchange :

    # --- Get all fields names and location -----------------------------------
    all_fld_names = []
    all_locs = []
    for part_zone in part_zones:
      container = PT.request_child_from_name(part_zone, container_name)
      fld_names = {PT.get_name(n) for n in PT.iter_children_from_label(container, "DataArray_t")}
      py_utils.append_unique(all_fld_names, fld_names)
      py_utils.append_unique(all_locs, PT.Subset.GridLocation(container))
    if len(part_zones) > 0:
      assert len(all_locs) == len(all_fld_names) == 1
      tag = comm.Get_rank()
      loc_and_fields = all_locs[0], list(all_fld_names[0])
    else:
      tag = -1
      loc_and_fields = None
    master = comm.allreduce(tag, op=MPI.MAX) # No check global ?
    gridLocation, flds_in_container_names = comm.bcast(loc_and_fields, master)
    assert(gridLocation in ['Vertex','CellCenter'])


    # --- FlowSolution node def by zone -------------------------------------------------
    FS_ep = []
    for i_part, part_zone in enumerate(part_zones):
      FS_ep.append(PT.new_FlowSolution(container_name, loc=gridLocation, parent=ep_part_zone))


    # --- Field exchange ----------------------------------------------------------------
    for i_fld,fld_name in enumerate(flds_in_container_names):
      fld_data = []
      # fld_stri = []
      for i_part, part_zone in enumerate(part_zones):
        container = PT.get_child_from_name(part_zone, container_name)
        fld_node  = PT.get_child_from_name(container, fld_name)
        fld_data.append(PT.get_value(fld_node))

        
      # Reverse iexch
      if   gridLocation=="Vertex"    :
        ptp_loc = ptp.get('vertex')
        stride = 1

      elif gridLocation=="CellCenter":
        ptp_loc = ptp.get('cell')
        stride = 1 #fld_stri[i_fld]

      req_id = ptp_loc.reverse_iexch( PDM._PDM_MPI_COMM_KIND_P2P,
                                      PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART2,
                                      fld_data,
                                      part2_stride=stride)
      part1_strid, part1_data = ptp_loc.reverse_wait(req_id)

      
      # Interpolation and placement
      for i_part, part_zone in enumerate(part_zones):
        path = fld_name
        if   gridLocation=="Vertex"    :
          PT.new_DataArray(path, part1_data[i_part], parent=FS_ep[i_part])    
      
        elif gridLocation=="CellCenter":
          PT.new_DataArray(path, part1_data[i_part], parent=FS_ep[i_part])

# =======================================================================================


# =======================================================================================
def _exchange_field(part_tree, part_tree_ep, ptp,exchange, comm) :
  """
  Exchange field between part_tree and part_tree_ep
  for exchange vol field 
  """

  # Get zones by domains
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm).values()

  # Check : monodomain
  assert(len(part_tree_per_dom)==1)

  # Loop over domains
  for i_domain, part_zones in enumerate(part_tree_per_dom):
    exchange_field_one_domain(part_zones, part_tree_ep, ptp, exchange, comm)

# =======================================================================================











# =======================================================================================
# ---------------------------------------------------------------------------------------
def extract_part_one_domain(part_zones, zsrpath, comm,
                            equilibrate=1,
                            graph_part_tool="hilbert",
                            put_pe=False):
  """
  TODO : AJOUTER LE CHOIX PARTIONNEMENT
  """
  
  n_part = len(part_zones)
  dim    = 3
  # dim    = 2

  pdm_ep = PDM.ExtractPart(dim, # face/cells
                           n_part,
                           1, # n_part_out
                           equilibrate,
                           eval(f"PDM._PDM_SPLIT_DUAL_WITH_{graph_part_tool.upper()}"),
                           True,
                           comm)

  np_part1_cell_ln_to_gn = list()

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

    vtx_ln_to_gn, face_ln_to_gn, cell_ln_to_gn = TEU.get_entities_numbering(part_zone)

    # n_cell = cell_ln_to_gn.shape[0]
    n_cell = cell_ln_to_gn.shape[0]
    n_face = face_ln_to_gn.shape[0]
    n_edge = 0
    n_vtx  = vtx_ln_to_gn .shape[0]

    print('[MAIA][EXTRACT_PART] pdm_ep.part_set()')
    pdm_ep.part_set(i_part,
                    n_cell,
                    n_face,
                    n_edge,
                    n_vtx,
                    cell_face_idx,
                    cell_face    ,
                    None,
                    None,
                    None,
                    face_vtx_idx ,
                    face_vtx     ,
                    cell_ln_to_gn,
                    face_ln_to_gn,
                    None,
                    vtx_ln_to_gn ,
                    vtx_coords)

    np_part1_cell_ln_to_gn.append(cell_ln_to_gn)

    zsr           = PT.get_node_from_path(part_zone, zsrpath)
    extract_l_num = PT.get_child_from_name(zsr, "PointList")

    print('[MAIA][EXTRACT_PART] pdm_ep.selected_lnum_set()')
    pdm_ep.selected_lnum_set(i_part, extract_l_num[1])

  print('[MAIA][EXTRACT_PART] pdm_ep.compute()')
  pdm_ep.compute()


  # > Reconstruction du maillage de l'extract part --------------------------------------
  n_extract_cell = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_CELL  )
  n_extract_face = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_FACE  )
  n_extract_edge = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_EDGE  )
  n_extract_vtx  = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_VERTEX)
  print(f'[{comm.Get_rank()}][MAIA] n_extract_cell = {n_extract_cell}')
  print(f'[{comm.Get_rank()}][MAIA] n_extract_face = {n_extract_face}')
  print(f'[{comm.Get_rank()}][MAIA] n_extract_edge = {n_extract_edge}')
  print(f'[{comm.Get_rank()}][MAIA] n_extract_vtx  = {n_extract_vtx }')

  extract_vtx_coords = pdm_ep.vtx_coord_get(0)

  # > Zone construction
  # --- Extract_part 2D -----------------------------------------------------------------
  if (n_extract_cell == 0) and (n_extract_vtx != 0):
    # Ce IF pas cool : si pas de EP sur un proc, il peut croire qu'il est en 2D 
    #(mais en meme temps si c'est vide est ce que c'est un probleme d'ecrire en 2D ?)
    print('[MAIA] EXTRACT_PART : 2D not well implemented')
    extract_part_zone = PT.new_Zone(PT.maia.conv.add_part_suffix('Zone', comm.Get_rank(), 0),
                                    size=[[n_extract_vtx, n_extract_face, 0]],
                                    type='Unstructured')
    
    # > Grid coordinates
    cx, cy, cz = layouts.interlaced_to_tuple_coords(extract_vtx_coords)
    extract_grid_coord = PT.new_GridCoordinates(parent=extract_part_zone)
    PT.new_DataArray('CoordinateX', cx, parent=extract_grid_coord)
    PT.new_DataArray('CoordinateY', cy, parent=extract_grid_coord)
    PT.new_DataArray('CoordinateZ', cz, parent=extract_grid_coord)

    # print('Get PDM._PDM_CONNECTIVITY_TYPE_EDGE_VTX')
    # ep_face_vtx_idx, ep_face_vtx  = pdm_ep.connectivity_get(0, PDM._PDM_CONNECTIVITY_TYPE_EDGE_VTX)
    # ngon_n = PT.new_NGonElements( 'NGonElements',
    #                               erange  = [1, n_extract_face],
    #                               ec      = ep_face_vtx,
    #                               eso     = ep_face_vtx_idx,
    #                               parent  = extract_part_zone)

    # print('Get PDM._PDM_CONNECTIVITY_TYPE_FACE_VTX')
    # ep_cell_face_idx, ep_cell_face = pdm_ep.connectivity_get(0, PDM._PDM_CONNECTIVITY_TYPE_FACE_VTX)
    # nface_n = PT.new_NFaceElements( 'NFaceElements',
    #                                 erange  = [n_extract_edge+1, n_extract_edge+n_extract_face],
    #                                 ec      = ep_cell_face,
    #                                 eso     = ep_cell_face_idx,
    #                                 parent  = extract_part_zone)
    print('Get PDM._PDM_CONNECTIVITY_TYPE_FACE_VTX')
    ep_face_vtx_idx, ep_face_vtx  = pdm_ep.connectivity_get(0, PDM._PDM_CONNECTIVITY_TYPE_FACE_VTX)
    ngon_n = PT.new_NGonElements( 'NGonElements',
                                  erange  = [1, n_extract_face],
                                  ec      = ep_face_vtx,
                                  eso     = ep_face_vtx_idx,
                                  parent  = extract_part_zone)
    print("AH!")
    # ep_cell_face_idx, ep_cell_face = pdm_ep.connectivity_get(0, PDM._PDM_CONNECTIVITY_TYPE_FACE_VTX)
    # nface_n = PT.new_NFaceElements( 'NFaceElements',
    #                                 erange  = [n_extract_edge+1, n_extract_edge+n_extract_face],
    #                                 ec      = ep_cell_face,
    #                                 eso     = ep_cell_face_idx,
    #                                 parent  = extract_part_zone)

  # --- Extract_part 3D -----------------------------------------------------------------
  else:
    extract_part_zone = PT.new_Zone(PT.maia.conv.add_part_suffix('Zone', comm.Get_rank(), 0),
                                    size=[[n_extract_vtx, n_extract_cell, 0]],
                                    type='Unstructured')

    # > Grid coordinates
    cx, cy, cz = layouts.interlaced_to_tuple_coords(extract_vtx_coords)
    extract_grid_coord = PT.new_GridCoordinates(parent=extract_part_zone)
    PT.new_DataArray('CoordinateX', cx, parent=extract_grid_coord)
    PT.new_DataArray('CoordinateY', cy, parent=extract_grid_coord)
    PT.new_DataArray('CoordinateZ', cz, parent=extract_grid_coord)

    # > Elements
    ep_face_vtx_idx, ep_face_vtx  = pdm_ep.connectivity_get(0, PDM._PDM_CONNECTIVITY_TYPE_FACE_VTX)
    ngon_n = PT.new_NGonElements('NGonElements',
                                  erange  = [1, n_extract_face],
                                  ec      = ep_face_vtx,
                                  eso     = ep_face_vtx_idx,
                                  parent  = extract_part_zone)

    ep_cell_face_idx, ep_cell_face = pdm_ep.connectivity_get(0, PDM._PDM_CONNECTIVITY_TYPE_CELL_FACE)
    nface_n = PT.new_NFaceElements('NFaceElements',
                                    erange  = [n_extract_face+1, n_extract_face+n_extract_cell],
                                    ec      = ep_cell_face,
                                    eso     = ep_cell_face_idx,
                                    parent  = extract_part_zone)
    
    # Compute ParentElement nodes is requested
    if (put_pe):
      maia.algo.nface_to_pe(extract_part_zone, comm)
    
    # LN_TO_GN nodes
    vtx_ln_to_gn  = pdm_ep.ln_to_gn_get(0,PDM._PDM_MESH_ENTITY_VERTEX)
    face_ln_to_gn = pdm_ep.ln_to_gn_get(0,PDM._PDM_MESH_ENTITY_FACE)
    cell_ln_to_gn = pdm_ep.ln_to_gn_get(0,PDM._PDM_MESH_ENTITY_CELL)

    PT.maia.newGlobalNumbering({'Element' : face_ln_to_gn}, parent=ngon_n)
    
    PT.maia.newGlobalNumbering({'Element' : cell_ln_to_gn}, parent=nface_n)

    PT.maia.newGlobalNumbering({'Vertex' : vtx_ln_to_gn ,
                                'Cell'   : cell_ln_to_gn }, parent=extract_part_zone)

  # - Get PTP by vertex and cell
  ptp = dict()
  ptp['vertex'] = pdm_ep.part_to_part_get(PDM._PDM_MESH_ENTITY_VERTEX)
  ptp['cell'  ] = pdm_ep.part_to_part_get(PDM._PDM_MESH_ENTITY_CELL)

  return extract_part_zone,ptp
# ---------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------
def extract_part(part_tree, fspath, comm, equilibrate=1, exchange=None, graph_part_tool='hilbert'):
  """Extract vertex/edges/faces/cells from the ZSR node from the provided partitioned CGNSTree.

  ExtractPart is returned as an independant partitioned CGNSTree. 

  Important:
    - Input tree must be unstructured and have a ngon connectivity.
    - Partitions must come from a single initial domain on input tree.

  Note:
    Once created, fields from provided partitionned CGNSTree
    can be exchanged using
    ``_exchange_field(part_tree, iso_part_tree, containers_name, comm)``

  Args:
    part_tree     (CGNSTree)    : Partitioned tree from which ExtractPart is computed. Only U-NGon
      connectivities are managed.
    iso_field     (str)         : Path to the ZSR field.
    comm          (MPIComm)     : MPI communicator
    iso_val       (float, optional) : Value to use to compute isosurface. Defaults to 0.
    containers_name   (list of str) : List of the names of the FlowSolution_t nodes to transfer
      on the output isosurface tree.
    **options: Options related to plane extraction.
  Returns:
    isosurf_tree (CGNSTree): Surfacic tree (partitioned)

  Extraction can be controled thought the optional kwargs:

    - ``elt_type`` (str) -- Controls the shape of elements used to describe
      the isosurface. Admissible values are ``TRI_3, QUAD_4, NGON_n``. Defaults to ``TRI_3``.

  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #compute_iso_surface@start
      :end-before: #compute_iso_surface@end
      :dedent: 2
  """

  # Get zones by domains
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm).values()

  # Check : monodomain
  assert(len(part_tree_per_dom)==1)

  # Is there PE node
  if (PT.get_node_from_name(part_tree,'ParentElements') is not None): put_pe = True
  else                                                              : put_pe = False
  
  extract_part_tree = PT.new_CGNSTree()
  extract_part_base = PT.new_CGNSBase('Base', cell_dim=3, phy_dim=3, parent=extract_part_tree)


  # Compute extract part of each domain
  for i_domain, part_zones in enumerate(part_tree_per_dom):
    print('[MAIA][EXTRACT_PART] call to extract_part_one_domain()')
    extract_part_zone,ptp = extract_part_one_domain(part_zones, fspath, comm,
                                                    equilibrate=equilibrate,
                                                    graph_part_tool=graph_part_tool,
                                                    put_pe=put_pe)
    PT.add_child(extract_part_base, extract_part_zone)


  # Exchange fields between two parts
  if exchange is not None:
    print('[MAIA][EXTRACT_PART] call to _exchange_field()')
    _exchange_field(part_tree, extract_part_tree, ptp, exchange, comm)
  

  # TODO : communiquer sur les BC ?

  # CHECK : fonctionne sur des faces ?


  return extract_part_tree
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
# =======================================================================================