# =======================================================================================
# ---------------------------------------------------------------------------------------
import  numpy as np
# from    mpi4py import MPI

# MAIA
from    maia.utils                  import np_utils, layouts
import  maia.pytree                                           as PT
from    maia.factory                import dist_from_part     as disc
from    maia.pytree.sids            import node_inspect       as sids
from    maia.pytree.maia            import conventions        as conv
import  maia.transfer.utils                                   as TEU
from    maia.transfer.part_to_dist  import data_exchange      as PTD

# CASSIOPEE
import  Converter.Internal as I

# PARADIGM
import Pypdm.Pypdm as PDM
# ---------------------------------------------------------------------------------------
# =======================================================================================







# =======================================================================================
def exchange_field_one_domain(part_zones, part_tree_ep, exchange, comm) :

  # Part 1 : EXTRACT_PART
  # Part 2 : VOLUME
  
  # Get Isosurf zones (just one normally)
  part_tree_ep_zones = I.getZones(part_tree_ep)
  assert(len(part_tree_ep_zones)==1)
  ep_part_zone = part_tree_ep_zones[0]


  for container_name in exchange :

    # --- Get all fields names ------------------------------------------------
    flds_in_container_names = []
    first_zone_container    = I.getNodeFromName(part_zones[0],container_name)
    for i_fld,fld_node in enumerate(I.getNodesFromType(first_zone_container,'DataArray_t')):
      flds_in_container_names.append(fld_node[0])

    # --- P2P objects definition ----------------------------------------------
    part1_ln_to_gn      = []
    part2_ln_to_gn      = []
    part1_to_part2_idx  = []
    part1_to_part2      = []
    part1_weight        = []
    
    for i_part, part_zone in enumerate(part_zones):
      
      # Get : Field location
      container    = I.getNodeFromName(part_zone,container_name)
      gridLocation = I.getValue(I.getNodeFromName(container,'GridLocation'))
      # Check : correct GridLocation node
      assert(gridLocation in ['Vertex','CellCenter'])

      if gridLocation=='Vertex' :
        # EXTRACT_PART infos
        part1_node_gn       = I.getNodeFromName2(part_tree_ep, ":CGNS#GlobalNumbering")
        part1_node_gn_vtx   = I.getNodeFromName(part1_node_gn, "Vertex")
        part1_ln_to_gn.append(I.getValue(part1_node_gn_vtx))
        
        # VOLUME infos
        vtx_ln_to_gn, face_ln_to_gn, cell_ln_to_gn = TEU.get_entities_numbering(part_zone)
        part2_ln_to_gn.append(vtx_ln_to_gn)
        
        # LINK EP/VOL infos
        part1_to_part2.append(    I.getNodeFromName(part1_node_gn, "ParentVtx")[1])

        part1_to_part2_idx.append(np.arange(0, part1_ln_to_gn[i_part].shape[0]+1, dtype=np.int32 ))


      elif gridLocation=='CellCenter':
        # EXTRACT_PART infos
        # part1_node_gn_cell  = MTM.getGlobalNumbering(part_tree_ep)#, 'Cell') # 
        part1_node_gn       = I.getNodeFromName2(part_tree_ep, ":CGNS#GlobalNumbering")
        part1_node_gn_cell  = I.getNodeFromName2(part1_node_gn, "Cell")
        part1_ln_to_gn.append(I.getValue(part1_node_gn_cell))
        
        # VOLUME infos
        vtx_ln_to_gn, face_ln_to_gn, cell_ln_to_gn = TEU.get_entities_numbering(part_zone)
        part2_ln_to_gn.append(cell_ln_to_gn)
        
        # LINK ISO/VOL infos
        part1_to_part2.append(I.getNodeFromName(part1_node_gn, "ParentCell")[1])
        part1_to_part2_idx.append(np.arange(0, part1_ln_to_gn[i_part].shape[0]+1, dtype=np.int32 ))

    # --- P2P Object --------------------------------------------------------------------
    ptp = PDM.PartToPart(comm,
                         part1_ln_to_gn,
                         part2_ln_to_gn,
                         part1_to_part2_idx,
                         part1_to_part2     )


    # --- FlowSolution node def by zone -------------------------------------------------
    FS_ep = []
    for i_part, part_zone in enumerate(part_zones):
      FS_ep.append(I.newFlowSolution(container_name, gridLocation=gridLocation, parent=ep_part_zone))


    # --- Field exchange ----------------------------------------------------------------
    for i_fld,fld_name in enumerate(flds_in_container_names):
      fld_data = []
      # fld_stri = []
      for i_part, part_zone in enumerate(part_zones):
        container = I.getNodeFromName1(part_zone, container_name)
        fld_node  = I.getNodeFromName1(container, fld_name)
        fld_data.append(I.getValue(fld_node))
        # fld_stri.append(np.ones(fld_data[i_part].shape[0], dtype=np.int32))

        
      # Reverse iexch
      if   gridLocation=="Vertex"    : stride = 1
      elif gridLocation=="CellCenter": stride = 1 #fld_stri[i_fld]

      req_id = ptp.reverse_iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                                 PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART2,
                                 fld_data,
                                 part2_stride=stride)
      part1_strid, part1_data = ptp.reverse_wait(req_id)

      
      # Interpolation and placement
      for i_part, part_zone in enumerate(part_zones):
        path = fld_name
        if   gridLocation=="Vertex"    :
          I.newDataArray(path, part1_data[i_part], parent=FS_ep[i_part])    
      
        elif gridLocation=="CellCenter":
          I.newDataArray(path, part1_data[i_part], parent=FS_ep[i_part])

  return part_tree_ep
# =======================================================================================



# =======================================================================================
def _exchange_field(part_tree, part_tree_ep, exchange, comm) :
  """
  Exchange field between part_tree and part_tree_ep
  for exchange vol field 
  """

  # Get zones by domains
  part_tree_per_dom = disc.get_parts_per_blocks(part_tree, comm).values()

  # Check : monodomain
  assert(len(part_tree_per_dom)==1)

  # Loop over domains
  for i_domain, part_zones in enumerate(part_tree_per_dom):
    part_tree_ep = exchange_field_one_domain(part_zones, part_tree_ep, exchange, comm)


  return part_tree_ep
# =======================================================================================















# =======================================================================================
# ---------------------------------------------------------------------------------------
def extract_part_one_domain(part_zones, zsrpath, comm, equilibrate=1):
  """
  """
  n_part = len(part_zones)
  dim    = 3

  pdm_ep = PDM.ExtractPart(dim,
                           n_part,
                           1, # n_part_out
                           equilibrate,
                           PDM._PDM_SPLIT_DUAL_WITH_HILBERT,
                           comm)

  np_part1_cell_ln_to_gn = list()

  # Loop over domain zone : preparing extract part
  for i_part, part_zone in enumerate(part_zones):
    # Get NGon + NFac
    cx, cy, cz = PT.Zone.coordinates(part_zone)
    vtx_coords = np_utils.interweave_arrays([cx,cy,cz])
    
    ngon  = PT.Zone.NGonNode(part_zone)
    nface = PT.Zone.NFaceNode(part_zone)

    cell_face_idx = I.getNodeFromName1(nface, "ElementStartOffset")[1]
    cell_face     = I.getNodeFromName1(nface, "ElementConnectivity")[1]
    face_vtx_idx  = I.getNodeFromName1(ngon,  "ElementStartOffset")[1]
    face_vtx      = I.getNodeFromName1(ngon,  "ElementConnectivity")[1]

    vtx_ln_to_gn, face_ln_to_gn, cell_ln_to_gn = TEU.get_entities_numbering(part_zone)

    n_cell = cell_ln_to_gn.shape[0]
    n_face = face_ln_to_gn.shape[0]
    n_edge = 0
    n_vtx  = vtx_ln_to_gn .shape[0]

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

    zsr           = I.getNodeFromPath(part_zone, zsrpath)
    extract_l_num = I.getNodeFromName1(zsr, "PointList")

    pdm_ep.selected_lnum_set(i_part, extract_l_num[1])

  pdm_ep.compute()


  # > Reconstruction du maillage de l'extract part --------------------------------------
  n_extract_cell = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_CELL  )
  n_extract_face = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_FACE  )
  n_extract_edge = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_EDGE  )
  n_extract_vtx  = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_VERTEX)

  extract_vtx_coords = pdm_ep.vtx_coord_get(0)

  # > Base construction
  extract_part_base = I.newCGNSBase('Base', cellDim=dim, physDim=3)

  # > Zone construction
  # --- Extract_part 2D -----------------------------------------------------------------
  if n_extract_cell == 0:
    # Ce IF pas cool : si pas de EP sur un proc, il peut croire qu'il est en 2D 
    #(mais en meme temps si c'est vide est ce que c'est un probleme d'ecrire en 2D ?)
    print('[MAIA] EXTRACT_PART : 2D not well implemented')
    extract_part_zone = I.newZone(f'Zone.P{comm.Get_rank()}.N{0}', [[n_extract_vtx, n_extract_face, 0]],
                                  'Unstructured', parent=extract_part_base)
    
    # > Grid coordinates
    cx, cy, cz = layouts.interlaced_to_tuple_coords(extract_vtx_coords)
    extract_grid_coord = I.newGridCoordinates(parent=extract_part_zone)
    I.newDataArray('CoordinateX', cx, parent=extract_grid_coord)
    I.newDataArray('CoordinateY', cy, parent=extract_grid_coord)
    I.newDataArray('CoordinateZ', cz, parent=extract_grid_coord)

    ngon_n = I.newElements('NGonElements', 'NGON', erange = [1, n_extract_edge], parent=extract_part_zone)
    face_vtx_idx, face_vtx  = pdm_ep.connectivity_get(0, PDM._PDM_CONNECTIVITY_TYPE_EDGE_VTX)
    I.newDataArray('ElementConnectivity', face_vtx    , parent=ngon_n)
    I.newDataArray('ElementStartOffset' , face_vtx_idx, parent=ngon_n)

    nface_n = I.newElements('NFaceElements', 'NFACE', erange = [n_extract_edge+1, n_extract_edge+n_extract_face], parent=extract_part_zone)
    cell_face_idx, cell_face = pdm_ep.connectivity_get(0, PDM._PDM_CONNECTIVITY_TYPE_FACE_VTX)
    I.newDataArray('ElementConnectivity', cell_face    , parent=nface_n)
    I.newDataArray('ElementStartOffset' , cell_face_idx, parent=nface_n)

  # --- Extract_part 3D -----------------------------------------------------------------
  else:
    extract_part_zone = I.newZone(f'Zone.P{comm.Get_rank()}.N{0}', [[n_extract_vtx, n_extract_cell, 0]],
                                  'Unstructured', parent=extract_part_base)

    # > Grid coordinates
    cx, cy, cz = layouts.interlaced_to_tuple_coords(extract_vtx_coords)
    extract_grid_coord = I.newGridCoordinates(parent=extract_part_zone)
    I.newDataArray('CoordinateX', cx, parent=extract_grid_coord)
    I.newDataArray('CoordinateY', cy, parent=extract_grid_coord)
    I.newDataArray('CoordinateZ', cz, parent=extract_grid_coord)

    ngon_n = I.newElements('NGonElements', 'NGON', erange = [1, n_extract_face], parent=extract_part_zone)
    face_vtx_idx, face_vtx  = pdm_ep.connectivity_get(0, PDM._PDM_CONNECTIVITY_TYPE_FACE_VTX)
    I.newDataArray('ElementConnectivity', face_vtx    , parent=ngon_n)
    I.newDataArray('ElementStartOffset' , face_vtx_idx, parent=ngon_n)

    nface_n = I.newElements('NFaceElements', 'NFACE', erange = [n_extract_face+1, n_extract_face+n_extract_cell], parent=extract_part_zone)
    cell_face_idx, cell_face = pdm_ep.connectivity_get(0, PDM._PDM_CONNECTIVITY_TYPE_CELL_FACE)
    I.newDataArray('ElementConnectivity', cell_face    , parent=nface_n)
    I.newDataArray('ElementStartOffset' , cell_face_idx, parent=nface_n)
    
    # LN_TO_GN nodes
    vtx_ln_to_gn  = pdm_ep.ln_to_gn_get(0,PDM._PDM_MESH_ENTITY_VERTEX)
    face_ln_to_gn = pdm_ep.ln_to_gn_get(0,PDM._PDM_MESH_ENTITY_FACE)
    cell_ln_to_gn = pdm_ep.ln_to_gn_get(0,PDM._PDM_MESH_ENTITY_CELL)

    # PARENT_LN_TO_GN nodes
    # parent_ln_to_gn_cell = pdm_ep.parent_ln_to_gn_get(0,PDM._PDM_MESH_ENTITY_CELL)
    parent_ln_to_gn_vtx  = pdm_ep.parent_ln_to_gn_get(0,PDM._PDM_MESH_ENTITY_VERTEX)

    gn_face = I.newUserDefinedData(':CGNS#GlobalNumbering', parent=ngon_n)
    I.newDataArray('Element', face_ln_to_gn, parent=gn_face)
    
    gn_cell = I.newUserDefinedData(':CGNS#GlobalNumbering', parent=nface_n)
    I.newDataArray('Element', cell_ln_to_gn, parent=gn_cell)

    gn_zone = I.newUserDefinedData(':CGNS#GlobalNumbering', parent=extract_part_zone)
    I.newDataArray('Vertex'    , vtx_ln_to_gn        , parent=gn_zone)
    I.newDataArray('Cell'      , cell_ln_to_gn       , parent=gn_zone)



  # --- P2P exchange field --------------------------------------------------------------
  np_part2_cell        = cell_ln_to_gn
  np_part2_cell_parent = pdm_ep.parent_ln_to_gn_get(0, PDM._PDM_MESH_ENTITY_CELL  )
  I.newDataArray('ParentCell', np_part2_cell_parent, parent=gn_zone)
  I.newDataArray('ParentVtx' , parent_ln_to_gn_vtx , parent=gn_zone)

  part2_to_part1_idx = np.arange(0, np_part1_cell_ln_to_gn[0].shape[0], dtype=np.int32 )

  # > On a la connectivity part2_to_part1 car part1 = les parents
  ptp = PDM.PartToPart(comm,
                       [np_part2_cell],
                       np_part1_cell_ln_to_gn,
                       [part2_to_part1_idx],
                       [np_part2_cell_parent])

  part2_stri = np.ones(np_part1_cell_ln_to_gn[0].shape[0], dtype=np.int32)
  part2_data = np_part1_cell_ln_to_gn

  #￿> Stride variable
  req_id = ptp.reverse_iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                             PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART2,
                             np_part1_cell_ln_to_gn,
                             part2_stride=[part2_stri])
  part1_strid, part1_data = ptp.reverse_wait(req_id)

  return extract_part_base
# ---------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------
def extract_part(part_tree, fspath, comm, equilibrate=1, exchange=None):
  """
  """

  # Get zones by domains
  part_tree_per_dom = disc.get_parts_per_blocks(part_tree, comm).values()

  # Check : monodomain
  assert(len(part_tree_per_dom)==1)

  extract_doms = I.newCGNSTree()
  

  # Compute extract part of each domain
  for i_domain, part_zones in enumerate(part_tree_per_dom):
    extract_part = extract_part_one_domain(part_zones, fspath, comm,equilibrate=equilibrate)
    I._addChild(extract_doms, extract_part)


  # Exchange fields between two parts
  if exchange is not None:
    extract_part = _exchange_field(part_tree, extract_part, exchange, comm)
  

  # TODO : communiquer sur les BC ?


  # CHECK : fonctionne sur des faces ?



  return extract_doms
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
# =======================================================================================