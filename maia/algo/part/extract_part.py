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
# ---------------------------------------------------------------------------------------
class Extractor:
  def __init__( self,
                part_tree, point_list, location, comm,
                equilibrate=1,
                graph_part_tool="hilbert"):

    self.part_tree        = part_tree
    self.point_list       = point_list
    self.location         = location
    self.equilibrate      = equilibrate
    self.graph_part_tool  = graph_part_tool
    self.ptp              = list()

    # Get zones by domains
    part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm).values()

    # Check : monodomain
    assert(len(part_tree_per_dom)==1)

    # Is there PE node
    if (PT.get_node_from_name(part_tree,'ParentElements') is not None): self.put_pe = True
    else                                                              : self.put_pe = False
    
    # ExtractPart dimension
    select_dim  = { 'Vertex':0 ,'EdgeCenter':1 ,'FaceCenter':2 ,'CellCenter':3}
    assert self.location in select_dim.keys()
    self.dim    = select_dim[self.location]
    assert self.dim in [0,2,3],"[MAIA] Error : dimensions 0 and 1 not yet implemented"
    
    # ExtractPart CGNSTree
    self.extract_part_tree = PT.new_CGNSTree()
    self.extract_part_base = PT.new_CGNSBase('Base', cell_dim=self.dim, phy_dim=3, parent=self.extract_part_tree)

    # Compute extract part of each domain
    for i_domain, part_zones in enumerate(part_tree_per_dom):
      
      # extract part from point list
      extract_part_zone,ptpdom = extract_part_one_domain(part_zones, self.point_list, self.dim, comm,
                                                         equilibrate=self.equilibrate,
                                                         graph_part_tool=self.graph_part_tool,
                                                         put_pe=self.put_pe)
      self.ptp.append(ptpdom)
      PT.add_child(self.extract_part_base, extract_part_zone)
# ---------------------------------------------------------------------------------------
  

# ---------------------------------------------------------------------------------------
  def exchange_fields(self, fs_container, comm) :
    _exchange_field(self.part_tree, self.extract_part_tree, self.ptp, fs_container, comm)
    return None
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
  def exchange_zsr_fields(self, zsr_path, comm) :
    exchange_zsr_fields(self.part_tree, self.extract_part_tree, zsr_path, self.ptp, comm)
    return None
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
  def save_parent_num(self) :
    # Possible to get parent_num from p2p or only from pdm_ep ?
    return None
# ---------------------------------------------------------------------------------------
  
# ---------------------------------------------------------------------------------------
# =======================================================================================




# =======================================================================================
# ---------------------------------------------------------------------------------------
def exchange_field_one_domain(part_zones, part_zone_ep, ptp, exchange, comm) :
  
  # Part 1 : EXTRACT_PART
  # Part 2 : VOLUME
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
    try :
      FS_ep = PT.new_FlowSolution(container_name, loc=gridLocation, parent=part_zone_ep)
    except :
      FS_ep = PT.get_node_from_name(part_zone_ep,container_name)
    # --- Field exchange ----------------------------------------------------------------
    for fld_name in flds_in_container_names:
      fld_path = f"{container_name}/{fld_name}"
      fld_data = [PT.get_node_from_path(part_zone,fld_path)[1] for part_zone in part_zones]
        
      # Reverse iexch
      ptp_loc = ptp[gridLocation]

      req_id = ptp_loc.reverse_iexch( PDM._PDM_MPI_COMM_KIND_P2P,
                                      PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART2,
                                      fld_data,
                                      part2_stride=1)
      part1_strid, part1_data = ptp_loc.reverse_wait(req_id)

      
      # Interpolation and placement
      i_part = 0
      PT.new_DataArray(fld_name, part1_data[i_part], parent=FS_ep)

# ---------------------------------------------------------------------------------------

def _exchange_field(part_tree, part_tree_ep, ptp,exchange, comm) :
  """
  Exchange field between part_tree and part_tree_ep
  for exchange vol field 
  """

  # Get zones by domains
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm).values()

  # Check : monodomain
  assert(len(part_tree_per_dom)==1)
  assert(len(part_tree_per_dom)==len(ptp))

  # Get zone from extractpart
  part_zone_ep = PT.get_all_Zone_t(part_tree_ep)
  assert(len(part_zone_ep)<=1)
  part_zone_ep = part_zone_ep[0]

  # Loop over domains
  for i_domain, part_zones in enumerate(part_tree_per_dom):
    exchange_field_one_domain(part_zones, part_zone_ep, ptp[i_domain], exchange, comm)

# ---------------------------------------------------------------------------------------
# =======================================================================================





# =======================================================================================
# ---------------------------------------------------------------------------------------
def extract_part_one_domain(part_zones, point_list, dim, comm,
                            equilibrate=1,
                            graph_part_tool="hilbert",
                            put_pe=False):
  """
  TODO : AJOUTER LE CHOIX PARTIONNEMENT
  """
  n_part = len(part_zones)
  # print(n_par)
  pdm_ep = PDM.ExtractPart(dim, # face/cells
                           n_part,
                           1, # n_part_out
                           equilibrate,
                           eval(f"PDM._PDM_SPLIT_DUAL_WITH_{graph_part_tool.upper()}"),
                           True,
                           comm)

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
    # if (comm.Get_rank()==2):
    #   print("point_list[i_part]   = ",point_list[i_part]  )
    #   print("point_list[i_part]-1 = ",point_list[i_part]-1)
    adjusted_point_list = point_list[i_part]  # -1 because of CGNS norm
    pdm_ep.selected_lnum_set(i_part, adjusted_point_list)
    # if (comm.Get_rank()==0):
    #   print(f"i_part = {i_part} ; point_list[i_part]=",point_list[i_part])

  pdm_ep.compute()


  # > Reconstruction du maillage de l'extract part --------------------------------------
  n_extract_cell = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_CELL  ) # ; print(f'[{comm.Get_rank()}][MAIA] n_extract_cell = {n_extract_cell}')
  n_extract_face = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_FACE  ) # ; print(f'[{comm.Get_rank()}][MAIA] n_extract_face = {n_extract_face}')
  n_extract_edge = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_EDGE  ) # ; print(f'[{comm.Get_rank()}][MAIA] n_extract_edge = {n_extract_edge}')
  n_extract_vtx  = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_VERTEX) # ; print(f'[{comm.Get_rank()}][MAIA] n_extract_vtx  = {n_extract_vtx }')
  

  extract_vtx_coords = pdm_ep.vtx_coord_get(0)
  
  size_by_dim = {0: [[n_extract_vtx, 0             , 0]], # not yet implemented
                 1:   None                              , # not yet implemented
                 2: [[n_extract_vtx, n_extract_face, 0]],
                 3: [[n_extract_vtx, n_extract_cell, 0]] }


  # --- ExtractPart zone construction ---------------------------------------------------
  extract_part_zone = PT.new_Zone(PT.maia.conv.add_part_suffix('Zone', comm.Get_rank(), 0),
                                  size=size_by_dim[dim],
                                  type='Unstructured')

  # > Grid coordinates
  cx, cy, cz = layouts.interlaced_to_tuple_coords(extract_vtx_coords)
  extract_grid_coord = PT.new_GridCoordinates(parent=extract_part_zone)
  PT.new_DataArray('CoordinateX', cx, parent=extract_grid_coord)
  PT.new_DataArray('CoordinateY', cy, parent=extract_grid_coord)
  PT.new_DataArray('CoordinateZ', cz, parent=extract_grid_coord)

  # > NGON
  if (dim>=2) :
    ep_face_vtx_idx, ep_face_vtx  = pdm_ep.connectivity_get(0, PDM._PDM_CONNECTIVITY_TYPE_FACE_VTX)
    ngon_n = PT.new_NGonElements( 'NGonElements',
                                  erange  = [1, n_extract_face],
                                  ec      = ep_face_vtx,
                                  eso     = ep_face_vtx_idx,
                                  parent  = extract_part_zone)
  # > NFACES
  if (dim==3) :
    ep_cell_face_idx, ep_cell_face = pdm_ep.connectivity_get(0, PDM._PDM_CONNECTIVITY_TYPE_CELL_FACE)
    nface_n = PT.new_NFaceElements('NFaceElements',
                                    erange  = [n_extract_face+1, n_extract_face+n_extract_cell],
                                    ec      = ep_cell_face,
                                    eso     = ep_cell_face_idx,
                                    parent  = extract_part_zone)

    # Compute ParentElement nodes is requested
    if (put_pe):
      maia.algo.nface_to_pe(extract_part_zone, comm)

    
  # > LN_TO_GN nodes
  ep_vtx_ln_to_gn  = None
  ep_face_ln_to_gn = None
  ep_cell_ln_to_gn = None

  ep_vtx_ln_to_gn  = pdm_ep.ln_to_gn_get(0,PDM._PDM_MESH_ENTITY_VERTEX)

  if (dim>=2) : # NGON
    ep_face_ln_to_gn = pdm_ep.ln_to_gn_get(0,PDM._PDM_MESH_ENTITY_FACE)
    PT.maia.newGlobalNumbering({'Element' : ep_face_ln_to_gn}, parent=ngon_n)
    
  if (dim==3) : # NFACE
    ep_cell_ln_to_gn = pdm_ep.ln_to_gn_get(0,PDM._PDM_MESH_ENTITY_CELL)
    PT.maia.newGlobalNumbering({'Element' : ep_cell_ln_to_gn}, parent=nface_n)

  ln_to_gn_by_dim = { 0: {'Cell': ep_vtx_ln_to_gn },
                      1:   None,                                                  # not yet implemented
                      2: {'Vertex': ep_vtx_ln_to_gn , 'Cell': ep_face_ln_to_gn },
                      3: {'Vertex': ep_vtx_ln_to_gn , 'Cell': ep_cell_ln_to_gn } }
  PT.maia.newGlobalNumbering(ln_to_gn_by_dim[dim], parent=extract_part_zone)

  # - Get PTP by vertex and cell
  ptp = dict()
  ptp['Vertex']       = pdm_ep.part_to_part_get(PDM._PDM_MESH_ENTITY_VERTEX)
  if (dim>=2) : # NGON
    ptp['FaceCenter'] = pdm_ep.part_to_part_get(PDM._PDM_MESH_ENTITY_FACE)
  if (dim==3) : # NFACE
    ptp['CellCenter'] = pdm_ep.part_to_part_get(PDM._PDM_MESH_ENTITY_CELL)

  
  # DEBUG
  parent_cell    = pdm_ep.parent_ln_to_gn_get(0,PDM._PDM_MESH_ENTITY_CELL)
  parent_vertex  = pdm_ep.parent_ln_to_gn_get(0,PDM._PDM_MESH_ENTITY_VERTEX)
  parent_node    = PT.new_node('maia#parents', label='UserDefinedData_t', parent=extract_part_zone)
  PT.new_DataArray('Cell_parent'  , parent_cell  , parent=parent_node)
  PT.new_DataArray('Vertex_parent', parent_vertex, parent=parent_node)
  # FIN DEBUG



  return extract_part_zone, ptp
# ---------------------------------------------------------------------------------------
# =======================================================================================



# =======================================================================================
# ---------------------------------------------------------------------------------------
def extract_part_from_point_list(part_tree, point_list, location, comm, equilibrate=1, exchange=None, graph_part_tool='hilbert'):
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
  
  # ExtractPart dimension
  select_dim  = { 'Vertex':0 ,'EdgeCenter':1 ,'FaceCenter':2 ,'CellCenter':3}
  dim         = select_dim[location]
  assert dim in [0,2,3],"[MAIA] Error : dimensions 0 and 1 not yet implemented"
  
  # ExtractPart CGNSTree
  extract_part_tree = PT.new_CGNSTree()
  extract_part_base = PT.new_CGNSBase('Base', cell_dim=dim, phy_dim=3, parent=extract_part_tree)


  # Compute extract part of each domain
  # pdm_ep=list()
  ptp   =list()
  for i_domain, part_zones in enumerate(part_tree_per_dom):
    extract_part_zone,ptpdom = extract_part_one_domain(part_zones, point_list, dim, comm,
                                                       equilibrate=equilibrate,
                                                       graph_part_tool=graph_part_tool,
                                                       put_pe=put_pe)
    ptp.append(ptpdom)
    PT.add_child(extract_part_base, extract_part_zone)

  # Exchange fields between two parts
  if exchange is not None:
    _exchange_field(part_tree, extract_part_tree, ptp, exchange, comm)
  

  return extract_part_tree
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
def create_extractor_from_point_list(part_tree, point_list, location, comm, equilibrate=1, graph_part_tool='hilbert'):

  return Extractor(part_tree, point_list, location, comm,
                   equilibrate=equilibrate,
                   graph_part_tool=graph_part_tool)
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
# =======================================================================================





# =======================================================================================
# --- EXTRACT PART FROM ZSR -------------------------------------------------------------

# ---------------------------------------------------------------------------------------
def exchange_zsr_field_one_domain(part_zones, part_zone_ep, zsr_path, ptp, comm) :
  
  # Part 1 : EXTRACT_PART
  # Part 2 : VOLUME
  
  
  field = dict()
  
  # Get ZSR
  for i_part,part_zone in enumerate(part_zones) :
    zsr_node = PT.get_child_from_name(part_zone, zsr_path)

    # Field location
    grid_loc = PT.get_value(PT.get_child_from_name(zsr_node,'GridLocation')) 
    assert(grid_loc in ['Vertex','CellCenter'])
    
    # Get PointList
    point_list = PT.get_value(PT.get_child_from_label(zsr_node,'IndexArray_t'))

    # Get PTP by locatoin
    ptp_loc   = ptp[grid_loc]

    # Reordonnancement du field en fonction du P2P
    ref_lnum2 = ptp_loc.get_referenced_lnum2()[i_part] # Get partition order 
    # if (comm.Get_rank()==0):
    #   print(f'ipart={i_part} ; ref_lnum2  = ',ref_lnum2)
    #   print(f'ipart={i_part} ; point_list = ',point_list)
    sort_idx  = np.argsort(point_list+1)                 # Sort order of point_list ()
    order     = np.searchsorted(point_list+1,ref_lnum2,sorter=sort_idx)
    # gnum_pdm   = np.array([ 4, 9,25,12,31])
    # point_list = np.array([31, 4, 9,25,12])
    # field_pl   = 10.*point_list
    # sort_idx   = np.argsort(point_list)
    # pl_sorted  = point_list[sort_idx] 
    # order      = np.searchsorted(point_list,gnum_pdm,sorter=sort_idx)

    # FlowSol node
    try:
      FS_ep = PT.new_FlowSolution("ZSR_FlowSolution", loc=grid_loc, parent=part_zone_ep)
    except RuntimeError:
      FS_ep = PT.get_node_from_name(part_zone_ep,"ZSR_FlowSolution")

    field_nodes = PT.get_nodes_from_label(zsr_node,'DataArray_t')
    for field_node in field_nodes:
      field_name = PT.get_name(field_node)
      field_data = PT.get_value(field_node)
      field_data = field_data[sort_idx[order]]

      try : 
        field[field_name].append(field_data)
      except KeyError:
        field[field_name] = list()
        field[field_name].append(field_data)


  # print(field)
  
  i_part = 0 # Because just one
  for fld_name in field.keys():
    req_id = ptp_loc.reverse_iexch( PDM._PDM_MPI_COMM_KIND_P2P,
                                    PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART1_TO_PART2,
                                    field[fld_name],
                                    part2_stride=1)
    part1_strid, part1_data = ptp_loc.reverse_wait(req_id)

    # print(fld_name)
    PT.new_DataArray(fld_name, part1_data[i_part], parent=FS_ep)

# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
def exchange_zsr_fields(part_tree, extract_part_tree, zsr_path, ptp, comm):
  """
  Exchange field between part_tree and part_tree_ep
  for exchange vol field 
  """

  # Get zones by domains
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm).values()

  # Check : monodomain
  assert(len(part_tree_per_dom)==1)
  assert(len(part_tree_per_dom)==len(ptp))

  # Get zone from extractpart
  extract_part_zone = PT.get_all_Zone_t(extract_part_tree)
  assert(len(extract_part_zone)<=1)
  extract_part_zone = extract_part_zone[0]

  # Loop over domains
  for i_domain, part_zones in enumerate(part_tree_per_dom):
    exchange_zsr_field_one_domain(part_zones, extract_part_zone, zsr_path, ptp[i_domain], comm)

# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
def extract_part_from_zsr(part_tree, zsr_path, comm,
                          equilibrate=1, exchange=None, graph_part_tool='hilbert'):

  # Get zones by domains
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm).values()

  # Check : monodomain
  assert(len(part_tree_per_dom)==1)

  # Is there PE node
  if (PT.get_node_from_name(part_tree,'ParentElements') is not None): put_pe = True
  else                                                              : put_pe = False
  
  # ExtractPart dimension
  select_dim  = { 'Vertex':0 ,'EdgeCenter':1 ,'FaceCenter':2 ,'CellCenter':3}
  ZSR_node    = PT.get_node_from_name(part_tree,zsr_path)
  assert ZSR_node is not None 
  dim         = select_dim[PT.get_value(PT.get_child_from_name(ZSR_node,'GridLocation'))]
  assert dim in [0,2,3],"[MAIA] Error : dimensions 0 and 1 not yet implemented"
  
  # ExtractPart CGNSTree
  extract_part_tree = PT.new_CGNSTree()
  extract_part_base = PT.new_CGNSBase('Base', cell_dim=dim, phy_dim=3, parent=extract_part_tree)


  # Compute extract part of each domain
  # pdm_ep=list()
  ptp   =list()
  for i_domain, part_zones in enumerate(part_tree_per_dom):
    
    # Get point_list for each partitioned zone in the domain
    point_list = list()
    for part_zone in part_zones:
      # Get point_list from zsr node
      zsr_node    = PT.get_node_from_path(part_zone, zsr_path)
      zsr_pl_node = PT.get_child_from_name(zsr_node, "PointList")
      point_list.append(PT.get_value(zsr_pl_node))
    # print(point_list)
    # extract part from point list
    extract_part_zone,ptpdom = extract_part_one_domain(part_zones, point_list, dim, comm,
                                                       equilibrate=equilibrate,
                                                       graph_part_tool=graph_part_tool,
                                                       put_pe=put_pe)
    # print(point_list)
    ptp.append(ptpdom)
    PT.add_child(extract_part_base, extract_part_zone)

  exchange_zsr_fields(part_tree, extract_part_tree, zsr_path, ptp, comm)  

  # Exchange fields between two parts
  if exchange is not None:
    _exchange_field(part_tree, extract_part_tree, ptp, exchange, comm)
  

  return extract_part_tree
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
def create_extractor_from_zsr(part_tree, zsr_path, comm, equilibrate=1, graph_part_tool='hilbert'):

  # Get zones by domains
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm).values()
  assert(len(part_tree_per_dom)==1)

  # zsr node and location
  ZSR_node    = PT.get_node_from_name(part_tree,zsr_path)
  assert ZSR_node is not None 
  location    = PT.get_value(PT.get_child_from_name(ZSR_node,'GridLocation'))

  # Get point_list or each partitioned zone
  for i_domain, part_zones in enumerate(part_tree_per_dom):
    point_list = list()
    for part_zone in part_zones:
      # Get point_list from zsr node
      zsr_node    = PT.get_node_from_path(part_zone, zsr_path)
      zsr_pl_node = PT.get_child_from_name(zsr_node, "PointList")
      point_list.append(PT.get_value(zsr_pl_node))

  return Extractor(part_tree, point_list, location, comm,
                   equilibrate=equilibrate,
                   graph_part_tool=graph_part_tool)
# ---------------------------------------------------------------------------------------

# --- END EXTRACT PART FROM ZSR ---------------------------------------------------------
# =======================================================================================







# =======================================================================================
# ---------------------------------------------------------------------------------------
# # ---------------------------------------------------------------------------------------
# def extract_part_from_bnd():
#   return extract_part_tree

# def create_extractor_from_bnd():
#   # get point list
#   return Extractor
# # ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
# =======================================================================================

