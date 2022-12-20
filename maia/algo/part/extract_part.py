# =======================================================================================
# ---------------------------------------------------------------------------------------
from    mpi4py import MPI
import  numpy as np

import  maia.pytree as PT
import  maia
from    maia.transfer import utils                as TEU
from    maia.factory  import dist_from_part
from    maia.utils    import np_utils, layouts, py_utils

import Pypdm.Pypdm as PDM
from maia.algo.part.extract_boundary import compute_gnum_from_parent_gnum
# ---------------------------------------------------------------------------------------
# =======================================================================================

# =======================================================================================
# ---------------------------------------------------------------------------------------

LOC_TO_DIM   = {'Vertex':0, 'EdgeCenter':1, 'FaceCenter':2, 'CellCenter':3}
DIMM_TO_DIMF = { 0: {'Vertex':'Vertex'},
               # 1: {'Vertex': None,    'EdgeCenter':None, 'FaceCenter':None, 'CellCenter':None},
                 2: {'Vertex':'Vertex', 'EdgeCenter':'EdgeCenter', 'FaceCenter':'CellCenter'},
                 3: {'Vertex':'Vertex', 'EdgeCenter':'EdgeCenter', 'FaceCenter':'FaceCenter', 'CellCenter':'CellCenter'}}

def local_pl_offset(part_zone, dim):
  # Works only for ngon / nface 3D meshes
  if dim == 3:
    nface = PT.Zone.NFaceNode(part_zone)
    return PT.Element.Range(nface)[0] - 1
  if dim == 2:
    ngon = PT.Zone.NGonNode(part_zone)
    return PT.Element.Range(ngon)[0] - 1
  else:
    return 0

# ---------------------------------------------------------------------------------------
# =======================================================================================



# =======================================================================================
class Extractor:
  def __init__( self,
                part_tree, point_list, location, comm,
                equilibrate=True,
                graph_part_tool="hilbert"):

    self.part_tree        = part_tree
    self.exch_tool_box    = list()
    self.comm             = comm

    # Get zones by domains
    part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm).values()
    # Check : monodomain
    assert len(part_tree_per_dom) == 1

    # ExtractPart dimension
    self.dim    = LOC_TO_DIM[location]
    assert self.dim in [0,2,3], "[MAIA] Error : dimensions 0 and 1 not yet implemented"
    #CGNS does not support 0D, so keep input dim in this case (which is 3 since 2d is not managed)
    cell_dim = 3 if location == 'Vertex' else self.dim 
    
    # ExtractPart CGNSTree
    extracted_tree = PT.new_CGNSTree()
    extracted_base = PT.new_CGNSBase('Base', cell_dim=cell_dim, phy_dim=3, parent=extracted_tree)

    assert graph_part_tool in ["hilbert","parmetis","ptscotch"]
    assert not( (self.dim==0) and graph_part_tool in ['parmetis', 'ptscotch']),\
           '[MAIA] Vertex extraction not available with parmetis or ptscotch partitioning. Please check your script.' 

    # Compute extract part of each domain
    for i_domain, part_zones in enumerate(part_tree_per_dom):
      extracted_zone, etb = extract_part_one_domain(part_zones, point_list[i_domain], self.dim, comm,
                                                    equilibrate=equilibrate,
                                                    graph_part_tool=graph_part_tool)
      self.exch_tool_box.append(etb)
      PT.add_child(extracted_base, extracted_zone)
    self.extracted_tree = extracted_tree

  def exchange_fields(self, fs_container):
    # Get zones by domains (only one domain for now)
    part_tree_per_dom = dist_from_part.get_parts_per_blocks(self.part_tree, self.comm).values()

    # Get zone from extractpart
    extracted_zones = PT.get_all_Zone_t(self.extracted_tree)
    assert len(extracted_zones) <= 1
    extracted_zone = extracted_zones[0]

    for container_name in fs_container:
      # Loop over domains
      for i_domain, part_zones in enumerate(part_tree_per_dom):
        exchange_field_one_domain(part_zones, extracted_zone, self.dim, self.exch_tool_box[i_domain], \
            container_name, self.comm)

  def get_extract_part_tree(self) :
    return self.extracted_tree
# =======================================================================================




# =======================================================================================
# ---------------------------------------------------------------------------------------
def exchange_field_one_domain(part_zones, part_zone_ep, mesh_dim, exch_tool_box, container_name, comm) :

  loc_correspondance = {'Vertex'    : 'Vertex',
                        'FaceCenter': 'Cell',
                        'CellCenter': 'Cell'}

  # Part 1 : EXTRACT_PART
  # Part 2 : VOLUME
  # --- Get all fields names and location ---------------------------------------------
  all_ordering    = list()
  all_stride_int  = list()
  all_stride_bool = list()
  all_part_gnum1  = list()

  # Retrieve fields name + GridLocation + PointList if container
  # is not know by every partition
  mask_zone = ['MaskedZone', None, [], 'Zone_t']
  dist_from_part.discover_nodes_from_matching(mask_zone, part_zones, container_name, comm, \
      child_list=['GridLocation'])
  fields_query = lambda n: PT.get_label(n) in ['DataArray_t', 'IndexArray_t']
  dist_from_part.discover_nodes_from_matching(mask_zone, part_zones, [container_name, fields_query], comm)
  mask_container = PT.get_child_from_name(mask_zone, container_name)

  gridLocation = PT.Subset.GridLocation(mask_container)
  partial_field = PT.get_child_from_name(mask_container, 'PointList') is not None
  assert gridLocation in ['Vertex', 'FaceCenter', 'CellCenter']

  # --- FlowSolution node def by zone -------------------------------------------------
  if PT.get_label(mask_container) == 'FlowSolution_t':
    FS_ep = PT.new_FlowSolution(container_name, loc=DIMM_TO_DIMF[mesh_dim][gridLocation], parent=part_zone_ep)
  elif PT.get_label(mask_container) == 'ZoneSubRegion_t':
    FS_ep = PT.new_ZoneSubRegion(container_name, loc=DIMM_TO_DIMF[mesh_dim][gridLocation], parent=part_zone_ep)
  else:
    raise TypeError
  
  # --- Get PTP and parentElement for the good location
  ptp        = exch_tool_box['part_to_part'][gridLocation]
  parent_elt = exch_tool_box['parent_elt'][gridLocation]

  # Get reordering informations if point_list
  # https://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array
  if partial_field:
    for i_part, part_zone in enumerate(part_zones):
      container        = PT.get_child_from_name(part_zone, container_name)
      if container is not None:
        point_list_node  = PT.get_child_from_name(container, 'PointList')
        point_list  = point_list_node[1][0] - local_pl_offset(part_zone, LOC_TO_DIM[gridLocation]) # Gnum start at 1

      part_gnum1  = ptp.get_gnum1_come_from()[i_part]['come_from'] # Get partition order
      ref_lnum2   = ptp.get_referenced_lnum2()[i_part] # Get partition order

      if container is None or point_list.size == 0:
        ref_lnum2_idx = np.empty(0,dtype=np.int32)
        stride        = np.zeros(ref_lnum2.shape,dtype=np.int32)
        all_part_gnum1.append(np.empty(0,dtype=np.int32)) # Select only part1_gnum that is in part2 point_list
      else:
        sort_idx    = np.argsort(point_list)                 # Sort order of point_list ()
        order       = np.searchsorted(point_list,ref_lnum2,sorter=sort_idx)
        ref_lnum2_idx = np.take(sort_idx, order, mode="clip")
        
        stride = point_list[ref_lnum2_idx] == ref_lnum2
        all_part_gnum1.append(part_gnum1[stride]) # Select only part1_gnum that is in part2 point_list

      all_ordering   .append(ref_lnum2_idx)
      all_stride_bool.append(stride)
      all_stride_int .append(stride.astype(np.int32))


    # Echange gnum to retrieve flowsol new point_list
    req_id = ptp.reverse_iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                               PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_GNUM1_COME_FROM,
                               all_part_gnum1,
                               part2_stride=all_stride_int)
    part1_strid, part2_gnum = ptp.reverse_wait(req_id)
    
    if part2_gnum[0].size == 0:
      new_point_list = np.empty(0, dtype=np.int32)
    else :
      sort_idx       = np.argsort(part2_gnum[0]) # Sort order of point_list ()
      order          = np.searchsorted(part2_gnum[0],parent_elt,sorter=sort_idx)

      parent_elt_idx = np.take(sort_idx, order, mode="clip")

      stride         = part2_gnum[0][parent_elt_idx] == parent_elt
      local_point_list = np.where(stride)[0] + 1
      point_list = local_point_list + local_pl_offset(part_zone_ep, LOC_TO_DIM[gridLocation])

    new_pl_node    = PT.new_PointList(name='PointList', value=point_list.reshape((1,-1), order='F'), parent=FS_ep)

    # Boucle sur les partitoins de l'extraction pour get PL
    gnum = PT.maia.getGlobalNumbering(part_zone_ep, f'{loc_correspondance[gridLocation]}')[1]
    partial_gnum = compute_gnum_from_parent_gnum([gnum[local_point_list-1]], comm)[0]
    
    # Boucle sur les partitoins de l'extracttion pour placer PL        
    maia.pytree.maia.newGlobalNumbering({'Index' : partial_gnum}, parent=FS_ep)

  # --- Field exchange ----------------------------------------------------------------
  for fld_node in PT.get_children_from_label(mask_container, 'DataArray_t'):
    fld_name = fld_node[0]
    fld_path = f"{container_name}/{fld_name}"
    
    # Reordering if ZSR container
    if partial_field: 
      
      fld_data = list()
      for i_part, part_zone in enumerate(part_zones):
        fld_part_n = PT.get_node_from_path(part_zone, fld_path)
        if fld_part_n is None:
          fld_part = np.empty(0, dtype=np.float64) #We should search the numpy dtype
        else:
          fld_part = PT.get_value(fld_part_n)
          if fld_part.size != 0:
            fld_part = fld_part[all_ordering[i_part]][all_stride_bool[i_part]]
        fld_data.append(fld_part)

      req_id = ptp.reverse_iexch( PDM._PDM_MPI_COMM_KIND_P2P,
                                      PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_GNUM1_COME_FROM,
                                      fld_data,
                                      part2_stride=all_stride_int)
    else:
      fld_data = [PT.get_node_from_path(part_zone, fld_path)[1] for part_zone in part_zones]
      req_id = ptp.reverse_iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                                 PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART2,
                                 fld_data,
                                 part2_stride=1)

    part1_strid, part1_data = ptp.reverse_wait(req_id)

    # Interpolation and placement
    i_part = 0
    PT.new_DataArray(fld_name, part1_data[i_part], parent=FS_ep)
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
# =======================================================================================





# =======================================================================================
# ---------------------------------------------------------------------------------------
def extract_part_one_domain(part_zones, point_list, dim, comm,
                            equilibrate=True,
                            graph_part_tool="hilbert"):
  """
  TODO : AJOUTER LE CHOIX PARTIONNEMENT
  """
  n_part_in  = len(part_zones)
  n_part_out = 1 if equilibrate else n_part_in
  
  pdm_ep = PDM.ExtractPart(dim, # face/cells
                           n_part_in,
                           n_part_out,
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

    pdm_ep.selected_lnum_set(i_part, point_list[i_part] - local_pl_offset(part_zone, dim) - 1)

  pdm_ep.compute()

  # > Reconstruction du maillage de l'extract part --------------------------------------
  n_extract_cell = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_CELL  )
  n_extract_face = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_FACE  )
  n_extract_edge = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_EDGE  )
  n_extract_vtx  = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_VERTEX)
  
  size_by_dim = {0: [[n_extract_vtx, 0             , 0]], # not yet implemented
                 1:   None                              , # not yet implemented
                 2: [[n_extract_vtx, n_extract_face, 0]],
                 3: [[n_extract_vtx, n_extract_cell, 0]] }

  # --- ExtractPart zone construction ---------------------------------------------------
  extracted_zone = PT.new_Zone(PT.maia.conv.add_part_suffix('Zone', comm.Get_rank(), 0),
                               size=size_by_dim[dim],
                               type='Unstructured')

  ep_vtx_ln_to_gn  = pdm_ep.ln_to_gn_get(0,PDM._PDM_MESH_ENTITY_VERTEX)
  PT.maia.newGlobalNumbering({"Vertex" : ep_vtx_ln_to_gn}, parent=extracted_zone)

  # > Grid coordinates
  cx, cy, cz = layouts.interlaced_to_tuple_coords(pdm_ep.vtx_coord_get(0))
  extract_grid_coord = PT.new_GridCoordinates(parent=extracted_zone)
  PT.new_DataArray('CoordinateX', cx, parent=extract_grid_coord)
  PT.new_DataArray('CoordinateY', cy, parent=extract_grid_coord)
  PT.new_DataArray('CoordinateZ', cz, parent=extract_grid_coord)

  if dim == 0:
    PT.maia.newGlobalNumbering({'Cell' : np.empty(0, dtype=ep_vtx_ln_to_gn.dtype)}, parent=extracted_zone)

  # > NGON
  if dim >= 2:
    ep_face_vtx_idx, ep_face_vtx  = pdm_ep.connectivity_get(0, PDM._PDM_CONNECTIVITY_TYPE_FACE_VTX)
    ngon_n = PT.new_NGonElements( 'NGonElements',
                                  erange  = [1, n_extract_face],
                                  ec      = ep_face_vtx,
                                  eso     = ep_face_vtx_idx,
                                  parent  = extracted_zone)

    ep_face_ln_to_gn = pdm_ep.ln_to_gn_get(0, PDM._PDM_MESH_ENTITY_FACE)
    PT.maia.newGlobalNumbering({'Element' : ep_face_ln_to_gn}, parent=ngon_n)
    if dim == 2:
      PT.maia.newGlobalNumbering({'Cell' : ep_face_ln_to_gn}, parent=extracted_zone)

  # > NFACES
  if dim == 3:
    ep_cell_face_idx, ep_cell_face = pdm_ep.connectivity_get(0, PDM._PDM_CONNECTIVITY_TYPE_CELL_FACE)
    nface_n = PT.new_NFaceElements('NFaceElements',
                                    erange  = [n_extract_face+1, n_extract_face+n_extract_cell],
                                    ec      = ep_cell_face,
                                    eso     = ep_cell_face_idx,
                                    parent  = extracted_zone)

    ep_cell_ln_to_gn = pdm_ep.ln_to_gn_get(0, PDM._PDM_MESH_ENTITY_CELL)
    PT.maia.newGlobalNumbering({'Element' : ep_cell_ln_to_gn}, parent=nface_n)
    PT.maia.newGlobalNumbering({'Cell' : ep_cell_ln_to_gn}, parent=extracted_zone)

    maia.algo.nface_to_pe(extracted_zone, comm)


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

  return extracted_zone, exch_tool_box
# ---------------------------------------------------------------------------------------
# =======================================================================================



# =======================================================================================
# --- EXTRACT PART FROM ZSR -------------------------------------------------------------

# ---------------------------------------------------------------------------------------
def extract_part_from_zsr(part_tree, zsr_name, comm,
                          # equilibrate=True,
                          containers_name=None,
                          **options):
  """Extract the submesh defined by the provided ZoneSubRegion from the input volumic
  partitioned tree.

  Dimension of the output mesh is set up accordingly to the GridLocation of the ZoneSubRegion.
  Submesh is returned as an independant partitioned CGNSTree and includes the relevant connectivities.

  In addition, containers specified in ``containers_name`` list are transfered to the extracted tree.
  Containers to be transfered can be either of label FlowSolution_t or ZoneSubRegion_t.

  Important:
    - Input tree must be unstructured and have a ngon connectivity.
    - Partitions must come from a single initial domain on input tree.
  
  See also:
    :func:`create_extractor_from_zsr` takes the same parameters, excepted ``containers_name``,
    and returns an Extractor object which can be used to exchange containers more than once through its
    ``Extractor.exchange_fields(container_name)`` method.

  Args:
    part_tree       (CGNSTree)    : Partitioned tree from which extraction is computed. Only U-NGon
      connectivities are managed.
    zsr_name        (str)         : Name of the ZoneSubRegion_t node
    comm            (MPIComm)     : MPI communicator
    containers_name (list of str) : List of the names of the fields containers to transfer
                                    on the output extracted tree.
    **options: Options related to the extraction.
  Returns:
    extracted_tree (CGNSTree)  : Extracted submesh (partitioned)

  Extraction can be controled thought the optional kwargs:

    - ``graph_part_tool`` (str) -- Partitioning tool used to balance the extracted zones.
      Admissible values are ``hilbert, parmetis, ptscotch``. Note that
      vertex-located extractions require hilbert partitioning. Defaults to ``hilbert``.

  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #extract_from_zsr@start
      :end-before:  #extract_from_zsr@end
      :dedent: 2
  """


  extractor = create_extractor_from_zsr(part_tree, zsr_name, comm, **options)

  if containers_name is not None:
    extractor.exchange_fields(containers_name)

  return extractor.get_extract_part_tree()

# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
def create_extractor_from_zsr(part_tree, zsr_path, comm,
                              # equilibrate=True,
                              **options
                              ):
  """Same as extract_part_from_zsr, but return the extractor object."""
  # Get zones by domains

  graph_part_tool = options.get("graph_part_tool", "hilbert")

  part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm)

  # Get point_list for each partitioned zone and group it by domain
  point_list = list()
  location = ''
  for domain, part_zones in part_tree_per_dom.items():
    point_list_domain = list()
    for part_zone in part_zones:
      zsr_node     = PT.get_node_from_path(part_zone, zsr_path)
      if zsr_node is not None:
        #Follow BC or GC link
        related_node = PT.getSubregionExtent(zsr_node, part_zone)
        zsr_node     = PT.get_node_from_path(part_zone, related_node)
        point_list_domain.append(PT.get_child_from_name(zsr_node, "PointList")[1][0])
        location = PT.Subset.GridLocation(zsr_node)
      else: # ZSR does not exists on this partition
        point_list_domain.append(np.empty(0, np.int32))
    point_list.append(point_list_domain)
  
  # Get location if proc has no zsr
  location = comm.allreduce(location, op=MPI.MAX)

  return Extractor(part_tree, point_list, location, comm,
                   # equilibrate=equilibrate,
                   graph_part_tool=graph_part_tool
                   )
# ---------------------------------------------------------------------------------------

# --- END EXTRACT PART FROM ZSR ---------------------------------------------------------
# =======================================================================================







# =======================================================================================
# ---------------------------------------------------------------------------------------
def extract_part_from_bc_name(part_tree, bc_name, comm,
                              # equilibrate=True,
                              containers_name=None,
                              **options):
  """Extract the submesh defined by the provided BC name from the input volumic
  partitioned tree.

  Dimension of the output mesh is set up accordingly to the GridLocation of the BC.
  Submesh is returned as an independant partitioned CGNSTree and includes the relevant connectivities.

  In addition, containers specified in ``containers_name`` list are transfered to the extracted tree.
  Containers to be transfered can only be from or ZoneSubRegion_t. BCDataSet from BC can be transfered too
  using the ``transfer_dataset`` option.

  Important:
    - Input tree must be unstructured and have a ngon connectivity.
    - Partitions must come from a single initial domain on input tree.
  
  See also:
    :func:`create_extractor_from_bc_name` takes the same parameters, excepted ``containers_name``,
    and returns an Extractor object which can be used to exchange containers more than once through its
    ``Extractor.exchange_fields(container_name)`` method.

  Args:
    part_tree       (CGNSTree)    : Partitioned tree from which extraction is computed. Only U-NGon
      connectivities are managed.
    zsr_name        (str)         : Name of the BC node
    comm            (MPIComm)     : MPI communicator
    containers_name (list of str) : List of the names of the fields containers to transfer
                                    on the output extracted tree.
    **options: Options related to the extraction.
  Returns:
    extracted_tree (CGNSTree)  : Extracted submesh (partitioned)

  Extraction can be controled thought the optional kwargs:

    - ``graph_part_tool`` (str) -- Partitioning tool used to balance the extracted zones.
      Admissible values are ``hilbert, parmetis, ptscotch``. Note that
      vertex-located extractions require hilbert partitioning. Defaults to ``hilbert``.
    - ``transfer_dataset`` (bool) -- Allows the BCDataSet transfer from input to output tree
      (in a FlowSolution_t node). Defaults to ``True``.

  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #extract_from_bc_name@start
      :end-before:  #extract_from_bc_name@end
      :dedent: 2
  """

  transfer_dataset = options.get("transfer_dataset", True)

  # Local copy of the part_tree to add ZSR 
  local_part_tree   = PT.shallow_copy(part_tree)
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(local_part_tree, comm)

  # Adding ZSR to tree
  for domain, part_zones in part_tree_per_dom.items():
    for part_zone in part_zones:
      zsr_bc_name = bc_name+'_zsr'
      bc_n = PT.get_node_from_name_and_label(part_zone, bc_name, 'BC_t') 
      if bc_n is not None:
        zsr_bc_n  = PT.new_ZoneSubRegion(name=bc_name, bc_name=bc_name, parent=part_zone)
        if transfer_dataset:
          bc_dataset = PT.get_children_from_predicates(bc_n, 'BCDataSet_t/BCData_t/DataArray_t')
          assert PT.get_child_from_predicates(bc_n, 'BCDataSet_t/IndexArray_t') is None,\
                 'BCDataSet_t with PointList aren\'t managed'
          for dataset in bc_dataset:
            PT.new_DataArray(name=dataset[0], value=dataset[1], parent=zsr_bc_n)
          if len(bc_dataset)!=0:
            if containers_name is None: containers_name = [bc_name]
            else                      : containers_name.append(bc_name)
            bc_pl =              PT.get_node_from_name(bc_n, 'PointList'   )[1]
            bc_gl = PT.get_value(PT.get_node_from_name(bc_n, 'GridLocation'))
            PT.new_PointList('PointList', bc_pl, parent=zsr_bc_n)
            PT.new_GridLocation(          bc_gl, parent=zsr_bc_n)

  import Converter.Internal as I
  extractor = create_extractor_from_zsr(local_part_tree, bc_name, comm, **options)

  if containers_name is not None:
    extractor.exchange_fields(containers_name)


  return extractor.get_extract_part_tree()

# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
# =======================================================================================

