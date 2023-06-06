import time
import mpi4py.MPI as MPI

import maia
import maia.pytree        as PT
import maia.utils.logging as mlog
from   maia.factory  import dist_from_part
from   maia.transfer import utils                as TEU
from   maia.utils    import np_utils, layouts, py_utils
from   .extraction_utils   import local_pl_offset, LOC_TO_DIM, get_partial_container_stride_and_order
from   .point_cloud_utils  import create_sub_numbering

import numpy as np

import Pypdm.Pypdm as PDM

DIMM_TO_DIMF = { 0: {'Vertex':'Vertex'},
               # 1: {'Vertex': None,    'EdgeCenter':None, 'FaceCenter':None, 'CellCenter':None},
                 2: {'Vertex':'Vertex', 'EdgeCenter':'EdgeCenter', 'FaceCenter':'CellCenter'},
                 3: {'Vertex':'Vertex', 'EdgeCenter':'EdgeCenter', 'FaceCenter':'FaceCenter', 'CellCenter':'CellCenter'}}


class Extractor:
  def __init__( self,
                part_tree, point_list, location, comm,
                equilibrate=True,
                graph_part_tool="hilbert"):

    self.part_tree     = part_tree
    self.exch_tool_box = list()
    self.comm          = comm

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



def exchange_field_one_domain(part_zones, part_zone_ep, mesh_dim, exch_tool_box, container_name, comm) :

  loc_correspondance = {'Vertex'    : 'Vertex',
                        'FaceCenter': 'Cell',
                        'CellCenter': 'Cell'}

  # > Retrieve fields name + GridLocation + PointList if container
  #   is not know by every partition
  mask_zone = ['MaskedZone', None, [], 'Zone_t']
  dist_from_part.discover_nodes_from_matching(mask_zone, part_zones, container_name, comm, \
      child_list=['GridLocation', 'BCRegionName', 'GridConnectivityRegionName'])
  
  fields_query = lambda n: PT.get_label(n) in ['DataArray_t', 'IndexArray_t']
  dist_from_part.discover_nodes_from_matching(mask_zone, part_zones, [container_name, fields_query], comm)
  mask_container = PT.get_child_from_name(mask_zone, container_name)
  if mask_container is None:
    raise ValueError("[maia-extract_part] asked container for exchange is not in tree")

  # > Manage BC and GC ZSR
  ref_zsr_node    = mask_container
  bc_descriptor_n = PT.get_child_from_name(mask_container, 'BCRegionName')
  gc_descriptor_n = PT.get_child_from_name(mask_container, 'GridConnectivityRegionName')
  assert not (bc_descriptor_n and gc_descriptor_n)
  if bc_descriptor_n is not None:
    bc_name      = PT.get_value(bc_descriptor_n)
    dist_from_part.discover_nodes_from_matching(mask_zone, part_zones, ['ZoneBC_t', bc_name], comm, child_list=['PointList', 'GridLocation_t'])
    ref_zsr_node = PT.get_child_from_predicates(mask_zone, f'ZoneBC_t/{bc_name}')
  elif gc_descriptor_n is not None:
    gc_name      = PT.get_value(gc_descriptor_n)
    dist_from_part.discover_nodes_from_matching(mask_zone, part_zones, ['ZoneGridConnectivity_t', gc_name], comm, child_list=['PointList', 'GridLocation_t'])
    ref_zsr_node = PT.get_child_from_predicates(mask_zone, f'ZoneGridConnectivity_t/{gc_name})')
  
  gridLocation = PT.Subset.GridLocation(ref_zsr_node)
  partial_field = PT.get_child_from_name(ref_zsr_node, 'PointList') is not None
  assert gridLocation in ['Vertex', 'FaceCenter', 'CellCenter']


  # > FlowSolution node def by zone
  if PT.get_label(mask_container) == 'FlowSolution_t':
    FS_ep = PT.new_FlowSolution(container_name, loc=DIMM_TO_DIMF[mesh_dim][gridLocation], parent=part_zone_ep)
  elif PT.get_label(mask_container) == 'ZoneSubRegion_t':
    FS_ep = PT.new_ZoneSubRegion(container_name, loc=DIMM_TO_DIMF[mesh_dim][gridLocation], parent=part_zone_ep)
  else:
    raise TypeError
  

  # > Get PTP and parentElement for the good location
  ptp        = exch_tool_box['part_to_part'][gridLocation]
  
  # LN_TO_GN
  _gridLocation    = {"Vertex" : "Vertex", "FaceCenter" : "Element", "CellCenter" : "Cell"}
  elt_n            = part_zone_ep if gridLocation!='FaceCenter' else PT.Zone.NGonNode(part_zone_ep)
  if elt_n is None :return
  part1_elt_gnum_n = PT.maia.getGlobalNumbering(elt_n, _gridLocation[gridLocation])
  part1_ln_to_gn   = [PT.get_value(part1_elt_gnum_n)]

  # Get reordering informations if point_list
  # https://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array
  if partial_field:
    pl_gnum1, stride = get_partial_container_stride_and_order(part_zones, container_name, gridLocation, ptp, comm)

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
    i_part = 0
    PT.new_DataArray(fld_name, part1_data[i_part], parent=FS_ep)
  
  # Build PL with the last exchange stride
  if partial_field:
    new_point_list = np.where(part1_stride[0]==1)[0] if part1_data[0].size!=0 else np.empty(0, dtype=np.int32)
    point_list = new_point_list + local_pl_offset(part_zone_ep, LOC_TO_DIM[gridLocation])+1
    new_pl_node = PT.new_PointList(name='PointList', value=point_list.reshape((1,-1), order='F'), parent=FS_ep)

    # Update global numbering in FS
    partial_gnum = create_sub_numbering([part1_ln_to_gn[0][new_point_list]], comm)[0]
    PT.maia.newGlobalNumbering({'Index' : partial_gnum}, parent=FS_ep)



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
                    n_cell, n_face, n_edge, n_vtx,
                    cell_face_idx, cell_face    ,
                    None, None, None,
                    face_vtx_idx , face_vtx     ,
                    cell_ln_to_gn, face_ln_to_gn,
                    None,
                    vtx_ln_to_gn , vtx_coords)

    pdm_ep.selected_lnum_set(i_part, point_list[i_part] - local_pl_offset(part_zone, dim) - 1)

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



def extract_part_from_zsr(part_tree, zsr_name, comm, containers_name=[], **options):
  """Extract the submesh defined by the provided ZoneSubRegion from the input volumic
  partitioned tree.

  Dimension of the output mesh is set up accordingly to the GridLocation of the ZoneSubRegion.
  Submesh is returned as an independant partitioned CGNSTree and includes the relevant connectivities.

  In addition, containers specified in ``containers_name`` list are transfered to the extracted tree.
  Containers to be transfered can be either of label FlowSolution_t or ZoneSubRegion_t.

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

  Extraction can be controled by the optional kwargs:

    - ``graph_part_tool`` (str) -- Partitioning tool used to balance the extracted zones.
      Admissible values are ``hilbert, parmetis, ptscotch``. Note that
      vertex-located extractions require hilbert partitioning. Defaults to ``hilbert``.
  
  Important:
    - Input tree must be unstructured and have a ngon connectivity.
    - Partitions must come from a single initial domain on input tree.
  
  See also:
    :func:`create_extractor_from_zsr` takes the same parameters, excepted ``containers_name``,
    and returns an Extractor object which can be used to exchange containers more than once through its
    ``Extractor.exchange_fields(container_name)`` method.
  
  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #extract_from_zsr@start
      :end-before:  #extract_from_zsr@end
      :dedent: 2
  """
  mlog.info(f"Extract part...")
  start = time.time()
  extractor = create_extractor_from_zsr(part_tree, zsr_name, comm, **options)
  end = time.time()
  mlog.info(f"Extract part done ({end-start:.2f} s) --")

  if containers_name:
    mlog.info(f"Data exchange from initial tree to extracted tree...")
    start = time.time()
    extractor.exchange_fields(containers_name)
    end = time.time()
    mlog.info(f"Exchange done ({end-start:.2f} s) --")

  return extractor.get_extract_part_tree()


def create_extractor_from_zsr(part_tree, zsr_path, comm, **options):
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
                   graph_part_tool=graph_part_tool)



def extract_part_from_bc_name(part_tree, bc_name, comm,
                              transfer_dataset=True,
                              containers_name=[],
                              **options):
  """Extract the submesh defined by the provided BC name from the input volumic
  partitioned tree.

  Behaviour and arguments of this function are similar to those of :func:`extract_part_from_zsr`
  (``zsr_name`` becomes ``bc_name``). Optional ``transfer_dataset`` argument allows to 
  transfer BCDataSet from BC to the extracted mesh (default to ``True``).

  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #extract_from_bc_name@start
      :end-before:  #extract_from_bc_name@end
      :dedent: 2
  """

  # Local copy of the part_tree to add ZSR 
  local_part_tree   = PT.shallow_copy(part_tree)
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(local_part_tree, comm)

  # Adding ZSR to tree
  there_is_bcdataset = False
  for domain, part_zones in part_tree_per_dom.items():
    for part_zone in part_zones:
      bc_n = PT.get_node_from_name_and_label(part_zone, bc_name, 'BC_t') 
      if bc_n is not None:
        zsr_bc_n  = PT.new_ZoneSubRegion(name=bc_name, bc_name=bc_name, parent=part_zone)
        if transfer_dataset:
          assert PT.get_child_from_predicates(bc_n, 'BCDataSet_t/IndexArray_t') is None,\
                 'BCDataSet_t with PointList aren\'t managed'
          ds_arrays = PT.get_children_from_predicates(bc_n, 'BCDataSet_t/BCData_t/DataArray_t')
          for ds_array in ds_arrays:
            PT.new_DataArray(name=PT.get_name(ds_array), value=PT.get_value(ds_array), parent=zsr_bc_n)
          if len(ds_arrays) != 0:
            there_is_bcdataset = True
            # PL and Location is needed for data exchange, but this should be done in ZSR func
            for name in ['PointList', 'GridLocation']:
              PT.add_child(zsr_bc_n, PT.get_child_from_name(bc_n, name))

  if transfer_dataset and comm.allreduce(there_is_bcdataset, MPI.LOR):
    containers_name.append(bc_name)

  return extract_part_from_zsr(local_part_tree, bc_name, comm, containers_name, **options)