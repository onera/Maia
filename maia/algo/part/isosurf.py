'''
NOTES:

[Questions - Julien]
  - API
  - disc.discover_nodes_from_matching() ca sert à quoix ?

[Questions - Bruno] 
  - le type SHPERE fait avec le field ?
  - field nécessaire même pour le plan ?

[A FAIRE] 
  -> prévoir intégration du niveau de l'iso
  -> prévoir intégration de l'interpolation

'''
# import sys, time


# Import from CASSIOPEE
import Converter.PyTree   as C
import Generator.PyTree   as G
import Converter.Internal as I


# Import from MAIA
from   maia.transfer    import utils                as TEU
from   maia.factory     import dist_from_part       as disc
from   maia.factory     import recover_dist_tree    as part_to_dist
import maia.pytree.maia                             as MTM
from   maia.pytree.maia import conventions          as conv
from   maia.pytree.sids import node_inspect         as sids
from   maia.utils       import np_utils,layouts

# Import from PARADIGM
import Pypdm.Pypdm as PDM

# Import NUMPY
import numpy as np



# _NODE_INTERPOLATION = False
_NODE_INTERPOLATION = True




# =======================================================================================
def exchange_field_one_domain_cc(part_zones, part_tree_iso, interpolate, comm) :


  # Part 1 : ISOSURF
  # Part 2 : VOLUME
  
  # Get Isosurf zones (just one normally)
  part_tree_iso_zones = I.getZones(part_tree_iso)
  assert(len(part_tree_iso_zones)==1)
  iso_part_zone = part_tree_iso_zones[0]


  # --- P2P objects definition ----------------------------------------------
  part1_elmt_ln_to_gn = []
  part2_cell_ln_to_gn = []
  part1_to_part2_idx  = []
  part1_to_part2      = []

  for i_part, part_zone in enumerate(part_zones):
    # ISOSURF infos
    # part1_node_gn_cell  = MTM.getGlobalNumbering(part_tree_iso)#, 'Cell') # 
    part1_node_gn       = I.getNodeFromName3(part_tree_iso, ":CGNS#GlobalNumbering")
    part1_node_gn_cell  = I.getNodeFromName3(part1_node_gn, "Cell")
    part1_elmt_ln_to_gn.append(I.getValue(part1_node_gn_cell))
    
    # VOLUME infos
    vtx_ln_to_gn, face_ln_to_gn, cell_ln_to_gn = TEU.get_entities_numbering(part_zone)
    part2_cell_ln_to_gn.append(cell_ln_to_gn)
    
    # LINK ISO/VOL infos
    part1_to_part2.append(I.getNodeFromName(part1_node_gn, "Elt_parent_gnum")[1])
    part1_to_part2_idx.append(np.arange(0, part1_elmt_ln_to_gn[i_part].shape[0]+1, dtype=np.int32 ))


  # --- P2P Object ----------------------------------------------------------------------
  ptp = PDM.PartToPart(comm,
                       part1_elmt_ln_to_gn,
                       part2_cell_ln_to_gn,
                       part1_to_part2_idx ,
                       part1_to_part2     )


  # --- FlowSolution node def by zone ------------------------------------------------------------------
  FS_iso = []
  for i_part, part_zone in enumerate(part_zones):
    FS_iso.append(I.newFlowSolution('FlowSolution', gridLocation="CellCenter", parent=iso_part_zone))

 
  # --- Field exchange ------------------------------------------------------------------
  for ifld,path in enumerate(interpolate):
    fld_vol     = []
    part2_stri  = []

    for i_part, part_zone in enumerate(part_zones):
      path_to_fld = "FlowSolution/"+path.split('/')[1]
      fld_vol .append(I.getNodeFromPath(part_zone, path_to_fld)[1])
      part2_stri.append(np.ones(fld_vol[0].shape[0], dtype=np.int32))

    # RMQ : OK avec une stride constante ?
    req_id = ptp.reverse_iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                               PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART2,
                               fld_vol,
                               part2_stride=part2_stri)
                               # part2_stride=1)

    part1_strid, part1_data = ptp.reverse_wait(req_id)

    for i_part, part_zone in enumerate(part_zones):
      I.newDataArray(path.split('/')[1], part1_data[i_part], parent=FS_iso[i_part])


  return part_tree_iso
# =======================================================================================




# =======================================================================================
def exchange_field_one_domain_nc(part_zones, part_tree_iso, interpolate, comm) :


  # Part 1 : ISOSURF
  # Part 2 : VOLUME
  
  # Get Isosurf zones (just one normally)
  part_tree_iso_zones = I.getZones(part_tree_iso)
  assert(len(part_tree_iso_zones)==1)
  iso_part_zone = part_tree_iso_zones[0]


  # --- P2P objects definition ----------------------------------------------
  part1_vtx_ln_to_gn  = []
  part2_vtx_ln_to_gn  = []
  part1_to_part2_idx  = []
  part1_to_part2      = []
  part1_weight        = []
  
  for i_part, part_zone in enumerate(part_zones):
    
    # ISOSURF infos
    part1_node_gn       = I.getNodeFromName3(part_tree_iso, ":CGNS#GlobalNumbering")
    part1_node_gn_vtx   = I.getNodeFromName(part1_node_gn, "Vertex")
    part1_vtx_ln_to_gn.append(I.getValue(part1_node_gn_vtx))
    
    # VOLUME infos
    vtx_ln_to_gn, face_ln_to_gn, cell_ln_to_gn = TEU.get_entities_numbering(part_zone)
    part2_vtx_ln_to_gn.append(vtx_ln_to_gn)
    
    # LINK ISO/VOL infos
    part1_to_part2.append(    I.getNodeFromName(part1_node_gn, "Vtx_parent_gnum")[1])
    part1_to_part2_idx.append(I.getNodeFromName(part1_node_gn, "Vtx_parent_idx") [1])
    
    # Get weight
    part1_node_gn = I.getNodeFromName3(part_tree_iso, ":CGNS#GlobalNumbering")
    part1_weight.append(I.getNodeFromName(part1_node_gn, "Vtx_parent_weight")[1])


  # --- P2P Object ----------------------------------------------------------------------
  ptp = PDM.PartToPart(comm,
                       part1_vtx_ln_to_gn,
                       part2_vtx_ln_to_gn,
                       part1_to_part2_idx,
                       part1_to_part2     )


  # --- FlowSolution node def by zone ------------------------------------------------------------------
  FS_iso = []
  for i_part, part_zone in enumerate(part_zones):
    FS_iso.append(I.newFlowSolution('FlowSolution', gridLocation="Vertex", parent=iso_part_zone))


  # --- Field exchange ------------------------------------------------------------------
  for ifld,path in enumerate(interpolate):
    fld_vol     = []
    part2_stri  = []

    for i_part, part_zone in enumerate(part_zones):
      fld_vol   .append(I.getNodeFromPath(part_zone, path)[1])
      part2_stri.append(np.ones(fld_vol[0].shape[0], dtype=np.int32))
      
    # Reverse iexch
    # RMQ : OK avec une stride constante ?
    req_id = ptp.reverse_iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                               PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART2,
                               fld_vol,
                               # part2_stride=part2_stri)
                               part2_stride=1)

    # Get result
    part1_strid, part1_data = ptp.reverse_wait(req_id)

    # Interpolation and placement
    for i_part, part_zone in enumerate(part_zones):
      weighted_fld = part1_data[i_part]*part1_weight[i_part]
      part1_fld    = np.add.reduceat(weighted_fld, part1_to_part2_idx[i_part][:-1])
  
      I.newDataArray(path.split('/')[1], part1_fld, parent=FS_iso[i_part])

  return part_tree_iso
# =======================================================================================



# =======================================================================================
def _exchange_field(part_tree, part_tree_iso, interpolate, comm) :
  """
  Exchange field between part_tree and part_tree_iso
  for interpolate vol field 
  """

  # Get zones by domains
  part_tree_per_dom_dict = disc.get_parts_per_blocks(part_tree, comm)#.values()
  # part_tree_per_dom_keys = part_tree_per_dom_dict.keys()
  part_tree_per_dom      = part_tree_per_dom_dict.values()

  # Check : monodomain (PAS PLUTOT SUR LES KEYS ?)
  assert(len(part_tree_per_dom)==1)

  # Loop over domains
  for i_domain, part_zones in enumerate(part_tree_per_dom):
    
    # Isosurf computation for zone
    if (_NODE_INTERPOLATION):
      part_tree_iso = exchange_field_one_domain_nc(part_zones, part_tree_iso, interpolate, comm)
    else :
      part_tree_iso = exchange_field_one_domain_cc(part_zones, part_tree_iso, interpolate, comm)
  
  return part_tree_iso

# =======================================================================================







# =======================================================================================
def iso_surface_one_domain(part_zones, iso_kind, comm):
  """
  Compute isosurface in a zone
  """ 

  # --- Type of isosurf ---------------------------------------------------------------------
  # print("[i] TYPE_of_ISOSURF : ", iso_kind[0],flush=True)
  assert(len(iso_kind   )==2)
  
  fldpath   = "FlowSolution/Density" #-> DEBUG (fonctionne pas si on donne pas le champ)
  iso_value = 0.                        #-> DEBUG (fonctionne pas si on donne pas le champ)

  if   iso_kind[0]=="PLANE" :
    assert(len(iso_kind[1])==4)
    PDM_type  = PDM._PDM_ISO_SURFACE_KIND_PLANE

  elif iso_kind[0]=="SPHERE": 
    assert(len(iso_kind[1])==4)
    PDM_type  = PDM._PDM_ISO_SURFACE_KIND_SPHERE

  elif iso_kind[0]=="ELLIPSE": 
    assert(len(iso_kind[1])==7)
    PDM_type  = PDM._PDM_ISO_SURFACE_KIND_ELLIPSE

  elif iso_kind[0]=="QUADRIC": 
    assert(len(iso_kind[1])==10)
    PDM_type  = PDM._PDM_ISO_SURFACE_KIND_QUADRIC

  elif iso_kind[0]=="HEART": 
    PDM_type  = PDM._PDM_ISO_SURFACE_KIND_HEART

  elif iso_kind[0]=="FIELD" : 
    assert(len(iso_kind[1])==2)
    
    # Check : vertex centered solution
    flowsol_node = I.getNodeFromName(part_zones  ,"FlowSolution")
    gridloc_node = I.getNodeFromName(flowsol_node,"GridLocation")
    # assert(I.getValue(gridloc_node)=="Vertex")
    assert(I.getValue(gridloc_node)=="CellCenter")
    
    PDM_type = PDM._PDM_ISO_SURFACE_KIND_FIELD 
    fldpath   = iso_kind[1][0]
    iso_value = iso_kind[1][1]

  else:
    print("[!][WARNING] isosurface.py : Error in type of IsoSurface ; Check your script")
    return None
  # ------------------------------------------------------------------------------------------


  n_part = len(part_zones)
  dim    = 3 # Mauvaise idée le codage en dur

  # Definition of the PDM object IsoSurface
  pdm_isos = PDM.IsoSurface(comm, dim, PDM_type, n_part)


  # PDM plane/sphere equation definition
  if   iso_kind[0]=="PLANE"   : pdm_isos.plane_equation_set(*iso_kind[1])
  elif iso_kind[0]=="SPHERE"  : pdm_isos.sphere_equation_set(*iso_kind[1])
  elif iso_kind[0]=="ELLIPSE" : pdm_isos.ellipse_equation_set(*iso_kind[1])
  elif iso_kind[0]=="QUADRIC" : pdm_isos.quadric_equation_set(*iso_kind[1])


  # Loop over domain zones
  for i_part, part_zone in enumerate(part_zones):
    # Get NGon + NFac
    gridc_n    = I.getNodeFromName1(part_zone, 'GridCoordinates')
    cx         = I.getNodeFromName1(gridc_n  , 'CoordinateX'    )[1]
    cy         = I.getNodeFromName1(gridc_n  , 'CoordinateY'    )[1]
    cz         = I.getNodeFromName1(gridc_n  , 'CoordinateZ'    )[1]
    vtx_coords = np_utils.interweave_arrays([cx,cy,cz])

    # Julien : fonction pour faire ca ?
    ngons  = [e for e in I.getNodesFromType1(part_zone, 'Elements_t') if sids.Element.CGNSName(e) == 'NGON_n']
    nfaces = [e for e in I.getNodesFromType1(part_zone, 'Elements_t') if sids.Element.CGNSName(e) == 'NFACE_n']
    assert len(nfaces) == len(ngons) == 1

    cell_face_idx = I.getNodeFromName1(nfaces[0], "ElementStartOffset" )[1]
    cell_face     = I.getNodeFromName1(nfaces[0], "ElementConnectivity")[1]
    face_vtx_idx  = I.getNodeFromName1( ngons[0], "ElementStartOffset" )[1]
    face_vtx      = I.getNodeFromName1( ngons[0], "ElementConnectivity")[1]

    vtx_ln_to_gn, face_ln_to_gn, cell_ln_to_gn = TEU.get_entities_numbering(part_zone)

    n_cell = cell_ln_to_gn.shape[0]
    n_face = face_ln_to_gn.shape[0]
    n_edge = 0
    n_vtx  = vtx_ln_to_gn .shape[0]

    # Partition definition for PDM object
    pdm_isos.part_set(i_part,
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

    # Get field from path to compute the isosurf / Placement in PDM object
    if iso_kind[0]=="FIELD" : 
      field    = I.getNodeFromPath(part_zone, fldpath)
      field[1] = field[1] - np.full(field[1].shape[0], iso_value)
      pdm_isos.part_field_set(i_part, field[1])


  # Isosurfaces compute in PDM  
  pdm_isos.compute()


  # Mesh build from result
  results = pdm_isos.part_iso_surface_surface_get()
  n_iso_vtx = results['np_vtx_ln_to_gn'].shape[0]
  n_iso_elt = results['np_elt_ln_to_gn'].shape[0]

  # print(results.keys())

  # > Tree construction
  iso_part_tree = I.newCGNSTree()
  iso_part_base = I.newCGNSBase('Base', cellDim=dim-1, physDim=3, parent=iso_part_tree)

  # > Zone construction (Zone.P{rank}.N0 because one part of zone on every proc a priori)
  iso_part_zone = I.newZone(f'Zone.P{comm.Get_rank()}.N{0}',
                            [[n_iso_vtx, n_iso_elt, 0]],
                            'Unstructured',
                            parent=iso_part_base)

  # > Grid coordinates
  cx, cy, cz      = layouts.interlaced_to_tuple_coords(results['np_vtx_coord'])
  iso_grid_coord  = I.newGridCoordinates(parent=iso_part_zone)
  I.newDataArray('CoordinateX', cx, parent=iso_grid_coord)
  I.newDataArray('CoordinateY', cy, parent=iso_grid_coord)
  I.newDataArray('CoordinateZ', cz, parent=iso_grid_coord)

  # > Elements
  ngon_n = I.newElements('NGonElements', 'NGON', erange = [1, n_iso_elt], parent=iso_part_zone)
  I.newDataArray('ElementConnectivity', results['np_elt_vtx'    ]     , parent=ngon_n)
  I.newDataArray('ElementStartOffset' , results['np_elt_vtx_idx']     , parent=ngon_n)

  gn_elmt = I.newUserDefinedData(':CGNS#GlobalNumbering', parent=ngon_n )
  I.newDataArray('Element', results['np_elt_ln_to_gn']  , parent=gn_elmt)

  # > LN to GN
  gn_zone = I.newUserDefinedData(':CGNS#GlobalNumbering', parent=iso_part_zone)
  I.newDataArray('Vertex'         , results['np_vtx_ln_to_gn']    , parent=gn_zone)
  I.newDataArray('Cell'           , results['np_elt_ln_to_gn']    , parent=gn_zone)

  # I.printTree(iso_part_tree)

  # > Link between vol and isosurf
  if (_NODE_INTERPOLATION):
    print("_NODE_INTERPOLATION",_NODE_INTERPOLATION)
    results_vtx = pdm_isos.part_iso_surface_vtx_interpolation_data_get()
    I.newDataArray('Vtx_parent_idx'   , results_vtx["vtx_volume_vtx_idx"]   , parent=gn_zone)
    I.newDataArray('Vtx_parent_gnum' , results_vtx["vtx_volume_vtx_g_num"] , parent=gn_zone)
    I.newDataArray('Vtx_parent_weight', results_vtx["vtx_volume_vtx_weight"], parent=gn_zone)
  
  else:
    I.newDataArray('Elt_parent_gnum', results["np_elt_parent_g_num"], parent=gn_zone)

  # elmt_parent_gn = results["np_elt_parent_g_num"]
  # print("[DBG] FIN ISOSURF",flush=True)
 
  return iso_part_base
# =======================================================================================








# =======================================================================================
def _iso_surface(part_tree, iso_kind, comm):
  '''
  Arguments :
   - part_tree : [partitioned tree] from which isosurf is created
   - iso_field : [string] used for isosurf computation (FlowSolutionName/FieldName)
   - iso_value : [float] iso surf val
  '''

  # Get zones by domains
  part_tree_per_dom = disc.get_parts_per_blocks(part_tree, comm).values()
  
  # Check : monodomain
  assert(len(part_tree_per_dom)==1)

  # Loop over domains
  part_tree_iso = I.newCGNSTree()
  for i_domain, part_zones in enumerate(part_tree_per_dom):

    # Isosurf computation for zone
    part_zone_iso = iso_surface_one_domain(part_zones, iso_kind, comm)

    I._addChild(part_tree_iso,part_zone_iso)


  return part_tree_iso
# =======================================================================================




# =======================================================================================
def iso_surface(part_tree,isosurf_kind,comm,interpolate=None):
  ''' 
  Compute isosurface from field for a partitioned tree
  Return partition of the isosurface
  Arguments :
    - part_tree     : [partitioned tree] from which isosurf is created
    - isosurf_kind  : [list]             type of isosurface and params
    - interpolate   : [list]             paths to the field to interpolate
    - comm          : [MPI communicator]
  '''
  # Check : format of isosurf_kind
  assert(len(isosurf_kind)==2)

  # Isosurface extraction
  part_tree_iso = _iso_surface(part_tree,isosurf_kind,comm)

  # Interpolation
  if interpolate is not None :
    # Assert ?
    part_tree_iso = _exchange_field(part_tree, part_tree_iso, interpolate, comm)

  return part_tree_iso
# =======================================================================================



# # =======================================================================================
# def slice(part_tree,slice_params,comm,interpolate=None):
#   ''' 
#   Compute isosurface from field for a partitioned tree
#   Return partition of the isosurface
#   Arguments :
#     - part_tree     : [partitioned tree] from which isosurf is created
#     - isosurf_kind  : [list]             type of isosurface and params
#     - interpolate   : [list]             paths to the field to interpolate
#     - comm          : [MPI communicator]
#   '''
#   # Check : format of isosurf_kind
#   assert(len(isosurf_kind)==2)

#   # Isosurface extraction
#   part_tree_iso = _iso_surface(part_tree,isosurf_kind,comm)

#   # Interpolation
#   if interpolate is not None :
#     # Assert ?
#     part_tree_iso = _exchange_field(part_tree, part_tree_iso, interpolate, comm)

#   return part_tree_iso
# # =======================================================================================
