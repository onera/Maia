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
import maia.pytree                                  as PT
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


# =======================================================================================
def exchange_field_one_domain(part_zones, iso_part_zone, interpolate, comm) :

  # Part 1 : ISOSURF
  # Part 2 : VOLUME
  for container_name in interpolate :

    # --- Get all fields names ------------------------------------------------
    # A MODIFIER
    flds_in_container_names = []
    first_zone_container    = I.getNodeFromName(part_zones[0],container_name)
    for fld_node in I.getNodesFromType(first_zone_container,'DataArray_t'):
      flds_in_container_names.append(fld_node[0])
    

    # Get : Field location
    gridLocation = I.getValue(I.getNodeFromName(first_zone_container,'GridLocation'))
    # Check : correct GridLocation node
    assert(gridLocation in ['Vertex','CellCenter'])


    # --- Part1 (ISOSURF) objects definition ----------------------------------
    # LN_TO_GN
    _gridLocation     = {"Vertex" : "Vertex", "CellCenter" : "Cell"}
    part1_node_gn_elt = PT.maia.getGlobalNumbering(iso_part_zone, _gridLocation[gridLocation])
    part1_ln_to_gn    = [PT.get_value(part1_node_gn_elt)]
    
    # Link between part1 and part2
    part1_maia_iso_zone = PT.get_child_from_name(iso_part_zone, "maia#surface_data")
    if gridLocation=='Vertex' :
      part1_weight        = [PT.get_child_from_name(part1_maia_iso_zone, "Vtx_parent_weight" )[1]]
      part1_to_part2      = [PT.get_child_from_name(part1_maia_iso_zone, "Vtx_parent_gnum"   )[1]]
      part1_to_part2_idx  = [PT.get_child_from_name(part1_maia_iso_zone, "Vtx_parent_idx"    )[1]]
    if gridLocation=='CellCenter' :
      part1_to_part2      = [PT.get_child_from_name(part1_maia_iso_zone, "Cell_parent_gnum")[1]]
      part1_to_part2_idx  = [np.arange(0, part1_ln_to_gn[i_part].shape[0]+1, dtype=np.int32 )]
    

    # --- Part2 (VOLUME) objects definition ----------------------------------
    part2_ln_to_gn      = []
    for i_part, part_zone in enumerate(part_zones):
      vtx_ln_to_gn, face_ln_to_gn, cell_ln_to_gn = TEU.get_entities_numbering(part_zone)
      loc_dict = {"Vertex" : vtx_ln_to_gn, "CellCenter" : cell_ln_to_gn}
      part2_ln_to_gn.append(loc_dict[gridLocation])
        

    # --- P2P Object --------------------------------------------------------------------
    ptp = PDM.PartToPart(comm,
                         part1_ln_to_gn,
                         part2_ln_to_gn,
                         part1_to_part2_idx,
                         part1_to_part2     )


    # --- FlowSolution node def by zone -------------------------------------------------
    FS_iso = I.newFlowSolution(container_name, gridLocation=gridLocation, parent=iso_part_zone)


    # --- Field exchange ----------------------------------------------------------------
    for fld_name in flds_in_container_names:
      fld_path = f"{container_name}/{fld_name}"
      fld_data = [PT.get_node_from_path(part_zone,fld_path)[1] for part_zone in part_zones]

      # Reverse iexch
      req_id = ptp.reverse_iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                                 PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART2,
                                 fld_data,
                                 part2_stride=1)
      part1_strid, part1_data = ptp.reverse_wait(req_id)

      # Placement
      i_part = 0 # One isosurface partition

      # Ponderation if vertex
      if   gridLocation=="Vertex"    :
        weighted_fld        = part1_data[i_part]*part1_weight[i_part]
        part1_data[i_part]  = np.add.reduceat(weighted_fld, part1_to_part2_idx[i_part][:-1])

      I.newDataArray(fld_name, part1_data[i_part], parent=FS_iso)    
    

# =======================================================================================



# =======================================================================================
def _exchange_field(part_tree, part_tree_iso, interpolate, comm) :
  """
  Exchange field between part_tree and part_tree_iso
  for interpolate vol field 
  """

  # Get zones by domains
  part_tree_per_dom = disc.get_parts_per_blocks(part_tree, comm).values()

  # Check : monodomain
  assert(len(part_tree_per_dom)==1)

  # Get zone from isosurf
  iso_part_zone = PT.get_all_Zone_t(part_tree_iso)
  assert(len(iso_part_zone)<=1)
  iso_part_zone = iso_part_zone[0]

  # Loop over domains
  for i_domain, part_zones in enumerate(part_tree_per_dom):
    exchange_field_one_domain(part_zones, iso_part_zone, interpolate, comm)

# =======================================================================================







# =======================================================================================
# ---------------------------------------------------------------------------------------
def iso_surface_one_domain(part_zones, iso_kind, iso_params, elt_type, comm):
  """
  Compute isosurface in a zone
  """ 
  _KIND_TO_SET_FUNC = {"PLANE"   : PDM.IsoSurface.plane_equation_set,
                       "SPHERE"  : PDM.IsoSurface.sphere_equation_set,
                       "ELLIPSE" : PDM.IsoSurface.ellipse_equation_set,
                       "QUADRIC" : PDM.IsoSurface.quadric_equation_set}

  PDM_iso_type = eval(f"PDM._PDM_ISO_SURFACE_KIND_{iso_kind}")
  PDM_elt_type = eval(f"PDM._PDM_MESH_NODAL_{elt_type}") # A CHANGER

  if iso_kind=="FIELD" : 
    assert isinstance(iso_params, list) and len(iso_params) == len(part_zones)


  n_part = len(part_zones)

  # Definition of the PDM object IsoSurface
  pdm_isos = PDM.IsoSurface(comm, 3, PDM_iso_type, n_part)
  pdm_isos.isosurf_elt_type_set(PDM_elt_type)


  if iso_kind=="FIELD":
    for i_part, part_zone in enumerate(part_zones):
      pdm_isos.part_field_set(i_part, iso_params[i_part])
  else:
    _KIND_TO_SET_FUNC[iso_kind](pdm_isos, *iso_params)


  # Loop over domain zones
  for i_part, part_zone in enumerate(part_zones):
    cx, cy, cz = PT.Zone.coordinates(part_zone)
    vtx_coords = np_utils.interweave_arrays([cx,cy,cz])

    ngon  = PT.Zone.NGonNode(part_zone)
    nface = PT.Zone.NFaceNode(part_zone)

    cell_face_idx = I.getNodeFromName1(nface, "ElementStartOffset" )[1]
    cell_face     = I.getNodeFromName1(nface, "ElementConnectivity")[1]
    face_vtx_idx  = I.getNodeFromName1(ngon, "ElementStartOffset" )[1]
    face_vtx      = I.getNodeFromName1(ngon, "ElementConnectivity")[1]

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


  # Isosurfaces compute in PDM  
  pdm_isos.compute()

  # Mesh build from result
  results = pdm_isos.part_iso_surface_surface_get()
  n_iso_vtx = results['np_vtx_ln_to_gn'].shape[0]
  n_iso_elt = results['np_elt_ln_to_gn'].shape[0]

  # > Tree construction
  # iso_part_tree = I.newCGNSTree()
  # iso_part_base = I.newCGNSBase('Base', cellDim=dim-1, physDim=3, parent=iso_part_tree)

  # > Zone construction (Zone.P{rank}.N0 because one part of zone on every proc a priori)
  iso_part_zone = PT.new_Zone(PT.maia.conv.add_part_suffix('Zone', comm.Get_rank(), 0),
                              size=[[n_iso_vtx, n_iso_elt, 0]],
                              type='Unstructured')

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

  # > Link between vol and isosurf
  maia_iso_zone = I.newUserDefinedData('maia#surface_data', parent=iso_part_zone)
  results_vtx   = pdm_isos.part_iso_surface_vtx_interpolation_data_get()
  results_geo   = pdm_isos.part_iso_surface_geom_data_get()
  I.newDataArray('Cell_parent_gnum' , results    ["np_elt_parent_g_num"]  , parent=maia_iso_zone)
  I.newDataArray('Vtx_parent_gnum'  , results_vtx["vtx_volume_vtx_g_num"] , parent=maia_iso_zone)
  I.newDataArray('Vtx_parent_idx'   , results_vtx["vtx_volume_vtx_idx"]   , parent=maia_iso_zone)
  I.newDataArray('Vtx_parent_weight', results_vtx["vtx_volume_vtx_weight"], parent=maia_iso_zone)
  I.newDataArray('Surface'          , results_geo["elt_surface"]          , parent=maia_iso_zone)

  return iso_part_zone
# ---------------------------------------------------------------------------------------
# =======================================================================================








# =======================================================================================
# ---------------------------------------------------------------------------------------
def _iso_surface(part_tree, iso_field_path, iso_val, elt_type, comm):
  """
  Arguments :
   - part_tree : [partitioned tree] from which isosurf is created
   - iso_kind  : [list]             [type_of_isosurf,isosurf_params]
  """
  fs_name, field_name = iso_field_path.split('/')

  # Get zones by domains
  part_tree_per_dom = disc.get_parts_per_blocks(part_tree, comm).values()
  
  # Check : monodomain
  assert(len(part_tree_per_dom)==1)

  iso_part_tree = PT.new_CGNSTree()
  iso_part_base = PT.new_CGNSBase('Base', cell_dim=3-1, phy_dim=3, parent=iso_part_tree)

  # Loop over domains : compute isosurf for each
  for i_domain, part_zones in enumerate(part_tree_per_dom):

    field_values = []
    for part_zone in part_zones:
      # Check : vertex centered solution (PDM_isosurf doesnt work with cellCentered field)
      flowsol_node = PT.get_child_from_name(part_zone, fs_name)
      field_node   = PT.get_child_from_name(flowsol_node, field_name)
      assert PT.Subset.GridLocation(flowsol_node) == "Vertex"
      field_values.append(PT.get_value(field_node) - iso_val)

    iso_part_zone = iso_surface_one_domain(part_zones, "FIELD", field_values, elt_type, comm)
    I._addChild(iso_part_base,iso_part_zone)

  return iso_part_tree
# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
def iso_surface(part_tree, iso_field, comm, iso_val=0., interpolate=None, elt_type="TRIA3"):
  ''' 
  Compute isosurface from field for a partitioned tree. Return partition of the isosurface.

  Args:
    part_tree     (CGNSTree)    : from which isosurf is created
    iso_field     (str)         : Path of the field to use to compute isosurface
    iso_val       (float)       : Value to use to compute isosurface (default = 0)
    interpolate   (list of str) : List of the names of the source FlowSolution_t nodes to transfer.
    elt_type      (str)         : Type of elt in isosurface ("TRIA3","QUAD4","POLY_2D")         
    comm          (MPIComm)     : MPI communicator
  '''

  assert(elt_type in ["TRIA3","QUAD4","POLY_2D"])

  # Isosurface extraction
  iso_part_tree = _iso_surface(part_tree, iso_field, iso_val, elt_type, comm)

  # Interpolation
  if interpolate is not None :
    # Assert ?
    _exchange_field(part_tree, iso_part_tree, interpolate, comm)

  return iso_part_tree
# ---------------------------------------------------------------------------------------
# =======================================================================================





# =======================================================================================
# ---------------------------------------------------------------------------------------
def _iso_plane(part_tree, plane_eq, elt_type, comm):
  """
  Arguments :
   - part_tree : [partitioned tree] from which isosurf is created
   - iso_kind  : [list]             [type_of_isosurf,isosurf_params]
  """
  # Get zones by domains
  part_tree_per_dom = disc.get_parts_per_blocks(part_tree, comm).values()
  
  # Check : monodomain
  assert(len(part_tree_per_dom)==1)

  iso_part_tree = PT.new_CGNSTree()
  iso_part_base = PT.new_CGNSBase('Base', cell_dim=3-1, phy_dim=3, parent=iso_part_tree)

  # Loop over domains : compute isosurf for each
  for i_domain, part_zones in enumerate(part_tree_per_dom):

    iso_part_zone = iso_surface_one_domain(part_zones, "PLANE", plane_eq, elt_type, comm)
    # I._addChild(part_tree_iso,part_zone_iso)
    PT.add_child(iso_part_base,iso_part_zone)

  return iso_part_tree
# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
def iso_plane(part_tree, plane_eq, comm, interpolate=None, elt_type="TRIA3"):
  ''' 
  Compute isosurface from field for a partitioned tree
  Return partition of the isosurface
  Arguments :
    - part_tree   : [partitioned tree] from which isosurf is created
    - plane_eq    : [list]             plane equation [a,b,c,d]
    - interpolate : [container]        
    - elt_type    : [str]              Type of elt in isosurface ("TRIA3","QUAD4","POLY_2D")         
    - comm        : [MPI communicator]
  '''
  assert(elt_type in ["TRIA3","QUAD4","POLY_2D"])

  # Isosurface extraction
  iso_part_tree = _iso_plane(part_tree, plane_eq, elt_type, comm)

  # Interpolation
  if interpolate is not None :
    # Assert ?
    _exchange_field(part_tree, iso_part_tree, interpolate, comm)

  return iso_part_tree
# ---------------------------------------------------------------------------------------
# =======================================================================================





# =======================================================================================
# ---------------------------------------------------------------------------------------
def _iso_sphere(part_tree, sphere_eq, elt_type, comm):
  """
  Arguments :
   - part_tree : [partitioned tree] from which isosurf is created
   - iso_kind  : [list]             [type_of_isosurf,isosurf_params]
  """
  # Get zones by domains
  part_tree_per_dom = disc.get_parts_per_blocks(part_tree, comm).values()
  
  # Check : monodomain
  assert(len(part_tree_per_dom)==1)

  part_tree_iso = I.newCGNSTree()

  # Loop over domains : compute isosurf for each
  for i_domain, part_zones in enumerate(part_tree_per_dom):

    part_zone_iso = iso_surface_one_domain(part_zones, "SPHERE", sphere_eq, elt_type, comm)
    # I._addChild(part_tree_iso,part_zone_iso)
    PT.add_child(part_tree_iso,part_zone_iso)

  return part_tree_iso
# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
def iso_sphere(part_tree, sphere_eq, comm, interpolate=None, elt_type="TRIA3"):
  ''' 
  Compute isosurface from field for a partitioned tree
  Return partition of the isosurface
  Arguments :
    - part_tree   : [partitioned tree] from which isosurf is created
    - sphere_eq   : [list]             plane equation [a,b,c,d]
    - interpolate : [container]        
    - elt_type    : [str]              Type of elt in isosurface ("TRIA3","QUAD4","POLY_2D")         
    - comm        : [MPI communicator]
  '''
  assert(elt_type in ["TRIA3","QUAD4","POLY_2D"])

  # Isosurface extraction
  part_tree_iso = _iso_sphere(part_tree, sphere_eq, elt_type, comm)

  # Interpolation
  if interpolate is not None :
    # Assert ?
    _exchange_field(part_tree, part_tree_iso, interpolate, comm)

  return part_tree_iso
# ---------------------------------------------------------------------------------------
# =======================================================================================








# =======================================================================================
# ---------------------------------------------------------------------------------------
def _iso_ellipse(part_tree, ellipse_eq, elt_type, comm):
  """
  Arguments :
   - part_tree : [partitioned tree] from which isosurf is created
   - iso_kind  : [list]             [type_of_isosurf,isosurf_params]
  """
  # Get zones by domains
  part_tree_per_dom = disc.get_parts_per_blocks(part_tree, comm).values()
  
  # Check : monodomain
  assert(len(part_tree_per_dom)==1)

  part_tree_iso = I.newCGNSTree()

  # Loop over domains : compute isosurf for each
  for i_domain, part_zones in enumerate(part_tree_per_dom):

    part_zone_iso = iso_surface_one_domain(part_zones, "ELLIPSE", ellipse_eq, elt_type, comm)
    # I._addChild(part_tree_iso,part_zone_iso)
    PT.add_child(part_tree_iso,part_zone_iso)

  return part_tree_iso
# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
def iso_ellipse(part_tree, ellipse_eq, comm, interpolate=None, elt_type="TRIA3"):
  ''' 
  Compute isosurface from field for a partitioned tree
  Return partition of the isosurface
  Arguments :
    - part_tree   : [partitioned tree] from which isosurf is created
    - ellipse_eq  : [list]             plane equation [a,b,c,d]
    - interpolate : [container]        
    - elt_type    : [str]              Type of elt in isosurface ("TRIA3","QUAD4","POLY_2D")         
    - comm        : [MPI communicator]
  '''
  assert(elt_type in ["TRIA3","QUAD4","POLY_2D"])

  # Isosurface extraction
  part_tree_iso = _iso_ellipse(part_tree, ellipse_eq, elt_type, comm)

  # Interpolation
  if interpolate is not None :
    # Assert ?
    _exchange_field(part_tree, part_tree_iso, interpolate, comm)

  return part_tree_iso
# ---------------------------------------------------------------------------------------
# =======================================================================================


