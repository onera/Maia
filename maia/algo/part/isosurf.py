from mpi4py import MPI
import numpy as np

import maia.pytree as PT

from maia.transfer import utils                as TEU
from maia.factory  import dist_from_part
from maia.utils    import np_utils, layouts, py_utils
from .point_cloud_utils import create_sub_numbering

import Pypdm.Pypdm as PDM

familyname_query = lambda n: PT.get_label(n) in ['FamilyName_t', 'AdditionalFamilyName_t']

def copy_referenced_families(source_base, target_base):
  """ Copy from source_base to target_base the Family_t nodes referenced
  by a (Additional)FamilyName (at zone level) in the target base """
  copied_families = []
  for fam_node in PT.get_children_from_predicates(target_base, ['Zone_t', familyname_query]):
    fam_name = PT.get_value(fam_node)
    if fam_name not in copied_families:
      copied_families.append(fam_name)
      family_node = PT.get_child_from_predicate(source_base, fam_name)
      PT.add_child(target_base, family_node)

# =======================================================================================
def exchange_field_one_domain(part_zones, iso_part_zone, containers_name, comm):

  # Part 1 : ISOSURF
  # Part 2 : VOLUME
  for container_name in containers_name :

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
      part1_to_part2_idx  = [np.arange(0, PT.get_value(part1_node_gn_elt).size+1, dtype=np.int32)]
    

    # --- Part2 (VOLUME) objects definition ----------------------------------
    part2_ln_to_gn      = []
    for part_zone in part_zones:
      part2_node_gn_elt = PT.maia.getGlobalNumbering(part_zone, _gridLocation[gridLocation])
      part2_ln_to_gn.append(PT.get_value(part2_node_gn_elt))
        

    # --- P2P Object --------------------------------------------------------------------
    ptp = PDM.PartToPart(comm,
                         part1_ln_to_gn,
                         part2_ln_to_gn,
                         part1_to_part2_idx,
                         part1_to_part2     )


    # --- FlowSolution node def by zone -------------------------------------------------
    FS_iso = PT.new_FlowSolution(container_name, loc=gridLocation, parent=iso_part_zone)


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

      PT.new_DataArray(fld_name, part1_data[i_part], parent=FS_iso)    

# =======================================================================================



# =======================================================================================
def _exchange_field(part_tree, iso_part_tree, containers_name, comm) :
  """
  Exchange fields found under each container from part_tree to iso_part_tree
  """
  # Get zones by domains
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm)

  # Loop over domains
  for domain_path, part_zones in part_tree_per_dom.items():
    # Get zone from isosurf (one zone by domain)
    iso_part_zone = TEU.get_partitioned_zones(iso_part_tree, f"{domain_path}_iso")[0]
    exchange_field_one_domain(part_zones, iso_part_zone, containers_name, comm)

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
  PDM_elt_type = PT.maia.pdm_elts.cgns_elt_name_to_pdm_element_type(elt_type)

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

  # > Discover BCs over part_zones
  dist_zone = PT.new_Zone('Zone')
  dist_from_part.discover_nodes_from_matching(dist_zone, part_zones, ["ZoneBC_t", 'BC_t'], comm)
  bcs_n     = PT.get_children_from_predicates(dist_zone, ['ZoneBC_t','BC_t'])
  gdom_bcs  = [PT.get_name(bc_n) for bc_n in bcs_n]
  n_gdom_bcs= len(gdom_bcs)

  # Loop over domain zones
  for i_part, part_zone in enumerate(part_zones):
    cx, cy, cz = PT.Zone.coordinates(part_zone)
    vtx_coords = np_utils.interweave_arrays([cx,cy,cz])

    ngon  = PT.Zone.NGonNode(part_zone)
    nface = PT.Zone.NFaceNode(part_zone)

    cell_face_idx = PT.get_child_from_name(nface, "ElementStartOffset" )[1]
    cell_face     = PT.get_child_from_name(nface, "ElementConnectivity")[1]
    face_vtx_idx  = PT.get_child_from_name(ngon, "ElementStartOffset" )[1]
    face_vtx      = PT.get_child_from_name(ngon, "ElementConnectivity")[1]

    vtx_ln_to_gn, face_ln_to_gn, cell_ln_to_gn = TEU.get_entities_numbering(part_zone)

    n_cell = cell_ln_to_gn.shape[0]
    n_face = face_ln_to_gn.shape[0]
    n_edge = 0
    n_vtx  = vtx_ln_to_gn .shape[0]

    # Partition definition for PDM object
    pdm_isos.part_set(i_part,
                      n_cell, n_face, n_edge, n_vtx,
                      cell_face_idx, cell_face,
                      None, None, None,
                      face_vtx_idx , face_vtx     ,
                      cell_ln_to_gn, face_ln_to_gn,
                      None,
                      vtx_ln_to_gn, vtx_coords)

    # Add BC information
    zone_bc_n      = PT.get_child_from_label(part_zone, "ZoneBC_t")
    group_face_idx = np.full(n_gdom_bcs+1, 0, dtype=np.int32)
    group_face     = np.empty(0, dtype=np.int32)
    for i_group, bc_name in enumerate(gdom_bcs):
      bc_n  = PT.get_child_from_name(zone_bc_n, bc_name)
      if bc_n is not None:
        bc_pl = PT.get_value(PT.get_child_from_name(bc_n, 'PointList'))
        group_face_idx[i_group+1] = bc_pl.shape[1]
        group_face = np.concatenate([group_face, bc_pl[0]])
    group_face_idx = np.cumsum(group_face_idx, dtype=np.int32)
    pdm_isos.isosurf_bnd_set(i_part, n_gdom_bcs, group_face_idx, group_face)

  # Isosurfaces compute in PDM  
  pdm_isos.compute()

  # Mesh build from result
  results = pdm_isos.part_iso_surface_surface_get()
  n_iso_vtx = results['np_vtx_ln_to_gn'].shape[0]
  n_iso_elt = results['np_elt_ln_to_gn'].shape[0]
  print(n_iso_vtx, n_iso_elt)

  # > Zone construction (Zone.P{rank}.N0 because one part of zone on every proc a priori)
  iso_part_zone = PT.new_Zone(PT.maia.conv.add_part_suffix('Zone', comm.Get_rank(), 0),
                              size=[[n_iso_vtx, n_iso_elt, 0]],
                              type='Unstructured')

  # > Grid coordinates
  cx, cy, cz      = layouts.interlaced_to_tuple_coords(results['np_vtx_coord'])
  iso_grid_coord  = PT.new_GridCoordinates(parent=iso_part_zone)
  PT.new_DataArray('CoordinateX', cx, parent=iso_grid_coord)
  PT.new_DataArray('CoordinateY', cy, parent=iso_grid_coord)
  PT.new_DataArray('CoordinateZ', cz, parent=iso_grid_coord)

  # > Elements
  if elt_type in ['TRI_3', 'QUAD_4']:
    ngon_n = PT.new_Elements( elt_type,
                              type=elt_type,
                              erange=[1, n_iso_elt],
                              econn=results['np_elt_vtx'],
                              parent=iso_part_zone)
  else:
    ngon_n = PT.new_NGonElements( 'NGonElements',
                                  erange = [1, n_iso_elt],
                                  ec=results['np_elt_vtx'],
                                  eso=results['np_elt_vtx_idx'],
                                  parent=iso_part_zone)
  
  # Bnd edges
  if elt_type in ['TRI_3']:
    # > Add element node
    results_edge = pdm_isos.isosurf_bnd_get()
    n_bnd_edge         = results_edge['n_bnd_edge']
    if n_bnd_edge!=0:
      bnd_edge_group_idx = results_edge['bnd_edge_group_idx']
      bar_n = PT.new_Elements('BAR_2', type='BAR_2', 
                              erange=np.array([n_iso_elt+1, n_iso_elt+n_bnd_edge]),
                              econn=results_edge['bnd_edge_vtx'],
                              parent=iso_part_zone)
      PT.maia.newGlobalNumbering({'Element' : results_edge['bnd_edge_lngn'],
                                  'Sections': results_edge['bnd_edge_lngn']}, parent=bar_n)

      # > Create BC described by edges
      zonebc_n = PT.new_ZoneBC(parent=iso_part_zone)
      for i_group, bc_name in enumerate(gdom_bcs):
        n_edge_in_bc = bnd_edge_group_idx[i_group+1]-bnd_edge_group_idx[i_group]
        edge_pl = np.arange(bnd_edge_group_idx[i_group  ],\
                            bnd_edge_group_idx[i_group+1], dtype=np.int32).reshape((1,-1), order='F')+n_iso_elt+1
        gnum    = PT.maia.getGlobalNumbering(bar_n, 'Element')[1]
        partial_gnum = create_sub_numbering([gnum[edge_pl[0]-n_iso_elt-1]], comm)[0]

        if partial_gnum.size!=0:
          bc_n = PT.new_BC(bc_name, point_list=edge_pl, loc="EdgeCenter", parent=zonebc_n)
          PT.maia.newGlobalNumbering({'Index' : partial_gnum}, parent=bc_n)
  else:
    n_bnd_edge = 0

  PT.maia.newGlobalNumbering({'Element' : results['np_elt_ln_to_gn'],
                              'Sections': results['np_elt_ln_to_gn']}, parent=ngon_n)

  # > LN to GN
  PT.maia.newGlobalNumbering({'Vertex' : results['np_vtx_ln_to_gn'],
                              'Cell'   : results['np_elt_ln_to_gn'] }, parent=iso_part_zone)

  # > Link between vol and isosurf
  maia_iso_zone = PT.new_node('maia#surface_data', label='UserDefinedData_t', parent=iso_part_zone)
  results_vtx   = pdm_isos.part_iso_surface_vtx_interpolation_data_get()
  results_geo   = pdm_isos.part_iso_surface_geom_data_get()
  PT.new_DataArray('Cell_parent_gnum' , results    ["np_elt_parent_g_num"]  , parent=maia_iso_zone)
  PT.new_DataArray('Vtx_parent_gnum'  , results_vtx["vtx_volume_vtx_g_num"] , parent=maia_iso_zone)
  PT.new_DataArray('Vtx_parent_idx'   , results_vtx["vtx_volume_vtx_idx"]   , parent=maia_iso_zone)
  PT.new_DataArray('Vtx_parent_weight', results_vtx["vtx_volume_vtx_weight"], parent=maia_iso_zone)
  PT.new_DataArray('Surface'          , results_geo["elt_surface"]          , parent=maia_iso_zone)
  if elt_type in ['TRI_3']:
    PT.new_DataArray('Face_parent_bnd_edges', results_edge["bnd_edge_face_parent"], parent=maia_iso_zone)

  # > FamilyName(s)
  dist_from_part.discover_nodes_from_matching(iso_part_zone, part_zones, [familyname_query],
      comm, get_value='leaf')

  return iso_part_zone
# ---------------------------------------------------------------------------------------
# =======================================================================================



# =======================================================================================
# ---------------------------------------------------------------------------------------
def _iso_surface(part_tree, iso_field_path, iso_val, elt_type, comm):

  fs_name, field_name = iso_field_path.split('/')

  # Get zones by domains
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm)
  
  iso_part_tree = PT.new_CGNSTree()

  # Loop over domains : compute isosurf for each
  for domain_path, part_zones in part_tree_per_dom.items():
    dom_base_name, dom_zone_name = domain_path.split('/')
    iso_part_base = PT.update_child(iso_part_tree, dom_base_name, 'CGNSBase_t', [3-1,3])

    field_values = []
    for part_zone in part_zones:
      # Check : vertex centered solution (PDM_isosurf doesnt work with cellCentered field)
      flowsol_node = PT.get_child_from_name(part_zone, fs_name)
      field_node   = PT.get_child_from_name(flowsol_node, field_name)
      assert PT.Subset.GridLocation(flowsol_node) == "Vertex"
      field_values.append(PT.get_value(field_node) - iso_val)

    iso_part_zone    = iso_surface_one_domain(part_zones, "FIELD", field_values, elt_type, comm)
    iso_part_zone[0] = PT.maia.conv.add_part_suffix(f'{dom_zone_name}_iso', comm.Get_rank(), 0)
    PT.add_child(iso_part_base,iso_part_zone)

  copy_referenced_families(PT.get_all_CGNSBase_t(part_tree)[0], iso_part_base)

  return iso_part_tree
# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
def iso_surface(part_tree, iso_field, comm, iso_val=0., containers_name=[], **options):
  """ Create an isosurface from the provided field and value on the input partitioned tree.

  Isosurface is returned as an independant (2d) partitioned CGNSTree. 

  Important:
    - Input tree must be unstructured and have a ngon connectivity.
    - Input field for isosurface computation must be located at vertices.
    - This function requires ParaDiGMa access.

  Note:
    Once created, additional fields can be exchanged from volumic tree to isosurface tree using
    ``_exchange_field(part_tree, iso_part_tree, containers_name, comm)``

  Args:
    part_tree     (CGNSTree)    : Partitioned tree on which isosurf is computed. Only U-NGon
      connectivities are managed.
    iso_field     (str)         : Path (starting at Zone_t level) of the field to use to compute isosurface.
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

  elt_type = options.get("elt_type", "TRI_3")
  assert(elt_type in ["TRI_3","QUAD_4","NGON_n"])

  # Isosurface extraction
  iso_part_tree = _iso_surface(part_tree, iso_field, iso_val, elt_type, comm)
  
  # Interpolation
  if containers_name:
    _exchange_field(part_tree, iso_part_tree, containers_name, comm)

  return iso_part_tree
# ---------------------------------------------------------------------------------------
# =======================================================================================



# =======================================================================================
# ---------------------------------------------------------------------------------------
def _surface_from_equation(part_tree, surface_type, equation, elt_type, comm):

  assert(surface_type in ["PLANE","SPHERE","ELLIPSE"])
  assert(elt_type     in ["TRI_3","QUAD_4","NGON_n"])
  
  # Get zones by domains
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm)

  iso_part_tree = PT.new_CGNSTree()

  # Loop over domains : compute isosurf for each
  for domain_path, part_zones in part_tree_per_dom.items():
    dom_base_name, dom_zone_name = domain_path.split('/')
    iso_part_base = PT.update_child(iso_part_tree, dom_base_name, 'CGNSBase_t', [3-1,3])
    iso_part_zone    = iso_surface_one_domain(part_zones, surface_type, equation, elt_type, comm)
    iso_part_zone[0] = PT.maia.conv.add_part_suffix(f'{dom_zone_name}_iso', comm.Get_rank(), 0)
    PT.add_child(iso_part_base,iso_part_zone)

  copy_referenced_families(PT.get_all_CGNSBase_t(part_tree)[0], iso_part_base)

  return iso_part_tree
# ---------------------------------------------------------------------------------------
# =======================================================================================



# =======================================================================================
# ---------------------------------------------------------------------------------------
def plane_slice(part_tree, plane_eq, comm, containers_name=[], **options):
  """ Create a slice from the provided plane equation :math:`ax + by + cz - d = 0`
  on the input partitioned tree.

  Slice is returned as an independant (2d) partitioned CGNSTree. See :func:`iso_surface`
  for use restrictions and additional advices.

  Args:
    part_tree     (CGNSTree)    : Partitioned tree to slice. Only U-NGon connectivities are managed.
    sphere_eq     (list of float): List of 4 floats :math:`[a,b,c,d]` defining the plane equation.
    comm          (MPIComm)     : MPI communicator
    containers_name   (list of str) : List of the names of the FlowSolution_t nodes to transfer
      on the output slice tree.
    **options: Options related to plane extraction (see :func:`iso_surface`).
  Returns:
    slice_tree (CGNSTree): Surfacic tree (partitioned)

  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #compute_plane_slice@start
      :end-before: #compute_plane_slice@end
      :dedent: 2
  """
  elt_type = options.get("elt_type", "TRI_3")

  # Isosurface extraction
  iso_part_tree = _surface_from_equation(part_tree, 'PLANE', plane_eq, elt_type, comm)

  # Interpolation
  if containers_name:
    _exchange_field(part_tree, iso_part_tree, containers_name, comm)

  return iso_part_tree
# ---------------------------------------------------------------------------------------
# =======================================================================================



# =======================================================================================
# ---------------------------------------------------------------------------------------
def spherical_slice(part_tree, sphere_eq, comm, containers_name=[], **options):
  """ Create a spherical slice from the provided equation
  :math:`(x-x_0)^2 + (y-y_0)^2 + (z-z_0)^2 = R^2`
  on the input partitioned tree.

  Slice is returned as an independant (2d) partitioned CGNSTree. See :func:`iso_surface`
  for use restrictions and additional advices.

  Args:
    part_tree     (CGNSTree)    : Partitioned tree to slice. Only U-NGon connectivities are managed.
    plane_eq      (list of float): List of 4 floats :math:`[x_0, y_0, z_0, R]` defining the sphere equation.
    comm          (MPIComm)     : MPI communicator
    containers_name   (list of str) : List of the names of the FlowSolution_t nodes to transfer
      on the output slice tree.
    **options: Options related to plane extraction (see :func:`iso_surface`).
  Returns:
    slice_tree (CGNSTree): Surfacic tree (partitioned)

  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #compute_spherical_slice@start
      :end-before: #compute_spherical_slice@end
      :dedent: 2
  """
  elt_type = options.get("elt_type", "TRI_3")

  # Isosurface extraction
  iso_part_tree = _surface_from_equation(part_tree, 'SPHERE', sphere_eq, elt_type, comm)

  # Interpolation
  if containers_name:
    _exchange_field(part_tree, iso_part_tree, containers_name, comm)

  return iso_part_tree
# ---------------------------------------------------------------------------------------
# =======================================================================================



# =======================================================================================
# ---------------------------------------------------------------------------------------
def elliptical_slice(part_tree, ellipse_eq, comm, containers_name=[], **options):
  """ Create a elliptical slice from the provided equation
  :math:`(x-x_0)^2/a^2 + (y-y_0)^2/b^2 + (z-z_0)^2/c^2 = R^2`
  on the input partitioned tree.

  Slice is returned as an independant (2d) partitioned CGNSTree. See :func:`iso_surface`
  for use restrictions and additional advices.

  Args:
    part_tree     (CGNSTree)    : Partitioned tree to slice. Only U-NGon connectivities are managed.
    ellispe_eq   (list of float): List of 7 floats :math:`[x_0, y_0, z_0, a, b, c, R^2]`
      defining the ellipse equation.
    comm          (MPIComm)     : MPI communicator
    containers_name   (list of str) : List of the names of the FlowSolution_t nodes to transfer
      on the output slice tree.
    **options: Options related to plane extraction (see :func:`iso_surface`).
  Returns:
    slice_tree (CGNSTree): Surfacic tree (partitioned)

  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #compute_elliptical_slice@start
      :end-before: #compute_elliptical_slice@end
      :dedent: 2
  """
  elt_type = options.get("elt_type", "TRI_3")

  # Isosurface extraction
  iso_part_tree = _surface_from_equation(part_tree, 'ELLIPSE', ellipse_eq, elt_type, comm)

  # Interpolation
  if containers_name:
    _exchange_field(part_tree, iso_part_tree, containers_name, comm)

  return iso_part_tree
# ---------------------------------------------------------------------------------------
# =======================================================================================
