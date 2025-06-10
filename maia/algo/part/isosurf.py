import numpy as np
import time

from maia.typing        import *
from maia.pytree.typing import Predicate

import maia.pytree        as PT
import maia.pytree.maia   as MT
import maia.utils.logging as mlog

from maia          import npy_pdm_gnum_dtype   as pdm_gnum_dtype
from maia.transfer import utils                as TEU
from maia.factory  import dist_from_part
from maia.factory.partitioning import part_bound_orient as PBO
from maia.utils    import np_utils, layouts
from .extraction_utils  import local_pl_offset, LOC_TO_DIM, get_partial_container_stride_and_order
from .point_cloud_utils import create_sub_numbering

import Pypdm.Pypdm as PDM

IS_FAM_NAME = PT.pred.label_in(['FamilyName_t', 'AdditionalFamilyName_t'])

def copy_referenced_families(source_base: CGNSTree, target_base: CGNSTree) -> None:
  """ Copy from source_base to target_base the Family_t nodes referenced
  by a (Additional)FamilyName (at zone level) in the target base """
  copied_families = []
  for fam_node in PT.get_children_from_predicates(target_base, ['Zone_t', IS_FAM_NAME]):
    fam_name = PT.get_str_value(fam_node)
    if fam_name not in copied_families:
      copied_families.append(fam_name)
      family_node = PT.get_child_from_predicate(source_base, fam_name)
      PT.add_child(target_base, family_node)


def exchange_field_one_domain(part_zones: List[CGNSPartTree], 
                              iso_part_zone: Optional[CGNSTree], 
                              containers_name: List[str], 
                              comm: MPIComm) -> None:

  for container_name in containers_name:

    # > Retrieve fields name + GridLocation + PointList if container
    #   is not know by every partition
    mask_zone = PT.new_Zone('MaskedZone')
    dist_from_part.discover_nodes_from_matching(mask_zone, part_zones, container_name, comm, \
      child_list=['GridLocation', 'BCRegionName', 'GridConnectivityRegionName'])
  
    fields_query = PT.pred.label_in(['DataArray_t', 'IndexArray_t'])
    dist_from_part.discover_nodes_from_matching(mask_zone, part_zones, [container_name, fields_query], comm)
    mask_container = PT.get_child_from_name(mask_zone, container_name)
    if mask_container is None:
      raise ValueError("[maia-isosurfaces] asked container for exchange is not in tree")

    # > Manage BC and GC ZSR
    ref_zsr_node:Optional[CGNSTree] = mask_container
    bc_descriptor_n = PT.get_child_from_name(mask_container, 'BCRegionName')
    gc_descriptor_n = PT.get_child_from_name(mask_container, 'GridConnectivityRegionName')
    assert not (bc_descriptor_n and gc_descriptor_n)
    if bc_descriptor_n is not None:
      bc_name      = PT.get_str_value(bc_descriptor_n)
      dist_from_part.discover_nodes_from_matching(mask_zone, part_zones, ['ZoneBC_t', bc_name], comm, child_list=['PointList', 'GridLocation_t'])
      ref_zsr_node = PT.get_child_from_predicates(mask_zone, f'ZoneBC_t/{bc_name}')
    elif gc_descriptor_n is not None:
      gc_name      = PT.get_str_value(gc_descriptor_n)
      dist_from_part.discover_nodes_from_matching(mask_zone, part_zones, ['ZoneGridConnectivity_t', gc_name], comm, child_list=['PointList', 'GridLocation_t'])
      ref_zsr_node = PT.get_child_from_predicates(mask_zone, f'ZoneGridConnectivity_t/{gc_name})')
    
    assert ref_zsr_node is not None
    gridLocation = PT.Subset.GridLocation(ref_zsr_node)
    partial_field = PT.get_child_from_name(ref_zsr_node, 'PointList') is not None
    assert gridLocation in ['Vertex', 'FaceCenter', 'CellCenter']


    # > Part1 (ISOSURF) objects definition
    # LN_TO_GN
    _gridLocation    = {"Vertex" : "Vertex", "FaceCenter" : "Element", "CellCenter" : "Cell"}
    
    create_fs = True
    if iso_part_zone is not None:
      elt_n = iso_part_zone if gridLocation!='FaceCenter' else PT.get_child_from_name(iso_part_zone, 'BAR_2')

      create_fs = gridLocation!='FaceCenter' or \
                ( gridLocation=='FaceCenter' and PT.get_child_from_name(iso_part_zone, 'BAR_2') is not None)

      if elt_n is not None :
        part1_ln_to_gn   = [MT.globalnumbering_value(elt_n, _gridLocation[gridLocation])]
      else :
        part1_ln_to_gn   = []

      # > Link between part1 and part2
      part1_maia_iso_zone = PT.find_child_from_name(iso_part_zone, "maia#surface_data")
      if gridLocation=='Vertex' :
        part1_weight        = [PT.get_np_value(PT.find_child_from_name(part1_maia_iso_zone, "Vtx_parent_weight" ))]
        part1_to_part2      = [PT.get_np_value(PT.find_child_from_name(part1_maia_iso_zone, "Vtx_parent_gnum"   ))]
        part1_to_part2_idx  = [PT.get_np_value(PT.find_child_from_name(part1_maia_iso_zone, "Vtx_parent_idx"    ))]
      elif gridLocation=='FaceCenter' :
        # Output should be edge located so check if iso surface locally has edge
        if elt_n is not None:
          part1_to_part2      = [PT.get_np_value(PT.find_child_from_name(part1_maia_iso_zone, "Face_parent_bnd_edges"))] 
          part1_to_part2_idx  = [np.arange(0, part1_ln_to_gn[0].size+1, dtype=np.int32)]
        else:
          part1_to_part2      = []
          part1_to_part2_idx  = []
      elif gridLocation=='CellCenter' :
        part1_to_part2      = [PT.get_np_value(PT.find_child_from_name(part1_maia_iso_zone, "Cell_parent_gnum"))]
        part1_to_part2_idx  = [np.arange(0, part1_ln_to_gn[0].size+1, dtype=np.int32)]
      else:
        raise RuntimeError("Wrong location")


    if iso_part_zone is None:
      part1_ln_to_gn     = []
      part1_to_part2     = []
      part1_to_part2_idx = []


    # > Part2 (VOLUME) objects definition
    part2_ln_to_gn      = list()
    for part_zone in part_zones:
      elt_n            = part_zone if gridLocation!='FaceCenter' else PT.Zone.NGonNode(part_zone)
      part2_ln_to_gn.append(MT.globalnumbering_value(elt_n, _gridLocation[gridLocation]))
        

    # > P2P Object
    ptp = PDM.PartToPart(comm,
                         part1_ln_to_gn,
                         part2_ln_to_gn,
                         part1_to_part2_idx,
                         part1_to_part2     )


    # > FlowSolution node def in isosurf zone
    fs_loc = gridLocation if gridLocation!="FaceCenter" else "EdgeCenter"
    if iso_part_zone is not None and create_fs:
      FS_iso = PT.new_FlowSolution(container_name, loc=fs_loc, parent=iso_part_zone)
    else :
      FS_iso = None # Beware to the loop on containers_name (FS_iso could have been initialised with previous container_name)

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
          fld_data_tmp = PT.get_np_value(fld_n) if fld_n is not None else np.empty(0, dtype=np.float64)
          fld_data.append(fld_data_tmp[pl_gnum1[i_part]])
        p2p_type = PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_GNUM1_COME_FROM
      
      else :
        fld_data = [PT.find_node_from_path(part_zone,fld_path)[1] for part_zone in part_zones]
        stride   = 1
        p2p_type = PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART2

      # Reverse iexch
      req_id = ptp.reverse_iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                                 p2p_type,
                                 fld_data,
                                 part2_stride=stride)
      part1_stride, part1_data = ptp.reverse_wait(req_id)

      # > Placement
      if iso_part_zone is not None and create_fs:
        i_part = 0 # One isosurface partition
        # Ponderation if vertex
        if gridLocation=="Vertex" :
          weighted_fld       = part1_data[i_part]*part1_weight[i_part]
          part1_data[i_part] = np.add.reduceat(weighted_fld, part1_to_part2_idx[i_part][:-1])
        if part1_data[i_part].size!=0:
          PT.new_DataArray(fld_name, part1_data[i_part], parent=FS_iso)

    # Build PL with the last exchange stride
    if partial_field:
      if len(part1_data)!=0 and part1_data[0].size!=0:
        new_point_list = np.where(part1_stride[0]==1)[0]
        point_list = new_point_list + local_pl_offset(iso_part_zone, LOC_TO_DIM[gridLocation]-1)+1
        new_pl_node = PT.new_IndexArray(name='PointList', value=point_list.reshape((1,-1), order='F'), parent=FS_iso)
        partial_part1_lngn = [part1_ln_to_gn[0][new_point_list]] 
      else:
        partial_part1_lngn = []

      # Update global numbering in FS
      partial_gnum = create_sub_numbering(partial_part1_lngn, comm)
      if iso_part_zone is not None and create_fs and len(partial_gnum)!=0:
        MT.new_GlobalNumbering({'Index' : partial_gnum[0]}, parent=FS_iso)

    # Remove node if is empty
    if FS_iso is not None and len(PT.get_children_from_label(FS_iso, 'DataArray_t'))==0:
      assert iso_part_zone is not None
      PT.rm_child(iso_part_zone, FS_iso)


def _exchange_field(part_tree: CGNSPartTree, 
                    iso_part_tree: CGNSPartTree, 
                    containers_name: List[str], 
                    comm: MPIComm) -> None:
  """
  Exchange fields found under each container from part_tree to iso_part_tree
  """
  # Get zones by domains
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm)

  # Loop over domains
  for domain_path, part_zones in part_tree_per_dom.items():
    # Get zone from isosurf (one zone by domain)
    iso_part_zones = TEU.get_partitioned_zones(iso_part_tree, f"{domain_path}")
    iso_part_zone  = iso_part_zones[0] if len(iso_part_zones)!=0 else None
    exchange_field_one_domain(part_zones, iso_part_zone, containers_name, comm)



def iso_surface_one_domain(part_zones: List[CGNSPartTree], 
                           iso_kind: str, 
                           iso_params: Union[List[NDArray], Sequence[float]], 
                           elt_type: str, 
                           graph_part_tool: str, 
                           comm: MPIComm) -> CGNSTree:
  """
  Compute isosurface in a zone
  """ 
  _KIND_TO_SET_FUNC = {"PLANE"   : PDM.IsoSurface.plane_equation_set,
                       "SPHERE"  : PDM.IsoSurface.sphere_equation_set,
                       "ELLIPSE" : PDM.IsoSurface.ellipse_equation_set,
                       "QUADRIC" : PDM.IsoSurface.quadric_equation_set}

  PDM_iso_type = eval(f"PDM._PDM_ISO_SURFACE_KIND_{iso_kind}")
  PDM_elt_type = MT.pdm_elts.cgns_elt_name_to_pdm_element_type(elt_type)

  if iso_kind=="FIELD" : 
    assert isinstance(iso_params, list) and len(iso_params) == len(part_zones)

  if not PBO.orientation_preserved(part_zones, comm):
    if elt_type == 'NGON_n':
      raise RuntimeError("Isosurface and slice functionnalies with elt_typ='NGON_n' require the mesh to have been split with preserve_orientation=True")
    else:
      mlog.warning("Mesh has not been partitioned with preserve_orientation=True, which can lead to inconsistent orientations for isosurface and slice outputs")

  n_part = len(part_zones)

  # Definition of the PDM object IsoSurface
  pdm_isos = PDM.IsoSurface(comm, 3, PDM_iso_type, n_part)
  pdm_isos.isosurf_elt_type_set(PDM_elt_type)
  # > HILBERT : partitioning can be desequilibrated in some 2D case (but parallelism independant)
  pdm_isos.isosurf_part_method_set(eval(f"PDM._PDM_SPLIT_DUAL_WITH_{graph_part_tool.upper()}"))

  if iso_kind=="FIELD":
    for i_part, part_zone in enumerate(part_zones):
      pdm_isos.part_field_set(i_part, iso_params[i_part])
  else:
    _KIND_TO_SET_FUNC[iso_kind](pdm_isos, *iso_params)

  # > Discover BCs and GCs over part_zones
  dist_zone  = PT.new_Zone('Zone')
  # > BCs
  dist_from_part.discover_nodes_from_matching(dist_zone, part_zones, ["ZoneBC_t", 'BC_t'], comm)
  gdom_bcs_path = PT.predicates_to_paths(dist_zone, ['ZoneBC_t','BC_t'])
  n_gdom_bcs = len(gdom_bcs_path)
  # > GCs
  is_gc        = PT.pred.is_gc_with()
  gc_predicate = ['ZoneGridConnectivity_t', PT.pred.is_gc_with(match=False)]
  dist_from_part.discover_nodes_from_matching(dist_zone, part_zones, gc_predicate, comm, get_value='leaf')
  gc_predicate = ['ZoneGridConnectivity_t', MT.pred.is_gc_with(intra=False, match=True)]
  dist_from_part.discover_nodes_from_matching(dist_zone, part_zones, gc_predicate, comm,
        merge_rule=lambda path: MT.conv.get_split_prefix(path), get_value='leaf')
  for jn in PT.iter_children_from_predicates(dist_zone, gc_predicate):
    val = PT.get_str_value(jn)
    PT.set_value(jn, MT.conv.get_part_prefix(val))
  gdom_gcs_path = PT.predicates_to_paths(dist_zone, ['ZoneGridConnectivity_t',is_gc])
  n_gdom_gcs = len(gdom_gcs_path)

  # Loop over domain zones
  for i_part, part_zone in enumerate(part_zones):
    cx, cy, cz = PT.Zone.coordinates(part_zone)
    assert (cx is not None) and (cy is not None) and (cz is not None)
    vtx_coords = np_utils.interweave_arrays([cx,cy,cz])

    ngon  = PT.Zone.NGonNode(part_zone)
    nface = PT.Zone.NFaceNode(part_zone)

    cell_face = MT.Element.connectivity(nface)
    face_vtx  = MT.Element.connectivity(ngon)

    vtx_ln_to_gn, _, face_ln_to_gn, cell_ln_to_gn = TEU.get_entities_numbering(part_zone)
    assert (vtx_ln_to_gn is not None) and (face_ln_to_gn is not None) and (cell_ln_to_gn is not None)

    n_cell = cell_ln_to_gn.shape[0]
    n_face = face_ln_to_gn.shape[0]
    n_edge = 0
    n_vtx  = vtx_ln_to_gn .shape[0]

    # Partition definition for PDM object
    pdm_isos.part_set(i_part,
                      n_cell, n_face, n_edge, n_vtx,
                      cell_face.displs, cell_face.values,
                      None, None, None,
                      face_vtx.displs , face_vtx.values,
                      cell_ln_to_gn, face_ln_to_gn,
                      None,
                      vtx_ln_to_gn, vtx_coords)

    # Add BC information
    if elt_type in ['TRI_3']:
      all_bnd_pl = list()
      for bnd_path in gdom_bcs_path:
        bnd_n = PT.get_node_from_path(part_zone, bnd_path)
        if bnd_n is not None:
          all_bnd_pl.append(PT.get_np_value(PT.find_child_from_name(bnd_n, 'PointList')))
        else :
          all_bnd_pl.append(np.empty((1,0), np.int32))
      for bnd_path in gdom_gcs_path:
        # For gc, we glue the joins that have been splitted during partitioning
        container_name, jn_name = bnd_path.split('/')
        bnd_n_list = PT.get_nodes_from_names(part_zone, [container_name, jn_name+'*'])
        if len(bnd_n_list) > 0:
          pl_val_list = [PT.get_np_value(PT.find_node_from_name(bnd_n, 'PointList')) for bnd_n in bnd_n_list]
          all_bnd_pl.append(np_utils.concatenate_np_arrays(pl_val_list)[1])
        else:
          all_bnd_pl.append(np.empty((1,0), np.int32))

      group_face_idx, group_face = np_utils.concatenate_point_list(all_bnd_pl, dtype=np.int32)
      pdm_isos.isosurf_bnd_set(i_part, n_gdom_bcs+n_gdom_gcs, group_face_idx, group_face)

  # Isosurfaces compute in PDM
  pdm_isos.compute()

  # Mesh build from result
  results = pdm_isos.part_iso_surface_surface_get()
  n_iso_vtx = results['np_vtx_ln_to_gn'].shape[0]
  n_iso_elt = results['np_elt_ln_to_gn'].shape[0]

  # > Zone construction (Zone.P{rank}.N0 because one part of zone on every proc a priori)
  iso_part_zone = PT.new_Zone(MT.conv.add_part_suffix('Zone', comm.Get_rank(), 0),
                              size=[[n_iso_vtx, n_iso_elt, 0]],
                              type='Unstructured')

  # > Grid coordinates
  cx, cy, cz      = layouts.interlaced_to_tuple_coords(results['np_vtx_coord'])
  assert (cx is not None) and (cy is not None) and (cz is not None)
  iso_grid_coord  = PT.new_GridCoordinates(parent=iso_part_zone)
  PT.new_DataArray('CoordinateX', cx, parent=iso_grid_coord)
  PT.new_DataArray('CoordinateY', cy, parent=iso_grid_coord)
  PT.new_DataArray('CoordinateZ', cz, parent=iso_grid_coord)

  # > Elements
  if elt_type in ['TRI_3', 'QUAD_4']:
    elt_n = PT.new_Elements(elt_type,
                            type=elt_type,
                            erange=[1, n_iso_elt],
                            econn=results['np_elt_vtx'],
                            parent=iso_part_zone)
    MT.new_GlobalNumbering({'Element' : results['np_elt_ln_to_gn'],
                                'Sections': results['np_elt_ln_to_gn']}, parent=elt_n)
  else:
    ng_eso = results['np_elt_vtx_idx']
    ng_ec  = results['np_elt_vtx']
    # Retrieve edges on 2D mesh
    edge_data = PDM.compute_face_edge_from_face_vtx(comm, 
                                                    [n_iso_elt], 
                                                    [n_iso_vtx], 
                                                    [ng_eso], 
                                                    [ng_ec], 
                                                    [results['np_elt_ln_to_gn']], 
                                                    [results['np_vtx_ln_to_gn']])[0]
    nb_bar = edge_data['np_edge_ln_to_gn'].size

    bar_n = PT.new_Elements('EdgeElements', 'BAR_2', 
                    erange=[1, nb_bar], 
                    econn=edge_data['np_edge_vtx'], 
                    parent=iso_part_zone)
    MT.new_GlobalNumbering({'Element' : edge_data['np_edge_ln_to_gn']}, parent=bar_n)

    elt_n = PT.new_NGonElements('NGonElements',
                                 erange = [nb_bar+1, nb_bar+n_iso_elt],
                                 ec=ng_ec,
                                 eso=ng_eso,
                                 parent=iso_part_zone)
    MT.new_GlobalNumbering({'Element' : results['np_elt_ln_to_gn']}, parent=elt_n)
  
  # Bnd edges
  if elt_type in ['TRI_3']:
    # > Add element node
    results_edge = pdm_isos.isosurf_bnd_get()
    n_bnd_edge   = results_edge['n_bnd_edge']
    bnd_edge_group_idx = results_edge['bnd_edge_group_idx']
    if n_bnd_edge!=0:
      bar_n = PT.new_Elements('BAR_2', type='BAR_2', 
                              erange=np.array([n_iso_elt+1, n_iso_elt+n_bnd_edge]),
                              econn=results_edge['bnd_edge_vtx'],
                              parent=iso_part_zone)
      MT.new_GlobalNumbering({'Element' : results_edge['bnd_edge_lngn'],
                                  'Sections': results_edge['bnd_edge_lngn']}, parent=bar_n)

    # > Create BC described by edges
    gnum = MT.globalnumbering_value(bar_n, 'Element') if n_bnd_edge!=0 else np.empty(0, dtype=pdm_gnum_dtype)
    for i_group, bc_path in enumerate(gdom_bcs_path):
      n_edge_in_bc = bnd_edge_group_idx[i_group+1]-bnd_edge_group_idx[i_group]
      edge_pl = np.arange(bnd_edge_group_idx[i_group  ],\
                          bnd_edge_group_idx[i_group+1], dtype=np.int32).reshape((1,-1), order='F')+n_iso_elt+1
      partial_gnum = create_sub_numbering([gnum[edge_pl[0]-n_iso_elt-1]], comm)[0]

      if partial_gnum.size != 0:
        zonebc_n = PT.update_child(iso_part_zone, 'ZoneBC', 'ZoneBC_t')  
        bc_n = PT.new_BC(PT.utils.path_tail(bc_path), point_list=edge_pl, loc="EdgeCenter", parent=zonebc_n)
        MT.new_GlobalNumbering({'Index' : partial_gnum}, parent=bc_n)

    for i_group, gc_path in enumerate(gdom_gcs_path):
      gc_name = PT.utils.path_tail(gc_path)
      gc_val  = PT.get_value(PT.find_node_from_path(dist_zone, gc_path))

      i_group+=n_gdom_bcs
      
      n_edge_in_gc = bnd_edge_group_idx[i_group+1]-bnd_edge_group_idx[i_group]
      edge_pl = np.arange(bnd_edge_group_idx[i_group  ],\
                          bnd_edge_group_idx[i_group+1], dtype=np.int32).reshape((1,-1), order='F')+n_iso_elt+1
      partial_gnum = create_sub_numbering([gnum[edge_pl[0]-n_iso_elt-1]], comm)[0]

      if edge_pl.size != 0:
        zonebc_n = PT.update_child(iso_part_zone, 'ZoneBC', 'ZoneBC_t')
        bc_n = PT.new_BC(gc_name, point_list=edge_pl, loc="EdgeCenter", parent=zonebc_n)
        MT.new_GlobalNumbering({'Index' : partial_gnum}, parent=bc_n)


  else:
    n_bnd_edge = 0


  # > LN to GN
  MT.new_GlobalNumbering({'Vertex' : results['np_vtx_ln_to_gn'],
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
  if elt_type in ['TRI_3'] and n_bnd_edge!=0:
    PT.new_DataArray('Face_parent_bnd_edges', results_edge["bnd_edge_face_parent"], parent=maia_iso_zone)

  # > FamilyName(s)
  dist_from_part.discover_nodes_from_matching(iso_part_zone, part_zones, [IS_FAM_NAME],
      comm, get_value='leaf')

  return iso_part_zone



def _iso_surface(part_tree: CGNSPartTree, 
                 iso_field_path: str, 
                 iso_val: float, 
                 elt_type: str, 
                 graph_part_tool: str, 
                 comm: MPIComm) -> CGNSPartTree:

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
      flowsol_node = PT.find_child_from_name(part_zone, fs_name)
      field_node   = PT.find_child_from_name(flowsol_node, field_name)
      assert PT.Subset.GridLocation(flowsol_node) == "Vertex"
      field_values.append(PT.get_np_value(field_node) - iso_val)

    iso_part_zone    = iso_surface_one_domain(part_zones, "FIELD", field_values, elt_type, graph_part_tool, comm)
    PT.set_name(iso_part_zone, MT.conv.add_part_suffix(f'{dom_zone_name}', comm.Get_rank(), 0))
    if PT.Zone.n_cell(iso_part_zone)!=0:
      PT.add_child(iso_part_base,iso_part_zone)

  copy_referenced_families(PT.get_all_CGNSBase_t(part_tree)[0], iso_part_base)

  return iso_part_tree


def iso_surface(part_tree: CGNSPartTree, 
                iso_field: CGNSPath, 
                comm: MPIComm, 
                iso_val: float = 0., 
                containers_name: List[str] = [], 
                **options) -> CGNSPartTree:
  """ Create an isosurface from the provided field and value on the input partitioned tree.

  Isosurface is returned as an independant (2d) partitioned CGNSTree. 

  Important:
    - Input tree must be unstructured and have a ngon connectivity.
    - Input tree must have been partitioned with ``preserve_orientation=True`` partitioning option.
    - Input field for isosurface computation must be located at vertices.
    - This function requires ParaDiGMa access.

  Note:
    - Once created, additional fields can be exchanged from volumic tree to isosurface tree using
      ``_exchange_field(part_tree, iso_part_tree, containers_name, comm)``.
    - If ``elt_type`` is set to 'TRI_3', boundaries from volumic mesh are extracted as edges on
      the isosurface (GridConnectivity_t nodes become BC_t nodes) and FaceCenter fields are allowed to be exchanged.

  Args:
    part_tree     (CGNSPartTree): Partitioned tree on which isosurf is computed. Only U-NGon
      connectivities are managed.
    iso_field     (str)         : Path (starting at Zone_t level) of the field to use to compute isosurface.
    comm          (MPIComm)     : MPI communicator
    iso_val       (float, optional) : Value to use to compute isosurface. Defaults to 0.
    containers_name   (list of str) : List of the names of the FlowSolution_t nodes to transfer
      on the output isosurface tree.
    **options: Options related to plane extraction.
  Returns:
    CGNSTree: Surfacic tree (partitioned)

  Isosurface can be controled thought the optional kwargs:

    - ``elt_type`` (str) -- Controls the shape of elements used to describe
      the isosurface. Admissible values are ``TRI_3, QUAD_4, NGON_n``. Defaults to ``TRI_3``.
    - ``graph_part_tool`` (str) -- Controls the isosurface partitioning tool.
      Admissible values are ``hilbert, parmetis, ptscotch``.
      ``hilbert`` may produce unbalanced partitions for some configurations. Defaults to ``ptscotch``.

  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #compute_iso_surface@start
      :end-before: #compute_iso_surface@end
      :dedent: 2
  """
  MT.check_cgns_part_tree(part_tree)
  start = time.time()

  elt_type        = options.get("elt_type", "TRI_3")
  graph_part_tool = options.get("graph_part_tool", "ptscotch")
  assert(elt_type        in ["TRI_3","QUAD_4","NGON_n"])
  assert(graph_part_tool in ["ptscotch","parmetis","hilbert"])

  # Isosurface extraction
  iso_part_tree = _iso_surface(part_tree, iso_field, iso_val, elt_type, graph_part_tool, comm)
  
  # Interpolation
  if containers_name:
    _exchange_field(part_tree, iso_part_tree, containers_name, comm)
  
  end = time.time()
  mlog.info(f"Isosurface completed ({end-start:.2f} s)")

  return iso_part_tree



def _surface_from_equation(part_tree: CGNSPartTree, 
                           surface_type: str, 
                           equation: Sequence[float], 
                           elt_type: str, 
                           graph_part_tool: str, 
                           comm: MPIComm) -> CGNSPartTree:

  assert(surface_type in ["PLANE","SPHERE","ELLIPSE"])
  assert(elt_type     in ["TRI_3","QUAD_4","NGON_n"])
  
  # Get zones by domains
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm)

  iso_part_tree = PT.new_CGNSTree()

  # Loop over domains : compute isosurf for each
  for domain_path, part_zones in part_tree_per_dom.items():
    dom_base_name, dom_zone_name = domain_path.split('/')
    iso_part_base = PT.update_child(iso_part_tree, dom_base_name, 'CGNSBase_t', [3-1,3])
    iso_part_zone    = iso_surface_one_domain(part_zones, surface_type, equation, elt_type, graph_part_tool, comm)
    PT.set_name(iso_part_zone, MT.conv.add_part_suffix(f'{dom_zone_name}', comm.Get_rank(), 0))

    if PT.Zone.n_cell(iso_part_zone)!=0:
      PT.add_child(iso_part_base,iso_part_zone)

  copy_referenced_families(PT.get_all_CGNSBase_t(part_tree)[0], iso_part_base)

  return iso_part_tree


def plane_slice(part_tree: CGNSPartTree, 
                plane_eq: Sequence[float], 
                comm: MPIComm, 
                containers_name: List[str] = [], 
                **options) -> CGNSPartTree:
  """ Create a slice from the provided plane equation :math:`ax + by + cz - d = 0`
  on the input partitioned tree.

  Slice is returned as an independant (2d) partitioned CGNSTree. See :func:`iso_surface`
  for use restrictions and additional advices.

  Args:
    part_tree    (CGNSPartTree) : Partitioned tree to slice. Only U-NGon connectivities are managed.
    plane_eq     (list of float): List of 4 floats :math:`[a,b,c,d]` defining the plane equation.
    comm          (MPIComm)     : MPI communicator
    containers_name   (list of str) : List of the names of the FlowSolution_t nodes to transfer
      on the output slice tree.
    **options: Options related to plane extraction (see :func:`iso_surface`).
  Returns:
    CGNSTree: Surfacic tree (partitioned)

  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #compute_plane_slice@start
      :end-before: #compute_plane_slice@end
      :dedent: 2
  """
  MT.check_cgns_part_tree(part_tree)
  start = time.time()

  elt_type        = options.get("elt_type", "TRI_3")
  graph_part_tool = options.get("graph_part_tool", "ptscotch")
  assert(elt_type        in ["TRI_3","QUAD_4","NGON_n"])
  assert(graph_part_tool in ["ptscotch","parmetis","hilbert"])

  # Isosurface extraction
  iso_part_tree = _surface_from_equation(part_tree, 'PLANE', plane_eq, elt_type, graph_part_tool, comm)

  # Interpolation
  if containers_name:
    _exchange_field(part_tree, iso_part_tree, containers_name, comm)

  end = time.time()
  mlog.info(f"Plane slice completed ({end-start:.2f} s)")

  return iso_part_tree


def spherical_slice(part_tree: CGNSPartTree, 
                    sphere_eq: Sequence[float], 
                    comm: MPIComm, 
                    containers_name: List[str] = [], 
                    **options) -> CGNSPartTree:
  """ Create a spherical slice from the provided equation
  :math:`(x-x_0)^2 + (y-y_0)^2 + (z-z_0)^2 = R^2`
  on the input partitioned tree.

  Slice is returned as an independant (2d) partitioned CGNSTree. See :func:`iso_surface`
  for use restrictions and additional advices.

  Args:
    part_tree     (CGNSPartTree) : Partitioned tree to slice. Only U-NGon connectivities are managed.
    sphere_eq     (list of float): List of 4 floats :math:`[x_0, y_0, z_0, R]` defining the sphere equation.
    comm          (MPIComm)      : MPI communicator
    containers_name   (list of str) : List of the names of the FlowSolution_t nodes to transfer
      on the output slice tree.
    **options: Options related to plane extraction (see :func:`iso_surface`).
  Returns:
    CGNSTree: Surfacic tree (partitioned)

  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #compute_spherical_slice@start
      :end-before: #compute_spherical_slice@end
      :dedent: 2
  """
  MT.check_cgns_part_tree(part_tree)
  start = time.time()

  elt_type        = options.get("elt_type", "TRI_3")
  graph_part_tool = options.get("graph_part_tool", "ptscotch")
  assert(elt_type        in ["TRI_3","QUAD_4","NGON_n"])
  assert(graph_part_tool in ["ptscotch","parmetis","hilbert"])

  # Isosurface extraction
  iso_part_tree = _surface_from_equation(part_tree, 'SPHERE', sphere_eq, elt_type, graph_part_tool, comm)

  # Interpolation
  if containers_name:
    _exchange_field(part_tree, iso_part_tree, containers_name, comm)

  end = time.time()
  mlog.info(f"Spherical slice completed ({end-start:.2f} s)")

  return iso_part_tree


def elliptical_slice(part_tree: CGNSPartTree, 
                     ellipse_eq: Sequence[float], 
                     comm: MPIComm, 
                     containers_name: List[str] = [], 
                     **options: Any) -> CGNSPartTree:
  """ Create a elliptical slice from the provided equation
  :math:`(x-x_0)^2/a^2 + (y-y_0)^2/b^2 + (z-z_0)^2/c^2 = R^2`
  on the input partitioned tree.

  Slice is returned as an independant (2d) partitioned CGNSTree. See :func:`iso_surface`
  for use restrictions and additional advices.

  Args:
    part_tree     (CGNSPartTree): Partitioned tree to slice. Only U-NGon connectivities are managed.
    ellispe_eq   (list of float): List of 7 floats :math:`[x_0, y_0, z_0, a, b, c, R^2]`
      defining the ellipse equation.
    comm          (MPIComm)     : MPI communicator
    containers_name   (list of str) : List of the names of the FlowSolution_t nodes to transfer
      on the output slice tree.
    **options: Options related to plane extraction (see :func:`iso_surface`).
  Returns:
    CGNSTree: Surfacic tree (partitioned)

  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #compute_elliptical_slice@start
      :end-before: #compute_elliptical_slice@end
      :dedent: 2
  """
  start = time.time()

  elt_type        = options.get("elt_type", "TRI_3")
  graph_part_tool = options.get("graph_part_tool", "ptscotch")
  assert(elt_type        in ["TRI_3","QUAD_4","NGON_n"])
  assert(graph_part_tool in ["ptscotch","parmetis","hilbert"])

  # Isosurface extraction
  iso_part_tree = _surface_from_equation(part_tree, 'ELLIPSE', ellipse_eq, elt_type, graph_part_tool, comm)

  # Interpolation
  if containers_name:
    _exchange_field(part_tree, iso_part_tree, containers_name, comm)

  end = time.time()
  mlog.info(f"Elliptical slice completed ({end-start:.2f} s)")

  return iso_part_tree
