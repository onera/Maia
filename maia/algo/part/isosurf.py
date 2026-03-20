import numpy as np
import os
import time
from mpi4py import MPI

from maia.typing        import *

import maia.pytree        as PT
import maia.pytree.maia   as MT
from   maia.pytree.maia   import pdm_elts
import maia.utils.logging as mlog

import maia
from maia          import npy_pdm_gnum_dtype   as pdm_gnum_dtype
from maia.transfer import utils                as TEU
from maia.factory  import dist_from_part
from maia.factory.partitioning import part_bound_orient as PBO
from maia.utils    import np_utils, layouts, par_utils
from .extraction_utils  import local_pl_offset, LOC_TO_DIM, get_partial_container_stride_and_order
from .point_cloud_utils import create_sub_numbering
from .utils             import _gather_containers_name

import Pypdm.Pypdm as PDM

IS_FAM_NAME = PT.pred.label_in(['FamilyName_t', 'AdditionalFamilyName_t'])
IS_CNT      = PT.pred.label_in(['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t'])

def _ptp_retrieve_part1_to_part2(ptp, gnum2):
  req = ptp.reverse_iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                          PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART2,
                          gnum2)
  try:
    part1_to_part2_idx = ptp.lpart1_to_part2_idx[0].copy()
  except AttributeError:
    part1_to_part2_idx = ptp.get_part1_to_part2_idx()[0]
  part1_to_part2 = ptp.reverse_wait(req)[1][0]
  return part1_to_part2_idx, part1_to_part2

def _set_n_group_face(pdm_isosurface, n_group):
  """ A wrapper to call PDM_isosurface_n_group_set even if function is not available
  through Cython API (PDM < 2.8).
  """
  try:
    pdm_isosurface.n_group_set(PDM._PDM_MESH_ENTITY_FACE, n_group)
  except AttributeError:
    # Do dark magic to access directly C API
    import ctypes
    addr = id(pdm_isosurface)
  
    # Offset for PyObject_HEAD
    offset = ctypes.sizeof(ctypes.c_ssize_t) + ctypes.sizeof(ctypes.c_void_p)
    iso_ptr = ctypes.cast(addr + offset, ctypes.POINTER(ctypes.c_void_p)).contents

    lib = ctypes.CDLL("libpdm.so")
    lib.PDM_isosurface_n_group_set(iso_ptr, PDM._PDM_MESH_ENTITY_FACE, ctypes.c_int(n_group))

def find_matching_edge(all:NDArray, sub:NDArray) -> NDArray:
  """ For each edge in ``sub`` array, retrieve its position in ``all`` array.
  Edges are supposed to exist once in ``all`` array.
  Order does not matter [13, 18] == [18, 13] """
  mmax = all.max()
  all_sorted = np.sort(all.reshape(-1,2), axis=1)
  sub_sorted = np.sort(sub.reshape(-1,2), axis=1)
  key_all = np.empty(all.size // 2, dtype=np.int64)
  key_sub = np.empty(sub.size // 2, dtype=np.int64)
  key_all[:] = all_sorted[:,0] + mmax*all_sorted[:,1] # No overflow because input is int32
  key_sub[:] = sub_sorted[:,0] + mmax*sub_sorted[:,1]
  # Now search
  order = np.argsort(key_all)
  sorted_keys = key_all[order]
  pos = np.searchsorted(sorted_keys, key_sub)
  return order[pos]

def all_containers(tree:CGNSPartTree, comm:MPIComm) -> List[str]:
  all_nodes = list()
  for zone in PT.get_all_Zone_t(tree):
    predicate = IS_CNT & PT.pred.NodePredicate(lambda c : PT.Container.GridLocation(c, zone) != 'EdgeCenter')
    all_nodes.append(PT.get_children_from_predicate(zone, predicate))
  return _gather_containers_name(all_nodes, 'any', comm)

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

  # Create a fake tree for discovering phase, where dtype of arrays is present
  _part_zones = list()
  for pzone in part_zones:
    _pzone = PT.new_node(PT.get_name(pzone), PT.get_label(pzone))
    for container_name in containers_name:
      if (cnt := PT.get_child_from_name(pzone, container_name)) is not None:
        _cnt = PT.new_node(PT.get_name(cnt), PT.get_label(cnt), parent=_pzone)
        for da in PT.get_children_from_label(cnt, "DataArray_t"):
          PT.new_DataArray(PT.get_name(da), value=PT.get_np_value(da).dtype.str, parent=_cnt)
        # If working on linked ZSR, copy PL / GridLoc so we don't need to exchange related subset
        if PT.Container._is_partial(cnt):
          PT.new_IndexArray('PointList', parent=_cnt)
        PT.new_GridLocation(PT.Container.GridLocation(cnt, pzone), parent=_cnt)
    _part_zones.append(_pzone)

  for container_name in containers_name:

    # > Retrieve fields name + GridLocation + PointList if container
    #   is not know by every partition
    mask_zone = PT.new_Zone('MaskedZone')
    dist_from_part.discover_nodes_from_matching(mask_zone, _part_zones, container_name, comm, \
      child_list=['GridLocation', 'BCRegionName', 'GridConnectivityRegionName', 'DataArray_t', 'IndexArray_t'])

    mask_container = PT.get_child_from_name(mask_zone, container_name)
    if mask_container is None:
      raise ValueError(f"[maia-isosurfaces] asked container for exchange '{container_name}' is not in tree")

    partial_field = PT.Container._is_partial(mask_container)
    gridLocation = PT.Container.GridLocation(mask_container, mask_zone)
    assert gridLocation in ['Vertex', 'FaceCenter', 'CellCenter']

    # > Part1 (ISOSURF) objects definition
    # LN_TO_GN
    _gridLocation = {"Vertex" : "Vertex", "FaceCenter" : "Element", "CellCenter" : "Cell"}

    create_container = True
    part1_lids = None
    if iso_part_zone is not None:

      if os.environ.get('MAIA_OLD_ISOSURFACE') is not None:
        elt_n = iso_part_zone if gridLocation!='FaceCenter' else PT.get_child_from_name(iso_part_zone, 'BAR_2')
        create_container = gridLocation!='FaceCenter' or \
                  ( gridLocation=='FaceCenter' and PT.get_child_from_name(iso_part_zone, 'BAR_2') is not None)
      else:
        elt_n = iso_part_zone if gridLocation!='FaceCenter' else PT.get_child_from_predicate(iso_part_zone, PT.pred.is_element_of_type('BAR_2'))

      if elt_n is not None :
        part1_ln_to_gn   = [PT.get_np_value(MT.find_GlobalNumbering(elt_n, _gridLocation[gridLocation]))]
      else :
        part1_ln_to_gn   = []

      # > Link between part1 and part2
      part1_maia_iso_zone = PT.find_child_from_name(iso_part_zone, "maia#surface_data")
      if gridLocation=='Vertex' :
        part1_weight       = [PT.get_np_value(PT.find_child_from_name(part1_maia_iso_zone, "Vtx_parent_weight" ))]
        part1_to_part2     = [PT.get_np_value(PT.find_child_from_name(part1_maia_iso_zone, "Vtx_parent_gnum"   ))]
        part1_to_part2_idx = [PT.get_np_value(PT.find_child_from_name(part1_maia_iso_zone, "Vtx_parent_idx"    ))]
      elif gridLocation=='FaceCenter' :
        # Output should be edge located so check if iso surface locally has edge
        if elt_n is not None:
          part1_to_part2     = [PT.get_np_value(PT.find_child_from_name(part1_maia_iso_zone, "Face_parent_bnd_edges"))]
          if (node := PT.get_child_from_name(part1_maia_iso_zone, "Bnd_edge_to_internal")) is not None:
            # Face input + new API => extract boundary edges gnum using Bnd_edge_to_internal
            part1_lids = PT.get_np_value(node)
            part1_ln_to_gn[0] = part1_ln_to_gn[0][part1_lids]
          part1_to_part2_idx = [np.arange(0, part1_ln_to_gn[0].size+1, dtype=np.int32)]
        else:
          part1_to_part2     = []
          part1_to_part2_idx = []
      elif gridLocation=='CellCenter' :
        part1_to_part2     = [PT.get_np_value(PT.find_child_from_name(part1_maia_iso_zone, "Cell_parent_gnum"))]
        part1_to_part2_idx = [np.arange(0, part1_ln_to_gn[0].size+1, dtype=np.int32)]
      else:
        raise RuntimeError("Wrong location")

    if iso_part_zone is None:
      part1_ln_to_gn     = []
      part1_to_part2     = []
      part1_to_part2_idx = []

    # > Part2 (VOLUME) objects definition
    part2_ln_to_gn = list()
    for part_zone in part_zones:
      elt_n = part_zone if gridLocation!='FaceCenter' else PT.Zone.NGonNode(part_zone)
      part2_ln_to_gn.append(PT.get_np_value(MT.find_GlobalNumbering(elt_n, _gridLocation[gridLocation])))

    # > P2P Object
    ptp = PDM.PartToPart(comm,
                         part1_ln_to_gn,
                         part2_ln_to_gn,
                         part1_to_part2_idx,
                         part1_to_part2)

    # > FlowSolution node def in isosurf zone
    container_loc = gridLocation if gridLocation!="FaceCenter" else "EdgeCenter"
    if iso_part_zone is not None and create_container:
      container_iso = PT.new_node(name=container_name, label=PT.get_label(mask_container), parent=iso_part_zone)
      PT.new_GridLocation(container_loc, parent=container_iso)
    else :
      container_iso = None # Beware to the loop on containers_name (container_iso could have been initialised with previous container_name)

    if partial_field:
      pl_gnum1, stride = get_partial_container_stride_and_order(part_zones, container_name, gridLocation, ptp, comm)

    # > Field exchange
    cnt_data_arrays = PT.get_children_from_label(mask_container, 'DataArray_t')
    if len(cnt_data_arrays)==0:
      mlog.warning(f"{container_name} container seems to have no DataArray_t to exchange between mesh and computed isosurface")

    for fld_node in cnt_data_arrays:
      fld_name = PT.get_name(fld_node)
      fld_path = f"{container_name}/{fld_name}"

      if partial_field:
        # Get field and organize it according to the gnum1_come_from arrays order
        fld_data = list()
        for i_part, part_zone in enumerate(part_zones) :
          fld_n = PT.get_node_from_path(part_zone,fld_path)
          fld_data_tmp = PT.get_np_value(fld_n) if fld_n is not None else np.empty(0, dtype=PT.get_str_value(fld_node))
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
      if iso_part_zone is not None and create_container:
        i_part = 0 # One isosurface partition
        # Ponderation if vertex
        if gridLocation=="Vertex" :
          weighted_fld       = part1_data[i_part]*part1_weight[i_part]
          part1_data[i_part] = np.add.reduceat(weighted_fld, part1_to_part2_idx[i_part][:-1])
        if part1_data[i_part].size!=0:
          PT.new_DataArray(fld_name, part1_data[i_part], parent=container_iso)

    # Build PL with the last exchange stride
    if partial_field and len(cnt_data_arrays)>0:
      if len(part1_data)!=0 and part1_data[0].size!=0:
        # Retrieve elements of part1 having data using recv stride
        # If we have part1_lids indirection (eg when working on edges), use it
        # to build output pointlist
        mask = part1_stride[0] == 1
        if part1_lids is not None:
          new_point_list = part1_lids[mask]
        else:
          new_point_list = np.flatnonzero(mask)
        assert iso_part_zone is not None
        zdim = PT.Zone.CellDimension(iso_part_zone) 
        point_list = new_point_list + local_pl_offset(iso_part_zone, LOC_TO_DIM[zdim+1][gridLocation]-1)+1
        PT.new_IndexArray(name='PointList', value=point_list.reshape((1,-1), order='F'), parent=container_iso)
        partial_part1_lngn = [part1_ln_to_gn[0][mask]]
      else:
        partial_part1_lngn = []

      # Update global numbering in FS
      partial_gnum = create_sub_numbering(partial_part1_lngn, comm)
      if iso_part_zone is not None and create_container and len(partial_gnum)!=0:
        MT.new_GlobalNumbering({'Index' : partial_gnum[0]}, parent=container_iso)

    # Remove node if is empty
    if container_iso is not None and len(PT.get_children_from_label(container_iso, 'DataArray_t'))==0:
      assert iso_part_zone is not None
      PT.rm_child(iso_part_zone, container_iso)


def _exchange_field(part_tree: CGNSPartTree,
                    iso_part_tree: CGNSPartTree,
                    containers_name: List[str],
                    comm: MPIComm) -> None:
  """
  Exchange fields found under each container from part_tree to iso_part_tree
  """
  # Get zones by domains
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm)

  # Multidomain: allow containers_name that appear in at least one initial domain
  containers_name_per_dom = {key : [name for name in containers_name if par_utils.exists_anywhere(parts, name, comm)]
                              for key,parts in part_tree_per_dom.items()}
  for name in containers_name:
    if not any([name in vals for vals in containers_name_per_dom.values()]):
      raise ValueError(f"[maia-isosurfaces] asked container for exchange '{name}' is not in tree")


  # Loop over domains
  for domain_path, part_zones in part_tree_per_dom.items():
    # Get zone from isosurf (one zone by domain)
    iso_part_zones = MT.get_partitioned_zones(iso_part_tree, f"{domain_path}")
    iso_part_zone  = iso_part_zones[0] if len(iso_part_zones)!=0 else None
    exchange_field_one_domain(part_zones, iso_part_zone, containers_name_per_dom[domain_path], comm)



def iso_surface_one_domain_old(part_zones: List[CGNSPartTree],
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
  PDM_elt_type = pdm_elts.cgns_elt_name_to_pdm_element_type(elt_type)

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
  gc_predicate = ['ZoneGridConnectivity_t', PT.pred.is_gc_of_kind(is_1to1=False)]
  dist_from_part.discover_nodes_from_matching(dist_zone, part_zones, gc_predicate, comm, get_value='leaf')
  gc_predicate = ['ZoneGridConnectivity_t', MT.pred.is_gc_of_kind(is_intra=False, is_1to1=True)]
  dist_from_part.discover_nodes_from_matching(dist_zone, part_zones, gc_predicate, comm,
        merge_rule=lambda path: MT.conv.get_split_prefix(path), get_value='leaf')
  for jn in PT.iter_children_from_predicates(dist_zone, gc_predicate):
    val = PT.get_str_value(jn)
    PT.set_value(jn, MT.conv.get_part_prefix(val))
  gdom_gcs_path = PT.predicates_to_paths(dist_zone, ['ZoneGridConnectivity_t', PT.pred.IS_GC])
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
    gnum = MT.Element.globalnumbering(bar_n) if n_bnd_edge!=0 else np.empty(0, dtype=pdm_gnum_dtype)
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

def iso_surface_one_domain_new(part_zones: List[CGNSPartTree],
                               iso_kind: str,
                               iso_params: Union[List[NDArray], Sequence[float]],
                               graph_part_tool: str,
                               comm: MPIComm) -> CGNSTree:
  """
  Compute isosurface in a zone
  """

  PDM_iso_kind = eval(f"PDM.Isosurface.{iso_kind}")

  if iso_kind=="FIELD" :
    assert isinstance(iso_params, list) and len(iso_params) == len(part_zones)

  if not PBO.orientation_preserved(part_zones, comm):
    # For now, NG output only => preserve_orientation=True is mandatory
    raise RuntimeError("Isosurface and slice functionnalies require the mesh to have been split with preserve_orientation=True")


  zdim = comm.allreduce(max((PT.Zone.CellDimension(z) for z in part_zones), default=0), MPI.MAX)
  PDM_MESH_ENTITY_NATIVE_IN  = PDM._PDM_MESH_ENTITY_FACE if zdim == 2 else PDM._PDM_MESH_ENTITY_CELL
  PDM_MESH_ENTITY_NATIVE_OUT = PDM._PDM_MESH_ENTITY_EDGE if zdim == 2 else PDM._PDM_MESH_ENTITY_FACE
  # Definition of the PDM object IsoSurface
  pdm_isos = PDM.Isosurface(zdim, comm)
  pdm_isos.n_part_set(len(part_zones))
  pdm_isos.redistribution_set(PDM.Isosurface.REEQUILIBRATE, eval(f"PDM._PDM_SPLIT_DUAL_WITH_{graph_part_tool.upper()}"))
  pdm_iso = pdm_isos.add(PDM_iso_kind, [0])

  if iso_kind=="FIELD":
    for i_part, part_zone in enumerate(part_zones):
      pdm_isos.pfield_set(pdm_iso, i_part, iso_params[i_part])
  else:
    pdm_isos.equation_set(pdm_iso, iso_params)


  # > Discover BCs and GCs over part_zones
  if zdim == 3:
    dist_zone  = PT.new_Zone('Zone')
    # > BCs
    dist_from_part.discover_nodes_from_matching(dist_zone, part_zones, ["ZoneBC_t", 'BC_t'], comm)
    gdom_bcs_path = PT.predicates_to_paths(dist_zone, ['ZoneBC_t','BC_t'])
    n_gdom_bcs = len(gdom_bcs_path)
    # > GCs
    gc_predicate = ['ZoneGridConnectivity_t', PT.pred.is_gc_of_kind(is_1to1=False)]
    dist_from_part.discover_nodes_from_matching(dist_zone, part_zones, gc_predicate, comm, get_value='leaf')
    gc_predicate = ['ZoneGridConnectivity_t', MT.pred.is_gc_of_kind(is_intra=False, is_1to1=True)]
    dist_from_part.discover_nodes_from_matching(dist_zone, part_zones, gc_predicate, comm,
          merge_rule=lambda path: MT.conv.get_split_prefix(path), get_value='leaf')
    for jn in PT.iter_children_from_predicates(dist_zone, gc_predicate):
      val = PT.get_str_value(jn)
      PT.set_value(jn, MT.conv.get_part_prefix(val))
    gdom_gcs_path = PT.predicates_to_paths(dist_zone, ['ZoneGridConnectivity_t', PT.pred.IS_GC])
    n_gdom_gcs = len(gdom_gcs_path)
    _set_n_group_face(pdm_isos, n_gdom_bcs + n_gdom_gcs)

  keep_alive = list()
  # Loop over domain zones
  for i_part, part_zone in enumerate(part_zones):

    # Create required nodes if not already existing
    if zdim == 2:
      maia.algo.edge_pe_to_ngon(part_zone, None)
    else:
      maia.algo.pe_to_nface(part_zone, None)

    cx, cy, cz = PT.Zone.coordinates(part_zone)
    if cz is None:
      cz = np.zeros_like(cx)
    assert (cx is not None) and (cy is not None) and (cz is not None)
    vtx_coords = np_utils.interweave_arrays([cx,cy,cz])

    ngon = PT.Zone.NGonNode(part_zone)
    face_vtx = MT.Element.connectivity(ngon)

    pdm_isos.pcoordinates_set(i_part, vtx_coords)
    pdm_isos.pconnectivity_set(i_part, PDM._PDM_CONNECTIVITY_TYPE_FACE_VTX, face_vtx.displs, face_vtx.values)
    pdm_isos.ln_to_gn_set(i_part, PDM._PDM_MESH_ENTITY_VTX,  MT.Zone.vtx_globalnumbering(part_zone))
    pdm_isos.ln_to_gn_set(i_part, PDM_MESH_ENTITY_NATIVE_IN, MT.Zone.cell_globalnumbering(part_zone))

    if zdim == 3:
      # Additional set for 3D meshes : cell_face connectivity + face globalnumbering
      nface = PT.Zone.NFaceNode(part_zone)
      cell_face = MT.Element.connectivity(nface)
      face_ln_to_gn = MT.Element.globalnumbering(ngon)

      pdm_isos.pconnectivity_set(i_part, PDM._PDM_CONNECTIVITY_TYPE_CELL_FACE, cell_face.displs, cell_face.values)
      pdm_isos.ln_to_gn_set(i_part, PDM._PDM_MESH_ENTITY_FACE, face_ln_to_gn)

    keep_alive.append(vtx_coords)

    if zdim == 3:
      # Add BC information
      all_bnd_pl = list()
      all_bnd_gn = list()
      for bnd_path in gdom_bcs_path:
        bnd_n = PT.get_node_from_path(part_zone, bnd_path)
        if bnd_n is not None:
          all_bnd_pl.append(PT.get_np_value(PT.find_child_from_name(bnd_n, 'PointList')))
          all_bnd_gn.append(MT.Subset.globalnumbering(bnd_n))
        else :
          all_bnd_pl.append(np.empty((1,0), np.int32))
          all_bnd_gn.append(np.empty(0, pdm_gnum_dtype))
      for bnd_path in gdom_gcs_path:
        # For gc, we glue the joins that have been splitted during partitioning
        container_name, jn_name = bnd_path.split('/')
        bnd_n_list = PT.get_nodes_from_names(part_zone, [container_name, jn_name+'*'])
        if len(bnd_n_list) > 0:
          pl_val_list = [PT.get_np_value(PT.find_node_from_name(bnd_n, 'PointList')) for bnd_n in bnd_n_list]
          gn_val_list = [MT.Subset.globalnumbering(bnd_n) for bnd_n in bnd_n_list]
          all_bnd_pl.append(np_utils.concatenate_np_arrays(pl_val_list)[1])
          all_bnd_gn.append(np_utils.concatenate_np_arrays(gn_val_list)[1])
        else:
          all_bnd_pl.append(np.empty((1,0), np.int32))
          all_bnd_gn.append(np.empty(0, pdm_gnum_dtype))

      group_face_idx, group_face = np_utils.concatenate_point_list(all_bnd_pl, dtype=np.int32)
      _,              group_lngn = np_utils.concatenate_np_arrays(all_bnd_gn, dtype=pdm_gnum_dtype)
      pdm_isos.pgroup_set(i_part, PDM._PDM_MESH_ENTITY_FACE, group_face_idx, group_face, group_lngn)

      keep_alive.extend([group_face_idx, group_face, group_lngn])

  # Isosurfaces compute in PDM
  pdm_isos.part_to_part_enable(pdm_iso, PDM._PDM_MESH_ENTITY_VTX)
  pdm_isos.part_to_part_enable(pdm_iso, PDM._PDM_MESH_ENTITY_EDGE)
  pdm_isos.part_to_part_enable(pdm_iso, PDM._PDM_MESH_ENTITY_FACE)
  pdm_isos.compute(pdm_iso)

  # Mesh build from result
  out_vtx_ln_to_gn = pdm_isos.ln_to_gn_get(pdm_iso, 0, PDM._PDM_MESH_ENTITY_VTX)
  out_elt_ln_to_gn = pdm_isos.ln_to_gn_get(pdm_iso, 0, PDM_MESH_ENTITY_NATIVE_OUT)

  n_iso_vtx = out_vtx_ln_to_gn.shape[0]
  n_iso_elt = out_elt_ln_to_gn.shape[0]

  # > Zone construction (Zone.P{rank}.N0 because one part of zone on every proc a priori)
  iso_part_zone = PT.new_Zone(MT.conv.add_part_suffix('Zone', comm.Get_rank(), 0),
                              size=[[n_iso_vtx, n_iso_elt, 0]],
                              type='Unstructured')

  # > Grid coordinates
  cx, cy, cz      = layouts.interlaced_to_tuple_coords(pdm_isos.pcoordinates_get(pdm_iso, 0))
  assert (cx is not None) and (cy is not None) and (cz is not None)
  iso_grid_coord  = PT.new_GridCoordinates(parent=iso_part_zone)
  PT.new_DataArray('CoordinateX', cx, parent=iso_grid_coord)
  PT.new_DataArray('CoordinateY', cy, parent=iso_grid_coord)
  PT.new_DataArray('CoordinateZ', cz, parent=iso_grid_coord)

  # > Elements
  if zdim == 2:
    _, edge_vtx = pdm_isos.pconnectivity_get(pdm_iso, 0, PDM._PDM_CONNECTIVITY_TYPE_EDGE_VTX)
    bar_n = PT.new_Elements('BAR_2', 'BAR_2',
                    erange=[1, edge_vtx.size // 2],
                    econn=edge_vtx,
                    parent=iso_part_zone)
    MT.new_GlobalNumbering({'Element' : out_elt_ln_to_gn}, parent=bar_n)
  else:
    ng_eso, ng_ec = pdm_isos.pconnectivity_get(pdm_iso, 0, PDM._PDM_CONNECTIVITY_TYPE_FACE_VTX)
    # Retrieve edges on 2D mesh
    edge_data = PDM.compute_face_edge_from_face_vtx(comm,
                                                    [n_iso_elt],
                                                    [n_iso_vtx],
                                                    [ng_eso],
                                                    [ng_ec],
                                                    [out_elt_ln_to_gn],
                                                    [out_vtx_ln_to_gn])[0]
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
    MT.new_GlobalNumbering({'Element' : out_elt_ln_to_gn}, parent=elt_n)

  

  # Bnd edges
  if zdim == 3:
    bnd_group_idx, bnd_group, bnd_lngn = pdm_isos.pgroup_get(pdm_iso, 0, PDM._PDM_MESH_ENTITY_EDGE)
    # Group got from PDM are in "only bnd edges" numbering, but we reconstructed all
    # edges => we need to retrieve matching edge in all edges numbering
    if bnd_group.size > 0:
      bnd_edges = pdm_isos.pconnectivity_get(pdm_iso, 0, PDM._PDM_CONNECTIVITY_TYPE_EDGE_VTX)[1]
      all_edges = edge_data['np_edge_vtx']
      edge_bnd_to_all = find_matching_edge(all_edges, bnd_edges)
      bnd_group = edge_bnd_to_all[bnd_group-1]+1
    else:
      edge_bnd_to_all = np.empty(0, int)
    for i_group, bc_path in enumerate(gdom_bcs_path + gdom_gcs_path):
      n_edge_in_bc = bnd_group_idx[i_group+1]-bnd_group_idx[i_group]
      bnd_pl = np.empty((1, n_edge_in_bc), dtype=np.int32, order='F')
      bnd_pl[0,:] = bnd_group[bnd_group_idx[i_group]:bnd_group_idx[i_group+1]]
      bnd_gnum = bnd_lngn[bnd_group_idx[i_group]:bnd_group_idx[i_group+1]].copy()

      if n_edge_in_bc != 0:
        zonebc_n = PT.update_child(iso_part_zone, 'ZoneBC', 'ZoneBC_t')
        bc_n = PT.new_BC(PT.utils.path_tail(bc_path), point_list=bnd_pl, loc="EdgeCenter", parent=zonebc_n)
        MT.new_GlobalNumbering({'Index' : bnd_gnum}, parent=bc_n)


  # > LN to GN
  MT.new_GlobalNumbering({'Vertex' : out_vtx_ln_to_gn,
                          'Cell'   : out_elt_ln_to_gn}, parent=iso_part_zone)

  # > Link between vol and isosurf
  ptp_elt = pdm_isos.part_to_part_get(pdm_iso, PDM_MESH_ENTITY_NATIVE_OUT)
  elt_part1_to_part2_idx, elt_part1_to_part2 = \
    _ptp_retrieve_part1_to_part2(ptp_elt, [MT.Zone.cell_globalnumbering(z) for z in part_zones])

  if zdim == 3:
    # This one is for boundary edges (we get the id of parent face in volume mesh)
    ptp_edge = pdm_isos.part_to_part_get(pdm_iso, PDM._PDM_MESH_ENTITY_EDGE)
    _, edge_part1_to_part2 = \
      _ptp_retrieve_part1_to_part2(ptp_edge, [MT.Element.globalnumbering(PT.Zone.NGonNode(z)) for z in part_zones])

  # Vertices (with weights)
  ptp_vtx = pdm_isos.part_to_part_get(pdm_iso, PDM._PDM_MESH_ENTITY_VTX)
  vtx_part1_to_part2_idx, vtx_part1_to_part2 = \
    _ptp_retrieve_part1_to_part2(ptp_vtx, [MT.Zone.vtx_globalnumbering(z) for z in part_zones])
  # Assume that vtx weights are well ordered
  vtx_weight = pdm_isos.pparent_weight_get(pdm_iso, 0, PDM._PDM_MESH_ENTITY_VTX)[1]

  maia_iso_zone = PT.new_node('maia#surface_data', label='UserDefinedData_t', parent=iso_part_zone)
  PT.new_DataArray('Cell_parent_gnum',          elt_part1_to_part2,      parent=maia_iso_zone)
  PT.new_DataArray('Vtx_parent_gnum',           vtx_part1_to_part2,      parent=maia_iso_zone)
  PT.new_DataArray('Vtx_parent_idx',            vtx_part1_to_part2_idx,  parent=maia_iso_zone)
  PT.new_DataArray('Vtx_parent_weight',         vtx_weight,              parent=maia_iso_zone)
  if zdim == 3:
    PT.new_DataArray('Face_parent_bnd_edges', edge_part1_to_part2, parent=maia_iso_zone)
    PT.new_DataArray('Bnd_edge_to_internal',  edge_bnd_to_all,     parent=maia_iso_zone)

  # > FamilyName(s)
  dist_from_part.discover_nodes_from_matching(iso_part_zone, part_zones, [IS_FAM_NAME],
      comm, get_value='leaf')

  return iso_part_zone

def iso_surface_one_domain(part_zones: List[CGNSPartTree],
                           iso_kind: str,
                           iso_params: Union[List[NDArray], Sequence[float]],
                           elt_type: str,
                           graph_part_tool: str,
                           comm: MPIComm) -> CGNSTree:

  if os.environ.get('MAIA_OLD_ISOSURFACE') is not None:
    return iso_surface_one_domain_old(part_zones, iso_kind, iso_params, elt_type, graph_part_tool, comm)
  else:
    return iso_surface_one_domain_new(part_zones, iso_kind, iso_params, graph_part_tool, comm)



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
    input_base = PT.find_child_from_name(part_tree, dom_base_name)
    output_dims = PT.get_np_value(input_base) - np.array([1, 0], np.int32)
    iso_part_base = PT.update_child(iso_part_tree, dom_base_name, 'CGNSBase_t', output_dims)

    field_values = []
    for part_zone in part_zones:
      # Check : vertex centered solution (PDM_isosurf doesnt work with cellCentered field)
      flowsol_node = PT.find_child_from_name(part_zone, fs_name)
      field_node   = PT.find_child_from_name(flowsol_node, field_name)
      assert PT.Container.GridLocation(flowsol_node) == "Vertex"
      field_values.append(PT.get_np_value(field_node) - iso_val)

    iso_part_zone    = iso_surface_one_domain(part_zones, "FIELD", field_values, elt_type, graph_part_tool, comm)
    PT.set_name(iso_part_zone, MT.conv.add_part_suffix(f'{dom_zone_name}', comm.Get_rank(), 0))
    if output_dims[1] == 2:
      PT.rm_node_from_path(iso_part_zone, 'GridCoordinates/CoordinateZ')
    if PT.Zone.n_cell(iso_part_zone)!=0:
      PT.add_child(iso_part_base,iso_part_zone)

  copy_referenced_families(PT.get_all_CGNSBase_t(part_tree)[0], iso_part_base)

  return iso_part_tree


def iso_surface(part_tree: CGNSPartTree,
                iso_field: CGNSPath,
                comm: MPIComm,
                iso_val: float = 0.,
                containers_name: Union[List[str], Literal['ALL']] = [],
                **options) -> CGNSPartTree:
  """ Create an isosurface from the provided field and value on the input partitioned tree.

  Isosurface is returned as an independant partitioned CGNSTree.

  Important:
    - Input tree must be unstructured and have a ngon connectivity.
    - Input tree must have been partitioned with ``preserve_orientation=True`` partitioning option.
    - Input field for isosurface computation must be located at vertices.

  Note:
    - Boundaries from volumic mesh are extracted as edges on the isosurface
      (GridConnectivity_t nodes become BC_t nodes) and FaceCenter fields are allowed to be exchanged.
    - Partial or full containers can be transfered on the output isosurface tree.
    - Once created, additional fields can be exchanged from volumic tree to isosurface tree using
      ``_exchange_field(part_tree, iso_part_tree, containers_name, comm)``.

  Args:
    part_tree     (CGNSPartTree): Partitioned tree on which isosurf is computed. Only U-NGon
      connectivities are managed.
    iso_field     (str)         : Path (starting at Zone_t level) of the field to use to compute isosurface.
    comm          (MPIComm)     : MPI communicator
    iso_val       (float, optional) : Value to use to compute isosurface. Defaults to 0.
    containers_name   (list of str or ``'ALL'``) : Name of each container node to transfer
      on the output isosurface tree.
    **options: Options related to plane extraction.
  Returns:
    CGNSTree: Surfacic tree (partitioned)

  Isosurface can be controled thought the optional kwargs:

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
  if containers_name == 'ALL':
    containers_name = all_containers(part_tree, comm)
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
    input_base = PT.find_child_from_name(part_tree, dom_base_name)
    output_dims = PT.get_np_value(input_base) - np.array([1, 0], np.int32)
    iso_part_base = PT.update_child(iso_part_tree, dom_base_name, 'CGNSBase_t', output_dims)
    iso_part_zone    = iso_surface_one_domain(part_zones, surface_type, equation, elt_type, graph_part_tool, comm)
    PT.set_name(iso_part_zone, MT.conv.add_part_suffix(f'{dom_zone_name}', comm.Get_rank(), 0))

    if output_dims[1] == 2:
      PT.rm_node_from_path(iso_part_zone, 'GridCoordinates/CoordinateZ')
    if PT.Zone.n_cell(iso_part_zone)!=0:
      PT.add_child(iso_part_base,iso_part_zone)

  copy_referenced_families(PT.get_all_CGNSBase_t(part_tree)[0], iso_part_base)

  return iso_part_tree


def plane_slice(part_tree: CGNSPartTree,
                plane_eq: Sequence[float],
                comm: MPIComm,
                containers_name: Union[List[str], Literal['ALL']] = [],
                **options) -> CGNSPartTree:
  """ Create a slice from the provided plane equation :math:`ax + by + cz - d = 0`
  on the input partitioned tree.

  Slice is returned as an independant partitioned CGNSTree. See :func:`iso_surface`
  for use restrictions and additional advices.

  Args:
    part_tree    (CGNSPartTree) : Partitioned tree to slice. Only U-NGon connectivities are managed.
    plane_eq     (list of float): List of 4 floats :math:`[a,b,c,d]` defining the plane equation.
    comm          (MPIComm)     : MPI communicator
    containers_name   (list of str or ``'ALL'``) : Name of each container node to transfer
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
  if containers_name == 'ALL':
    containers_name = all_containers(part_tree, comm)
  if containers_name:
    _exchange_field(part_tree, iso_part_tree, containers_name, comm)

  end = time.time()
  mlog.info(f"Plane slice completed ({end-start:.2f} s)")

  return iso_part_tree


def spherical_slice(part_tree: CGNSPartTree,
                    sphere_eq: Sequence[float],
                    comm: MPIComm,
                    containers_name: Union[List[str], Literal['ALL']] = [],
                    **options) -> CGNSPartTree:
  """ Create a spherical slice from the provided equation
  :math:`(x-x_0)^2 + (y-y_0)^2 + (z-z_0)^2 = R^2`
  on the input partitioned tree.

  Slice is returned as an independant partitioned CGNSTree. See :func:`iso_surface`
  for use restrictions and additional advices.

  Args:
    part_tree     (CGNSPartTree) : Partitioned tree to slice. Only U-NGon connectivities are managed.
    sphere_eq     (list of float): List of 4 floats :math:`[x_0, y_0, z_0, R]` defining the sphere equation.
    comm          (MPIComm)      : MPI communicator
    containers_name   (list of str or ``'ALL'``) : Name of each container node to transfer
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
  if containers_name == 'ALL':
    containers_name = all_containers(part_tree, comm)
  if containers_name:
    _exchange_field(part_tree, iso_part_tree, containers_name, comm)

  end = time.time()
  mlog.info(f"Spherical slice completed ({end-start:.2f} s)")

  return iso_part_tree


def elliptical_slice(part_tree: CGNSPartTree,
                     ellipse_eq: Sequence[float],
                     comm: MPIComm,
                     containers_name: Union[List[str], Literal['ALL']] = [],
                     **options: Any) -> CGNSPartTree:
  """ Create a elliptical slice from the provided equation
  :math:`(x-x_0)^2/a^2 + (y-y_0)^2/b^2 + (z-z_0)^2/c^2 = R^2`
  on the input partitioned tree.

  Slice is returned as an independant partitioned CGNSTree. See :func:`iso_surface`
  for use restrictions and additional advices.

  Args:
    part_tree     (CGNSPartTree): Partitioned tree to slice. Only U-NGon connectivities are managed.
    ellispe_eq   (list of float): List of 7 floats :math:`[x_0, y_0, z_0, a, b, c, R^2]`
      defining the ellipse equation.
    comm          (MPIComm)     : MPI communicator
    containers_name   (list of str or ``'ALL'``) : Name of each container node to transfer
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
  if containers_name == 'ALL':
    containers_name = all_containers(part_tree, comm)
  if containers_name:
    _exchange_field(part_tree, iso_part_tree, containers_name, comm)

  end = time.time()
  mlog.info(f"Elliptical slice completed ({end-start:.2f} s)")

  return iso_part_tree
