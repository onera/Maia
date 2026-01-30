import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT
from maia import npy_pdm_gnum_dtype as pdm_dtype

from maia.utils                      import np_utils
from maia.transfer                   import protocols as EP
from maia.transfer                   import utils as tr_utils
from maia.factory.dist_from_part     import discover_nodes_from_matching, get_parts_per_blocks
from maia.algo.part.extract_boundary import extract_surf_from_bc
from maia.algo.part.geometry         import _compute_elements_center

from .point_cloud_utils              import get_point_cloud

import Pypdm.Pypdm as PDM

from maia.typing import *

IS_BND = PT.pred.label_in(['BC_t', 'GridConnectivity_t', 'GridConnectivity1to1_t'])

def _are_same_perio_abs(first: PT.PeriodicValues, second: PT.PeriodicValues) -> bool:
  """ Return True if the two periodic transformation are the same in absolute value"""
  first_center, first_angle, first_trans = first
  second_center, second_angle, second_trans = second
  if np.allclose(first_center, second_center):
    if np.allclose(first_angle, second_angle) and np.allclose(first_trans, second_trans):
      return True
    if np.allclose(first_angle, -second_angle) and np.allclose(first_trans, -second_trans):
      return True
  return False
      
def _create_output_container(zone, point_cloud, out_fs_name):

  if point_cloud in ['Vertex', 'CellCenter']:
    output_loc = point_cloud
  else:
    output_loc = PT.Container.GridLocation(PT.get_child_from_name(zone, point_cloud))
  
  # Test if FlowSolution already exists or create it
  fs_node = PT.get_child_from_name(zone, out_fs_name)
  if fs_node is None:
    fs_node = PT.new_DiscreteData(name=out_fs_name, loc=output_loc, parent=zone)
  assert PT.Container.GridLocation(fs_node) == output_loc

  return fs_node

def _get_output_shape(zone, out_container):
  output_loc = PT.Container.GridLocation(out_container)
  if output_loc == "CellCenter":
    shape = PT.Zone.CellSize(zone)
  elif output_loc == "Vertex":
    shape = PT.Zone.VertexSize(zone)
  else:
    raise RuntimeError("Unmanaged output location")
  return shape


def _shift_ids(part_dict:Dict[str, NDArray],
                face_offset:int,
                vtx_offset:int) -> Dict[str, NDArray]:

  new_part = dict()
  for key, val in part_dict.items():
    if key == 'face_lngn':
      new_part[key] = val + face_offset
    elif key == 'vtx_lngn':
      new_part[key] = val + vtx_offset
    else:
      new_part[key] = val
  return new_part

def _apply_perio(part_dict:Dict[str, NDArray],
                  perio) -> Dict[str, NDArray]:

  coords = part_dict['vtx_coords']
  cx, cy, cz = np_utils.transform_cart_vectors(coords[0::3], coords[1::3], coords[2::3],
                                                perio[2], perio[0], perio[1]) #Perio is center, angle, trans
  new_coords = np_utils.interweave_arrays([cx, cy, cz])

  new_part = {key: val for key, val in part_dict.items()}
  new_part['vtx_coords'] = new_coords
  
  return new_part

def detect_perio(part_tree:CGNSPartTree, comm:MPIComm) -> Dict[str, List[PT.PeriodicValues]]:
  """
  Create dist_zone_path -> [periodicities] for input tree
  """
  skeleton_tree = PT.new_CGNSTree()
  discover_nodes_from_matching(skeleton_tree, [part_tree], 'CGNSBase_t/Zone_t', comm,
      merge_rule = lambda path: MT.conv.get_part_prefix(path))

  gc_predicate = ['ZoneGridConnectivity_t', MT.pred.is_gc_of_kind(is_intra=False)]
  
  # Recover existing periodicities
  for dist_zone_path in PT.predicates_to_paths(skeleton_tree, 'CGNSBase_t/Zone_t'):
    dist_zone  = PT.find_node_from_path(skeleton_tree, dist_zone_path)
    part_zones = tr_utils.get_partitioned_zones(part_tree, dist_zone_path)

    discover_nodes_from_matching(dist_zone, part_zones, gc_predicate, comm,
      child_list=['GridConnectivityProperty_t', 'GridConnectivityType_t'],
      merge_rule=lambda path: MT.conv.get_split_prefix(path), get_value='leaf')
    #After GC discovery, cleanup donor name suffix
    for jn in PT.iter_children_from_predicates(dist_zone, gc_predicate):
      val = PT.get_str_value(jn)
      PT.set_value(jn, MT.conv.get_part_prefix(val))

  grouped_zone_paths = PT.Tree.find_connected_zones(skeleton_tree)
  periodicities_per_path = dict()
  for group in grouped_zone_paths:
    fake_tree = PT.new_CGNSTree()
    fake_base = PT.new_CGNSBase(group[0].split('/')[0], parent=fake_tree)
    for zone_path in group:
      PT.add_child(fake_base, PT.get_node_from_path(skeleton_tree, zone_path))
    all_periodicities, _ = PT.Tree.find_periodic_jns(fake_tree)
    group_periodicities:List[PT.PeriodicValues] = []
    for perio_val in all_periodicities:
      for u_perio in group_periodicities:
        if _are_same_perio_abs(perio_val, u_perio):
          break
      else:
        group_periodicities.append(perio_val)

    periodicities_per_path.update({path: group_periodicities for path in group})

  return periodicities_per_path

def _wd_setup_surf_mesh(surf_parts_per_dom, walldist, periodicities, comm: MPIComm):
  """
  Setup the surfacic mesh for wall distance computing
  """
  #This will concatenate part data of all initial domains
  all_parts_dict = []

  _n_face_bnd_tot_idx = [0]
  _n_vtx_bnd_tot_idx = [0]

  for dist_zone_path, surf_zones in surf_parts_per_dom.items():

    domain_parts = list()
    for surf_zone in surf_zones:
      pred = PT.pred.label_is('Elements_t') & PT.pred.NodePredicate(lambda n : PT.Element.Dimension(n)==PT.Zone.CellDimension(surf_zone))
      elts = PT.get_children_from_predicate(surf_zone, pred)
      assert len(elts) == 1, "Mutliple elt nodes not managed"
      elt = MT.Element.connectivity(elts[0])
      domain_parts.append(
        {'face_vtx_idx' : elt.displs,
        'face_vtx' : elt.values,
        'face_lngn' : MT.globalnumbering_value(surf_zone, 'Cell'),
        'vtx_coords' : np_utils.interweave_arrays(PT.Zone.coordinates(surf_zone)), #type: ignore[arg-type]
        'vtx_lngn' : MT.globalnumbering_value(surf_zone, 'Vertex')})


    domain_nface = MT.Zone.n_cell(surf_zones, comm)
    domain_nvtx  = MT.Zone.n_vtx(surf_zones, comm)

    all_parts_dict.extend([_shift_ids(part, _n_face_bnd_tot_idx[-1], _n_vtx_bnd_tot_idx[-1]) \
                            for part in domain_parts])

    _n_face_bnd_tot_idx.append(_n_face_bnd_tot_idx[-1] + domain_nface)
    _n_vtx_bnd_tot_idx.append(_n_vtx_bnd_tot_idx[-1] + domain_nvtx)
    
    parts_surf_to_dupl_l = [domain_parts]
    for perio_val in periodicities.get(dist_zone_path, []):
      perio_val_opp = PT.PeriodicValues(perio_val[0], -perio_val[1], -perio_val[2]) #Center, angle, translation

      parts_surf_to_dupl_next_l = []
      for parts_surf_to_dupl in parts_surf_to_dupl_l:
        parts_surf_to_dupl_next_l.append(parts_surf_to_dupl)
        
        # Apply periodicity to input partitions, without shifting gnums,
        # and add result to next duplication
        dupl_parts_surf = [_apply_perio(part, perio_val) for part in parts_surf_to_dupl]
        parts_surf_to_dupl_next_l.append(dupl_parts_surf)
        # Now shift gnums and register in all domain list
        shifted_dupl_parts_surf = [_shift_ids(part, _n_face_bnd_tot_idx[-1], _n_vtx_bnd_tot_idx[-1]) \
                                                    for part in dupl_parts_surf] # Shift
        all_parts_dict.extend(shifted_dupl_parts_surf)

        # Update shift values (reuse nface/nvtx, which are unchanged)
        _n_face_bnd_tot_idx.append(_n_face_bnd_tot_idx[-1] + domain_nface)
        _n_vtx_bnd_tot_idx.append(_n_vtx_bnd_tot_idx[-1] + domain_nvtx)

        # Same with opposite periodicity

        dupl_parts_surf = [_apply_perio(part, perio_val_opp) for part in parts_surf_to_dupl]# Apply periodicity
        parts_surf_to_dupl_next_l.append(dupl_parts_surf)

        shifted_dupl_parts_surf = [_shift_ids(part, _n_face_bnd_tot_idx[-1], _n_vtx_bnd_tot_idx[-1]) \
                                                    for part in dupl_parts_surf] # Shift
        all_parts_dict.extend(shifted_dupl_parts_surf)
        
        _n_face_bnd_tot_idx.append(_n_face_bnd_tot_idx[-1] + domain_nface)
        _n_vtx_bnd_tot_idx.append(_n_vtx_bnd_tot_idx[-1] + domain_nvtx)

      parts_surf_to_dupl_l = parts_surf_to_dupl_next_l

  #Get global data (total number of faces / vertices)
  #This create the surf_mesh objects in PDM, thus it must be done before surf_mesh_part_set
  walldist.n_part_surf = len(all_parts_dict)
  walldist.surf_mesh_global_data_set()
  
  #Setup partitions
  keep_alive = list()
  for i_part, part in enumerate(all_parts_dict):
    keep_alive.append(part)
    walldist.surf_mesh_part_set(i_part, part['face_lngn'].size,
                                        part['face_vtx_idx'],
                                        part['face_vtx'],
                                        part['face_lngn'],
                                        part['vtx_lngn'].size,
                                        part['vtx_coords'],
                                        part['vtx_lngn'])

  
  n_dupl_per_dom = np.array([3**len(periodicities.get(key, [])) for key in surf_parts_per_dom])
  all_dom_ids = np_utils.repeated_arange(n_dupl_per_dom)
  
  offsets = {'dom_id'      : all_dom_ids,
             'face_offset' : _n_face_bnd_tot_idx,
             'vtx_offset'  : _n_vtx_bnd_tot_idx}

  return keep_alive, offsets
  
def _wd_setup_vol_mesh(part_zones: List[CGNSPartTree], walldist):
  """
  Setup the volumic mesh for wall distance computing (only for propagation method)
  """
  #Setup global data
  walldist.vol_mesh_global_data_set()
  keep_alive = list()

  for i_part, part_zone in enumerate(part_zones):

    coords = [c for c in PT.Zone.coordinates(part_zone) if c is not None]
    assert len(coords) == 3, "PhyDim != 3 is not supported"
    vtx_coords = np_utils.interweave_arrays(coords)

    ngon = PT.Zone.NFaceNode(part_zone)
    face_vtx = MT.Element.connectivity(ngon)

    nface = PT.Zone.NFaceNode(part_zone)
    cell_face = MT.Element.connectivity(nface)

    vtx_ln_to_gn, _, face_ln_to_gn, cell_ln_to_gn = tr_utils.get_entities_numbering(part_zone)
    assert (vtx_ln_to_gn is not None) and (face_ln_to_gn is not None) and (cell_ln_to_gn is not None)

    n_vtx  = vtx_ln_to_gn .shape[0]
    n_cell = cell_ln_to_gn.shape[0]
    n_face = face_ln_to_gn.shape[0]

    center_cell = _compute_elements_center(part_zone, 'CellCenter')
    assert(center_cell.size == 3*n_cell)

    # Keep numpy alive
    for array in (cell_face, cell_ln_to_gn, face_vtx, face_ln_to_gn, \
        vtx_coords, vtx_ln_to_gn, center_cell):
      keep_alive.append(array)

    walldist.vol_mesh_part_set(i_part,
                                n_cell, cell_face.displs, cell_face.values, center_cell, cell_ln_to_gn,
                                n_face, face_vtx.displs, face_vtx.values, face_ln_to_gn,
                                n_vtx, vtx_coords, vtx_ln_to_gn)
  return keep_alive


def _wd_get(walldist, i_dom, i_part, offsets):
  fields = walldist.get(i_dom, i_part) if isinstance(walldist, PDM.DistCloudSurf) else walldist.get(i_part)

  closest_elt_dist = np.sqrt(fields['ClosestEltDistance'])
  closest_elt_proj = np.copy(fields['ClosestEltProjected'])
  closest_elt_gnum = np.copy(fields['ClosestEltGnum'])

  # Find domain to which the face belongs
  n_face_bnd_tot_idx = np.array(offsets['face_offset'], dtype=closest_elt_gnum.dtype)
  closest_surf_domain = np.searchsorted(n_face_bnd_tot_idx, closest_elt_gnum-1, side='right') -1
  closest_surf_domain = closest_surf_domain.astype(closest_elt_gnum.dtype)
  closest_elt_gnuml = closest_elt_gnum - n_face_bnd_tot_idx[closest_surf_domain]
  dom_id = offsets['dom_id']
  if not (dom_id == np.arange(len(dom_id))).all(): # Optim if not periodic
    closest_surf_domain = dom_id[closest_surf_domain]

  return {'Distance' : closest_elt_dist,
          'ClosestEltProjectedX' : closest_elt_proj[0::3],
          'ClosestEltProjectedY' : closest_elt_proj[1::3],
          'ClosestEltProjectedZ' : closest_elt_proj[2::3],
          'ClosestEltDomId' : closest_surf_domain,
          'ClosestEltGnum' : closest_elt_gnuml}

def _wd_get_defaults(size):
  return {'Distance' : np.full(size, np.inf, dtype=float),
          'ClosestEltProjectedX' : np.full(size, np.inf, dtype=float),
          'ClosestEltProjectedY' : np.full(size, np.inf, dtype=float),
          'ClosestEltProjectedZ' : np.full(size, np.inf, dtype=float),
          'ClosestEltDomId' : np.full(size, -1, dtype=pdm_dtype),
          'ClosestEltGnum' : np.full(size, -1, dtype=pdm_dtype)}

def update_closest_to_parent(surface_tree, points_tree, mpi_comm):
  # Update result (surface id) to refer to the corresponding parent face in volumic tree
  # with a PartToPart (part 1 = surfacic face, part 2 = closest face ids, data = parent)
  # Get part 1 + data (lngn shifted because of domains)
  face_parent_gnum_l = []
  face_ln_to_gn_l = []

  surf_per_doms = get_parts_per_blocks(surface_tree, mpi_comm)
  pts_per_doms = get_parts_per_blocks(points_tree, mpi_comm)

  ini_zone_offset = np.zeros(len(surf_per_doms)+1, pdm_dtype)
  for i,surf_zones in enumerate(surf_per_doms.values()):
    for surf_zone in surf_zones:
      face_parent_gnum_l.append(PT.get_node_from_path(surf_zone, 'DiscreteData/Parent')[1])
      face_ln_to_gn_l.append(MT.globalnumbering_value(surf_zone, 'Cell') + ini_zone_offset[i]) # -> Surface gnum for each partition of the surface, shifted ignoring periodics
    ini_zone_offset[i+1] = ini_zone_offset[i] + MT.Zone.n_cell(surf_zones, mpi_comm)
  
  if ini_zone_offset[-1] == 0:
    # Early exit if ini_zone_offset = 0 (no surface in tree)
    return

  # Get part 2 (use same shift)
  closest_elt_gnum = []

  for part_zones in pts_per_doms.values():
    for part in part_zones:
      closest_dom = PT.find_node_from_path(part, 'ClosestElement/ClosestEltDomId')[1].reshape(-1, order='F')
      gnum = PT.find_node_from_path(part, 'ClosestElement/ClosestEltGnum')[1].reshape(-1, order='F')

      closest_elt_gnum.append(gnum + ini_zone_offset[closest_dom])

  # PartToPart to put back the ClosestEltGnum in volumic numbering (construct it only once)
  closest_parent_face = EP.part_to_part(face_parent_gnum_l, face_ln_to_gn_l, closest_elt_gnum, mpi_comm)
  i_part = 0
  for part_zones in pts_per_doms.values():
    for part_zone in part_zones:
      fs_node = PT.get_child_from_name(part_zone, 'ClosestElement')
      shape = PT.get_child_from_name(fs_node, 'Distance')[1].shape
      PT.update_child(fs_node, "ClosestEltGnum", "DataArray_t", value=closest_parent_face[i_part].reshape(shape, order='F'))
      i_part += 1


PointCloud = Tuple[NDArray, NDArray]
# ------------------------------------------------------------------------
def dist_surf_cloud_compute(surf_part_tree: CGNSPartTree,
                            point_clouds: List[List[PointCloud]],
                            periodicities: Dict[str, List[PT.PeriodicValues]],
                            comm:MPIComm) -> List[List[Dict[str, NDArray]]]:
  """
  The lowest level of surf - point cloud computation
  Inputs are surfacic part tree + raw point clouds
  Output is raw dictionnary
  Periodicities are managed with periodicites dict (use empty dict to disable it)
  """

  # Exit with default values if no surface
  if comm.allreduce(sum(PT.Zone.n_cell(zone) for zone in PT.get_all_Zone_t(surf_part_tree))) == 0:
    return [[_wd_get_defaults(cloud[1].size) for cloud in clouds] for clouds in point_clouds]

  n_part_per_cloud = [len(clouds) for clouds in point_clouds]
  _walldist = PDM.DistCloudSurf(comm, 1, 0, point_clouds=n_part_per_cloud) # n_part_surf set later

  surf_per_doms = get_parts_per_blocks(surf_part_tree, comm)
  _keep_alive, offsets = _wd_setup_surf_mesh(surf_per_doms, _walldist, periodicities, comm)

  for i_dom, clouds in enumerate(point_clouds):
    for i_part, cloud in enumerate(clouds):
      coords, pts_lngn = cloud
      _walldist.cloud_set(i_dom, i_part, pts_lngn.shape[0], coords, pts_lngn)

  _walldist.compute()

  all_results = [[_wd_get(_walldist, i_dom, i_part, offsets) for i_part in range(n_part)] \
                 for i_dom, n_part in enumerate(n_part_per_cloud)]

  del _keep_alive
  return all_results


def find_closest_element(src_part_tree: CGNSPartTree,
                         tgt_part_tree: CGNSPartTree,
                         location: str,
                         comm: MPIComm,
                         **options) -> None:
  """
  Search the closest element in source tree.
  Source tree can be of dimension 1 or 2.
  Return in a container called 'ClosestElement'
  """
  parts_per_dom_pts = get_parts_per_blocks(tgt_part_tree, comm).values()

  all_clouds = [[get_point_cloud(part_zone, location) for part_zone in part_zones] \
                for part_zones in parts_per_dom_pts]

  periodicities = options.get('periodicities', dict())
  results = dist_surf_cloud_compute(src_part_tree, all_clouds, periodicities, comm)

  for dom_results, part_zones in zip(results, parts_per_dom_pts):
    for result, part_zone in zip(dom_results, part_zones):
      # Retrieve location
      fs_node = _create_output_container(part_zone, location, 'ClosestElement')
      shape = _get_output_shape(part_zone, fs_node)

      for key, val in result.items():
        PT.update_child(fs_node, key, 'DataArray_t', val.reshape(shape, order='F'))




def find_closest_boundary(src_part_tree: CGNSPartTree,
                          src_tgt_tree: CGNSPartTree,
                          location: str,
                          comm: MPIComm,
                          surf_predicate: PT.pred.NodePredicate = IS_BND,
                          perio: bool = True) -> None:

  periodicities = detect_perio(src_part_tree, comm) if perio else dict()
  bnd_tree = extract_surf_from_bc(src_part_tree, surf_predicate, comm)
  find_closest_element(bnd_tree,
                       src_tgt_tree,
                       location,
                       comm,
                       periodicities=periodicities)

  update_closest_to_parent(bnd_tree, src_tgt_tree, comm)




def find_closest_boundary_propagation(part_tree: CGNSPartTree,
                                      location: str,
                                      mpi_comm: MPIComm,
                                      surf_predicate: PT.pred.NodePredicate = IS_BND) -> None:

  parts_per_dom = get_parts_per_blocks(part_tree, mpi_comm)
  first_dom = next(iter(parts_per_dom.keys()))
  if len(parts_per_dom) > 1:
    raise NotImplementedError("Wall_distance computation with method 'propagation' does not support multiple domains")
  elif len(parts_per_dom[first_dom]) > 0 and PT.Zone.CellDimension(parts_per_dom[first_dom][0]) != 3:
    raise NotImplementedError("Wall_distance computation with method 'propagation' only supports 3D meshes")
  _walldist = PDM.DistCellCenterSurf(mpi_comm, 1, n_part_vol=1)

  surface_tree = extract_surf_from_bc(part_tree, surf_predicate, mpi_comm)
  
  surf_per_doms = get_parts_per_blocks(surface_tree, mpi_comm)
  _keep_alive, out = _wd_setup_surf_mesh(surf_per_doms, _walldist, dict(), mpi_comm)


  if out['face_offset'][-1] == 0:
    return

  for i_domain, part_zones in enumerate(parts_per_dom.values()):
    _walldist.n_part_vol = len(part_zones)
    if len(part_zones) > 0 and PT.Zone.Type(part_zones[0]) != 'Unstructured':
      raise NotImplementedError("Wall_distance computation with method 'propagation' does not support structured blocks")
    _keep_alive.append(_wd_setup_vol_mesh(part_zones, _walldist))

  #Compute
  _walldist.compute()
  

  for i_domain, part_zones in enumerate(parts_per_dom.values()):
    for i_part, part_zone in enumerate(part_zones):
      fields = _wd_get(_walldist, i_domain, i_part, out)

      # Retrieve location
      fs_node = _create_output_container(part_zone, location, 'ClosestElement')
      shape = _get_output_shape(part_zone, fs_node)

      for key, val in fields.items():
        PT.update_child(fs_node, key, 'DataArray_t', val.reshape(shape, order='F'))



  update_closest_to_parent(surface_tree, part_tree, mpi_comm)


