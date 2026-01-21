import numpy as np
from mpi4py import MPI
import time
import warnings

from maia.typing import *
import maia.pytree        as PT
import maia.pytree.maia   as MT
from maia import npy_pdm_gnum_dtype as pdm_dtype

from maia.utils                      import np_utils
from maia.utils                      import logging as mlog
from maia.transfer                   import protocols as EP
from maia.transfer                   import utils as tr_utils
from maia.factory.dist_from_part     import discover_nodes_from_matching, get_parts_per_blocks
from maia.algo.part.extract_boundary import extract_surf_from_bc
from maia.algo.part.geometry         import _compute_elements_center

from .point_cloud_utils              import get_point_cloud
import Pypdm.Pypdm as PDM

BC_WALLS = ['BCWall', 'BCWallViscous', 'BCWallViscousHeatFlux', 'BCWallViscousIsothermal']

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

def detect_wall_families(tree: CGNSTree, bcwalls: List[str] = BC_WALLS) -> List[str]:
  """
  Return the list of Families having a FamilyBC_t node whose value is in bcwalls list
  """
  IS_WALL_FAM = PT.pred.NodePredicate(lambda n : PT.get_value(PT.find_child_from_label(n, 'FamilyBC_t')) in bcwalls)
  fam_query = PT.pred.label_is('Family_t') & PT.pred.has_child_of_label('FamilyBC_t') & IS_WALL_FAM
  return [PT.get_name(family) for family in PT.iter_children_from_predicates(tree, ['CGNSBase_t', fam_query])]

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

def _detect_perio(part_tree:CGNSPartTree, comm:MPIComm) -> Dict[str, List[PT.PeriodicValues]]:
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
      val = PT.get_value(jn)
      PT.set_value(jn, MT.conv.get_part_prefix(val))

  grouped_zone_paths = PT.Tree.find_connected_zones(skeleton_tree)
  periodicities_per_path = dict()
  for group in grouped_zone_paths:
    fake_tree = PT.new_CGNSTree()
    fake_base = PT.new_CGNSBase(group[0].split('/')[0], parent=fake_tree)
    for zone_path in group:
      PT.add_child(fake_base, PT.get_node_from_path(skeleton_tree, zone_path))
    all_periodicities, _ = PT.Tree.find_periodic_jns(fake_tree)
    group_periodicities = []
    for perio_val in all_periodicities:
      for u_perio in group_periodicities:
        if _are_same_perio_abs(perio_val, u_perio):
          break
      else:
        group_periodicities.append(perio_val)

    periodicities_per_path.update({path: group_periodicities for path in group})

  return periodicities_per_path

def _setup_surf_mesh(surf_parts_per_dom,
                     walldist,
                     periodicities,
                     comm: MPIComm):
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
      pred = PT.pred.label_is('Elements_t') & (lambda n : PT.Element.Dimension(n)==PT.Zone.CellDimension(surf_zone))
      elts = PT.get_children_from_predicate(surf_zone, pred)
      assert len(elts) == 1, "Mutliple elt nodes not managed"
      elt = MT.Element.connectivity(elts[0])
      domain_parts.append(
        {'face_vtx_idx' : elt.displs,
        'face_vtx' : elt.values,
        'face_lngn' : MT.globalnumbering_value(surf_zone, 'Cell'),
        'vtx_coords' : np_utils.interweave_arrays(PT.Zone.coordinates(surf_zone)),
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
  
  """
  # NB :: on peut reconstruire facilement _n_face_bnd_tot_idx, _n_face_orig_bnd_tot_idx
  #####   et _n_vtx pas nécessaire
  ## 
  sizes = []
  sizes2 = []
  for zonepath, parts in surf_parts_per_dom.items():
    domain_nface = MT.Zone.n_cell(parts, comm)
    perios = periodicities.get(zonepath, [])
    sizes.extend([domain_nface] for _ in range(3**len(perios)))
    sizes2.append(domain_nface)
  assert (self._n_face_bnd_tot_idx == np_utils.sizes_to_indices(sizes)).all()
  assert (self._n_face_orig_bnd_tot_idx == np_utils.sizes_to_indices(sizes2)).all()
  """

def _setup_vol_mesh(part_zones: List[CGNSTree],
                    walldist):
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


def _get(walldist,
          i_domain: int, 
          part_zones: List[CGNSTree], 
          loc,
          out_fs_name,
          offsets) -> None:
  """
  Get results after wall distance computation and store it in the FlowSolution
  node of name self.out_fs_name
  """
  for i_part, part_zone in enumerate(part_zones):

    fields = walldist.get(i_domain, i_part) if isinstance(walldist, PDM.DistCloudSurf) else walldist.get(i_part)

    # Retrieve location
    fs_node = _create_output_container(part_zone, loc, out_fs_name)
    shape = _get_output_shape(part_zone, fs_node)

    # Wall distance
    wall_dist = np.sqrt(fields['ClosestEltDistance'])
    PT.update_child(fs_node, 'Distance', 'DataArray_t', value=wall_dist.reshape(shape,order='F'))

    # Closest projected element
    closest_elt_proj = np.copy(fields['ClosestEltProjected'])
    PT.update_child(fs_node, 'ClosestEltProjectedX', 'DataArray_t', closest_elt_proj[0::3].reshape(shape,order='F'))
    PT.update_child(fs_node, 'ClosestEltProjectedY', 'DataArray_t', closest_elt_proj[1::3].reshape(shape,order='F'))
    PT.update_child(fs_node, 'ClosestEltProjectedZ', 'DataArray_t', closest_elt_proj[2::3].reshape(shape,order='F'))

    # Closest gnum element (face)
    closest_elt_gnum = np.copy(fields['ClosestEltGnum'])

    # Find domain to which the face belongs
    n_face_bnd_tot_idx = np.array(offsets['face_offset'], dtype=closest_elt_gnum.dtype)
    closest_surf_domain = np.searchsorted(n_face_bnd_tot_idx, closest_elt_gnum-1, side='right') -1
    closest_surf_domain = closest_surf_domain.astype(closest_elt_gnum.dtype)
    closest_elt_gnuml = closest_elt_gnum - n_face_bnd_tot_idx[closest_surf_domain]
    dom_id = offsets['dom_id']
    if not (dom_id == np.arange(len(dom_id))).all(): # Optim is not perio
      closest_surf_domain = dom_id[closest_surf_domain]
    PT.update_child(fs_node, "ClosestEltDomId", "DataArray_t", value=closest_surf_domain.reshape(shape,order='F'))
    PT.update_child(fs_node, "ClosestEltGnum", "DataArray_t", value=closest_elt_gnuml.reshape(shape,order='F'))


# ------------------------------------------------------------------------
def wd_compute(part_tree: CGNSPartTree, 
               bc_predicate: Any, 
               mpi_comm: MPIComm, 
               *, 
               method: str = "cloud", 
               point_cloud: str = 'CellCenter', 
               out_fs_name: str = 'WallDistance', 
               perio: bool = True) -> None:

    """
    Prepare, compute and get wall distance
    """

    assert method in ["cloud", "propagation"]


    # Group partitions by original dist domain
    parts_per_dom = get_parts_per_blocks(part_tree, mpi_comm)
    assert len(parts_per_dom) >= 1
        
    if perio:
      if method == "cloud":
        periodicities_per_dom = _detect_perio(part_tree, mpi_comm)
      else:
        periodicities_per_dom = dict()
        warnings.warn("WallDistance do not manage periodicities except for 'cloud' method", RuntimeWarning, stacklevel=2)
    else:
      periodicities_per_dom = dict()

        
    # Create walldist structure
    # Multidomain is not managed for n_part_surf, n_part_surf is the total of partitions
    if method == "propagation":
      first_dom = next(iter(parts_per_dom.keys()))
      if len(parts_per_dom) > 1:
        raise NotImplementedError("Wall_distance computation with method 'propagation' does not support multiple domains")
      elif len(parts_per_dom[first_dom]) > 0 and PT.Zone.CellDimension(parts_per_dom[first_dom][0]) != 3:
        raise NotImplementedError("Wall_distance computation with method 'propagation' only supports 3D meshes")
      _walldist = PDM.DistCellCenterSurf(mpi_comm, 1, n_part_vol=1)

    elif method == "cloud":
      n_part_per_cloud = [len(part_zones) for part_zones in parts_per_dom.values()]
      _walldist = PDM.DistCloudSurf(mpi_comm, 1, 0, point_clouds=n_part_per_cloud) # n_part_surf set later


    surface_tree = extract_surf_from_bc(part_tree, bc_predicate, mpi_comm)
    
    surf_per_doms = get_parts_per_blocks(surface_tree, mpi_comm)
    _keep_alive, out = _setup_surf_mesh(surf_per_doms, _walldist, periodicities_per_dom, mpi_comm)


    if out['face_offset'][-1] == 0:
      return -1 # No surface found

    # Prepare mesh depending on method
    if method == "cloud":
      for i_domain, part_zones in enumerate(parts_per_dom.values()):
        for i_part, part_zone in enumerate(part_zones):
          points, points_lngn = get_point_cloud(part_zone, point_cloud)
          _keep_alive.extend([points, points_lngn])
          _walldist.cloud_set(i_domain, i_part, points_lngn.shape[0], points, points_lngn)

    elif method == "propagation":
      for i_domain, part_zones in enumerate(parts_per_dom.values()):
        _walldist.n_part_vol = len(part_zones)
        if len(part_zones) > 0 and PT.Zone.Type(part_zones[0]) != 'Unstructured':
          raise NotImplementedError("Wall_distance computation with method 'propagation' does not support structured blocks")
        _keep_alive.append(_setup_vol_mesh(part_zones, _walldist))

    #Compute
    _walldist.compute()


    for i_domain, (dist_zone_path, part_zones) in enumerate(parts_per_dom.items()):
      _get(_walldist, i_domain, part_zones, point_cloud, out_fs_name, out)



    # Update result (surface id) to refer to the corresponding parent face in volumic tree
    # with a PartToPart (part 1 = surfacic face, part 2 = closest face ids, data = parent)
    # Get part 1 + data (lngn shifted because of domains)
    face_parent_gnum_l = []
    face_ln_to_gn_l = []

    ini_zone_offset = np.zeros(len(surf_per_doms)+1, pdm_dtype)
    for i,surf_zones in enumerate(surf_per_doms.values()):
      for surf_zone in surf_zones:
        face_parent_gnum_l.append(PT.get_node_from_name(surf_zone, 'ParentFace')[1])
        face_ln_to_gn_l.append(MT.globalnumbering_value(surf_zone, 'Cell') + ini_zone_offset[i]) # -> Surface gnum for each partition of the surface, shifted ignoring periodics
      ini_zone_offset[i+1] = ini_zone_offset[i] + MT.Zone.n_cell(surf_zones, mpi_comm)
    # Get part 2 (use same shift)
    closest_elt_gnum = []
    for part_zones in parts_per_dom.values():
      for part in part_zones:
        closest_dom = PT.find_node_from_path(part, out_fs_name+'/ClosestEltDomId')[1].reshape(-1, order='F')
        gnum = PT.find_node_from_path(part, out_fs_name+'/ClosestEltGnum')[1].reshape(-1, order='F')

        closest_elt_gnum.append(gnum + ini_zone_offset[closest_dom])

    # PartToPart to put back the ClosestEltGnum in volumic numbering (construct it only once)
    closest_parent_face = EP.part_to_part(face_parent_gnum_l, face_ln_to_gn_l, closest_elt_gnum, mpi_comm)
    i_part = 0
    for part_zones in parts_per_dom.values():
      for part_zone in part_zones:
        fs_node = PT.get_child_from_name(part_zone, out_fs_name)
        shape = PT.get_child_from_name(fs_node, 'Distance')[1].shape
        PT.update_child(fs_node, "ClosestEltGnum", "DataArray_t", value=closest_parent_face[i_part].reshape(shape, order='F'))
        i_part += 1


    # Free unnecessary numpy
    del _keep_alive


# ------------------------------------------------------------------------
def compute_projection_to(part_tree, bc_predicate, comm, point_cloud='CellCenter', out_fs_name='SurfDistance', **options):

  start = time.time()
  
  out = wd_compute(part_tree, bc_predicate, comm, point_cloud=point_cloud, out_fs_name=out_fs_name, **options)
  end = time.time()
  if out == -1:
    mlog.error(f"Projection computing failed because no BC_t matches the given predicate")
  else:
    mlog.info(f"Projection computed ({end-start:.2f} s)")

def compute_wall_distance(part_tree: CGNSPartTree,
                          comm: MPIComm,
                          point_cloud: str = 'CellCenter',
                          out_fs_name: str = 'WallDistance',
                          **options: Any) -> None:
  """Compute wall distances and add it in tree.

  For each volumic point, compute the distance to the nearest face belonging to a BC of kind wall.
  BC are considered to be of kind wall if their BCType (or the one of their related family) is one of 
  ``'BCWall'``, ``'BCWallViscous'``, ``'BCWallViscousHeatFlux'`` or ``'BCWallViscousIsothermal'``.

  Note: 
    Propagation method requires ParaDiGMa access and is only available for unstructured cell centered
    NGon connectivities grids. In addition, partitions must have been created from a single initial domain
    with this method.

  Tree is modified inplace: computed distance are added in a DiscreteData container whose
  name can be specified with out_fs_name parameter.

  The following optional parameters can be used to control the underlying method:

    - ``method`` ({'cloud', 'propagation'}): Choice of the geometric method. Defaults to ``'cloud'``.
    - ``perio`` (bool): Take into account periodic connectivities. Defaults to ``True``.
      Only available when method=cloud.

  Args:
    part_tree (CGNSPartTree)   : Input partitioned tree
    comm       (MPIComm)       : MPI communicator
    point_cloud (str, optional): Points to project on the surface. Can either be one of
      "CellCenter" or "Vertex" (coordinates are retrieved from the mesh) or the name of a FlowSolution
      node in which coordinates are stored. Defaults to CellCenter.
    out_fs_name (str, optional): Name of the output DiscreteData_t node storing wall distance data.
    **options: Additional options related to geometric method (see above)

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #compute_wall_distance@start
        :end-before: #compute_wall_distance@end
        :dedent: 2
  """
  MT.check_cgns_part_tree(part_tree)
  start = time.time()
  
  # Retrieve Wall Families (warning -- if we have a Family_t appearing under two bases 
  # with the same name, it can be wrongly selected)
  wall_bc_families = detect_wall_families(part_tree)
  is_wall_bc = PT.pred.value_in(BC_WALLS) | PT.pred.any([PT.pred.belongs_to_family(family) for family in wall_bc_families])

  out = wd_compute(part_tree, is_wall_bc, comm, point_cloud=point_cloud, out_fs_name=out_fs_name, **options)
  end = time.time()
  if out == -1:
    mlog.warning(f"Wall distance computing skipped because no wall-like BC_t have been found in tree." \
                  " Default values used for output arrays.")

    for part_zone in PT.get_all_Zone_t(part_tree):
      fs_node = _create_output_container(part_zone, point_cloud, out_fs_name)
      shape = _get_output_shape(part_zone, fs_node)

      PT.update_child(fs_node, "ClosestEltGnum",       "DataArray_t", np.full(shape, -1, dtype=pdm_dtype, order='F'))
      PT.update_child(fs_node, "ClosestEltDomId",      "DataArray_t", np.full(shape, -1, dtype=pdm_dtype, order='F'))
      PT.update_child(fs_node, 'TurbulentDistance',    "DataArray_t", np.full(shape, np.inf, dtype=float, order='F'))
      PT.update_child(fs_node, 'ClosestEltProjectedX', "DataArray_t", np.full(shape, np.inf, dtype=float, order='F'))
      PT.update_child(fs_node, 'ClosestEltProjectedY', "DataArray_t", np.full(shape, np.inf, dtype=float, order='F'))
      PT.update_child(fs_node, 'ClosestEltProjectedZ', "DataArray_t", np.full(shape, np.inf, dtype=float, order='F'))
      
  else:
    mlog.info(f"Wall distance computed ({end-start:.2f} s)")
    for zone in PT.iter_all_Zone_t(part_tree): #Rename Distance -> TurbulentDistance
      container = PT.find_child_from_name(zone, out_fs_name)
      PT.rm_children_from_name(container, 'TurbulenceDistance') # Cleanup
      node = PT.find_child_from_name(container, "Distance")
      PT.set_name(node, 'TurbulentDistance')

