from mpi4py import MPI
import numpy as np

import Pypdm.Pypdm as PDM

import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia
from maia                        import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.utils                  import py_utils, np_utils, par_utils
from maia.transfer               import utils as te_utils
from maia.factory.dist_from_part import get_parts_per_blocks

from maia.transfer import protocols as EP

#from .point_cloud_utils  import get_shifted_point_clouds
#from .connectivity_utils import cell_vtx_connectivity_elts

from maia.algo.geometry  import _compute_elements_center

is_poly_3d_zone = lambda z: PT.Zone.CellDimension(z) == 3 and PT.Zone.has_ngon_elements(z)
is_poly_2d_zone = lambda z: PT.Zone.CellDimension(z) == 2 and \
                            PT.Zone.Type(z) == 'Unstructured' and \
                            all(PT.Element.CGNSName(e) in ['BAR_2', 'NGON_n'] for e in PT.get_children_from_label(z, 'Elements_t'))

def minimal_partitioning(zone, comm, use_geom=False):
  """
  Minimal partitioning without zones interfaces, groups, etc.
  Return direct arrays :
  For 3D poly zones : cell_face, face_vtx, coords, cell_gnum, face_gnum, vtx_gnum, 
  For 2D poly zones : face_edge, edge_vtx, coords, face_gnum, vtx_gnum
  For elt or S zones : cell_vtx ('volumic' cells only), coords, cell_gnum, vtx_gnum
  """
  dim = PT.Zone.CellDimension(zone)

  vtx_distri  = MT.getDistribution(zone, 'Vertex')[1]
  cell_distri = MT.getDistribution(zone, 'Cell')[1]

  dcoords = PT.Zone.coordinates(zone)

  if use_geom:
    raise NotImplementedError
  else:
    cell_gnum = np.arange(cell_distri[0]+1, cell_distri[1]+1, dtype=pdm_gnum_dtype)

  if is_poly_3d_zone(zone):
    maia.algo.pe_to_nface(zone, comm) # Create NFace if not already existing
    ngon  = PT.Zone.NGonNode(zone)
    nface = PT.Zone.NFaceNode(zone)

    dface_vtx_idx = PT.get_child_from_name(ngon, "ElementStartOffset")[1]
    dface_vtx     = PT.get_child_from_name(ngon, "ElementConnectivity")[1]
    dface_vtx_n    = np.diff(dface_vtx_idx)
    dcell_face_idx = PT.get_child_from_name(nface, "ElementStartOffset")[1]
    dcell_face     = PT.get_child_from_name(nface, "ElementConnectivity")[1]
    dcell_face_n   = np.diff(dcell_face_idx)
  
    # Compute part. like cell_face
    nface_distri = MT.get_distribution(nface, 'Element')[1]
    if not use_geom and comm.allreduce(np.array_equal(cell_distri, nface_distri), MPI.LAND):
      pcell_face_n, pcell_face = dcell_face_n, dcell_face
    else:
      pcell_face_n, pcell_face = EP.block_to_part_strided(dcell_face_n, dcell_face, nface_distri, cell_gnum-1, comm, legacy=False)
    face_gnum, inverse = np.unique(abs(pcell_face), return_inverse=True)
    pcell_face_idx = np_utils.sizes_to_indices(pcell_face_n, dtype=np.int32)
    pcell_face     = np.sign(pcell_face, dtype=np.int32) * np.arange(1, len(face_gnum)+1, dtype=np.int32)[inverse]
    
    # Compute part. like face_vtx
    ngon_distri = MT.get_distribution(ngon, 'Element')[1]
    pface_vtx_n, pface_vtx = EP.block_to_part_strided(dface_vtx_n, dface_vtx, ngon_distri, face_gnum-1, comm, legacy=False) 
    vtx_gnum, inverse = np.unique(pface_vtx, return_inverse=True) # Unique preserve dtype
    pface_vtx_idx = np_utils.sizes_to_indices(pface_vtx_n, dtype=np.int32)
    pface_vtx     = np.arange(1, len(vtx_gnum)+1, dtype=np.int32)[inverse]

    part_data = [pcell_face_idx, pcell_face, pface_vtx_idx, pface_vtx, \
        cell_gnum, face_gnum, vtx_gnum]
  

  elif is_poly_2d_zone(zone):

    edge  = MT.Zone.EdgeNode(zone)
    if PT.get_child_from_name(edge, 'ParentElements') is None:
      maia.algo.ngon_to_edge_pe(zone, comm)

    # Prepare dface_edge on distributed input
    from maia.algo.dist.ngon_tools import PDM_dfacecell_to_dcellface
    local_pe = maia.algo.indexing.get_pe_local(edge).reshape(-1, order='C')

    _edge_distri = par_utils.partial_to_full_distribution(MT.get_distribution(edge, 'Element')[1], comm)
    _face_distri = par_utils.partial_to_full_distribution(cell_distri, comm)
    dface_edge_idx, dface_edge = PDM_dfacecell_to_dcellface(comm, _edge_distri, _face_distri, local_pe)
    dface_edge_n = np.diff(dface_edge_idx)

    dedge_vtx = PT.get_child_from_name(edge, 'ElementConnectivity')[1]

    # Compute part. like face_edge
    if not use_geom:
      pface_edge_n, pface_edge = dface_edge_n, dface_edge
    else:
      pface_edge_n, pface_edge = EP.block_to_part_strided(dface_edge_n, dface_edge, _face_distri, cell_gnum-1, comm, legacy=False)
    edge_gnum, inverse = np.unique(abs(pface_edge), return_inverse=True)
    pface_edge_idx = np_utils.sizes_to_indices(pface_edge_n, dtype=np.int32)
    pface_edge     = np.sign(pface_edge, dtype=np.int32) * np.arange(1, len(edge_gnum)+1, dtype=np.int32)[inverse]

    # Compute part. like edge_vtx
    GI = EP.GlobalIndexer(_edge_distri, edge_gnum-1, comm)
    pedge_vtx = GI.Take(dedge_vtx, count=2)
    vtx_gnum, inverse = np.unique(pedge_vtx, return_inverse=True) # Unique preserve dtype
    pedge_vtx     = np.arange(1, len(vtx_gnum)+1, dtype=np.int32)[inverse]

    part_data = [pface_edge_idx, pface_edge, pedge_vtx, cell_gnum, vtx_gnum]

  else:
    # For elt and S zones: cell_vtx ('volumic' cells only), coords, cell_gnum, vtx_gnum

    from maia.algo.dist.connectivity_utils import entity_vtx_connectivity_elt, cell_vtx_connectivity_S
    if PT.Zone.Type(zone) == 'Structured':
      dcell_vtx_idx, dcell_vtx = cell_vtx_connectivity_S(zone, dim)
    else:
      dcell_vtx_idx, dcell_vtx = entity_vtx_connectivity_elt(zone, comm, dim, True)
    dcell_vtx_n = np.diff(dcell_vtx_idx)

    # Compute part. like cell_vtx
    if not use_geom:
      pcell_vtx_n, pcell_vtx = dcell_vtx_n, dcell_vtx
    else:
      pcell_vtx_n, pcell_vtx = EP.block_to_part_strided(dcell_vtx_n, dcell_vtx, cell_distri, cell_gnum-1, comm, legacy=False)
    vtx_gnum, inverse = np.unique(pcell_vtx, return_inverse=True)
    pcell_vtx_idx  = np_utils.sizes_to_indices(pcell_vtx_n, dtype=np.int32)
    pcell_vtx      = np.arange(1, len(vtx_gnum)+1, dtype=np.int32)[inverse]


    part_data = [pcell_vtx_idx, pcell_vtx, cell_gnum, vtx_gnum]

  # Bring back coordinates
  pcoords = EP.block_to_part(dcoords._asdict(), vtx_distri, vtx_gnum-1, comm, legacy=False)
  pvtx_coords = np_utils.interweave_arrays(list(pcoords.values()))
  if is_poly_3d_zone(zone):
    part_data.insert(4, pvtx_coords)
  elif is_poly_2d_zone(zone):
    part_data.insert(3, pvtx_coords)
  else:
    part_data.insert(2, pvtx_coords)
  return part_data



   




# TODO : factorize ?
def get_point_cloud(zone, comm, location='CellCenter'):
  """
  If location == Vertex, return the (interlaced) coordinates of vertices 
  and vertex global numbering of a partitioned zone
  If location == Center, compute and return the (interlaced) coordinates of
  cell centers and cell global numbering of a partitioned zone
  """
  vtx_distri   = MT.get_distribution(zone, 'Vertex')[1]
  cell_distri  = MT.get_distribution(zone, 'Cell')[1]

  if location == 'Vertex':
    vtx_ln_to_gn = np.arange(vtx_distri[0], vtx_distri[1], dtype=pdm_gnum_dtype) + 1
    coords = [c.reshape(-1, order='F') for c in PT.Zone.coordinates(zone)]
    vtx_coords   = np_utils.interweave_arrays(coords)
    return vtx_coords, vtx_ln_to_gn

  elif location == 'CellCenter':
    cell_distri   = MT.get_distribution(zone, 'Cell')[1]
    cell_ln_to_gn = np.arange(cell_distri[0], cell_distri[1], dtype=pdm_gnum_dtype) + 1
    center_cell = _compute_elements_center(zone, 'CellCenter', comm)
    return center_cell, cell_ln_to_gn
  
  else: #Try to catch a container with the given name
    container = PT.get_child_from_name(zone, location)
    if container:
      assert PT.get_child_from_name(container, 'PointList') is None
      assert PT.get_child_from_name(container, 'PointRange') is None
      coords = [PT.get_value(c).reshape(-1, order='F') for c in PT.get_children_from_name(container, 'Coordinate*')]
      int_coords = np_utils.interweave_arrays(coords)
      if PT.Subset.GridLocation(container) == 'Vertex':
        ln_to_gn = np.arange(vtx_distri[0], vtx_distri[1], dtype=pdm_gnum_dtype) + 1
      elif PT.Subset.GridLocation(container) == 'CellCenter':
        ln_to_gn = np.arange(cell_distri[0], cell_distri[1], dtype=pdm_gnum_dtype) + 1
      return int_coords, ln_to_gn

  raise RuntimeError("Unknow location or node")

# TODO : factorize ? 
def get_shifted_point_clouds(dom_list, location, comm):
  """ Wraps get_point_cloud around multiple domains,
  shifting lngn with previous values"""
  coords_per_dom = []
  lngn_per_dom = []
  clouds = []
  offset = np.zeros(len(dom_list)+1, dtype=pdm_gnum_dtype)
  for idom, zone in enumerate(dom_list):
    coords, lngn = get_point_cloud(zone, comm, location)
    coords_per_dom.append(coords)
    lngn_per_dom.append(lngn_per_dom)
    max = par_utils.arrays_max([lngn], comm)
    offset[idom+1] = offset[idom] + max
    lngn += offset[idom]

    clouds.append((coords, lngn))

  return offset, clouds





def _get_part_data_ngon(zone, comm):
  dim = PT.Zone.CellDimension(zone)
  
  pdata = minimal_partitioning(zone, comm)

  if dim == 3:
    
    pcell_face_idx, pcell_face, pface_vtx_idx, pface_vtx, pcoords, \
        cell_gnum, face_gnum, vtx_gnum = pdata

    return [pcell_face_idx, pcell_face, cell_gnum, \
        pface_vtx_idx, pface_vtx, face_gnum, pcoords, vtx_gnum]

  elif dim == 2:
    
    pface_edge_idx, pface_edge, pedge_vtx, pcoords, \
        cell_gnum, vtx_gnum = pdata
  
    return [pface_edge_idx, pface_edge, cell_gnum, pedge_vtx, pcoords, vtx_gnum]


def _get_part_data_elts(zone, comm):
  pcell_vtx_idx, pcell_vtx, pvtx_coords, cell_gnum, vtx_gnum = minimal_partitioning(zone, comm)

  return [pcell_vtx_idx, pcell_vtx, cell_gnum, pvtx_coords, vtx_gnum]
    

def _mesh_location(src_parts, tgt_clouds, comm, reverse=False, loc_tolerance=1E-6):
  """ Wrapper of PDM mesh location
  For now, only 1 domain is supported so we expect source parts and target clouds
  as flat lists :
  Parts are tuple dim, elt_kind, part_data 
   where dim = 2 or 3, elt_kind = 'Poly' or 'Element' and part_data stores the arrays
   expected by paradigm
  Cloud are tuple (coords, lngn)
  """

  n_part_src = len(src_parts)
  n_part_tgt = len(tgt_clouds)
  # > Create and setup global data
  mesh_loc = PDM.MeshLocation(n_point_cloud=1, comm=comm)
  mesh_loc.mesh_n_part_set(n_part_src)
  mesh_loc.n_part_cloud_set(0, n_part_tgt)

  # > Register source
  for i_part, part_data in enumerate(src_parts):
    dim, kind, _part_data = part_data
    assert dim >= 2, "Dimension lower than 2 are not supported"
    set_func = {'Poly' :    [mesh_loc.part_set_2d, mesh_loc.part_set],
                'Element' : [mesh_loc.nodal_part_set_2d, mesh_loc.nodal_part_set]}[kind][dim-2]
    set_func(i_part, *_part_data)


  # > Setup target
  for i_part, (coords, lngn) in enumerate(tgt_clouds):
    mesh_loc.cloud_set(0, i_part, coords, lngn)

  mesh_loc.tolerance = loc_tolerance
  mesh_loc.compute()

  # This is located and unlocated indices
  all_located_id   = [mesh_loc.located_get  (0,i_part) for i_part in range(n_part_tgt)]
  all_unlocated_id = [mesh_loc.unlocated_get(0,i_part) for i_part in range(n_part_tgt)]

  #This is result from the target perspective (api : (i_pt_cloud, i_part))
  all_target_data = [mesh_loc.location_get(0, i_tgt_part) for i_tgt_part in range(n_part_tgt)]
  # Add ids in dict
  for i_part, data in enumerate(all_target_data):
    data.pop('g_num')
    data['located_ids']   = all_located_id[i_part] - 1
    data['unlocated_ids'] = all_unlocated_id[i_part] - 1

  #This is result from the source perspective (api : ((i_pt_cloud, i_part))
  if reverse:
    all_located_inv = [{**mesh_loc.points_in_elt_get(0, i_src_part), **mesh_loc.cell_vertex_get(i_src_part)} \
                       for i_src_part in range(n_part_src)]
    return all_target_data, all_located_inv
  else:
    return all_target_data


def _localize_points(src_dom, tgt_dom, location, comm, \
    reverse=False, loc_tolerance=1E-6):
  """
  """
  locs = {'NGon2D' :{'Cell':2, 'Vtx':5},
          'NGon3D' :{'Cell':2, 'Face':5, 'Vtx':7},
          'Element':{'Cell':2, 'Vtx':4}}
  n_dom_src = len(src_dom)


  # > Register source
  connectivity_t = None
  src_parts = []
  for i_domain, dom in enumerate(src_dom):

    dim = PT.Zone.CellDimension(dom)
    if PT.Zone.has_ngon_elements(dom):
      if connectivity_t=='Element':
        raise NotImplementedError("Source mesh must have NGon or Element connectivity but not both.")
      connectivity_t = f'NGon{dim}D'
      src_parts.append((dim, 'Poly', _get_part_data_ngon(dom, comm)))
    else:
      if connectivity_t is not None and 'NGON' in connectivity_t:
        raise NotImplementedError("Source mesh must have NGon or Element connectivity but not both.")
      connectivity_t = 'Element'
      src_parts.append((dim, 'Element', _get_part_data_elts(dom, comm)))

  locs = locs[connectivity_t]
  src_offsets = {loc : np.zeros(n_dom_src+1, dtype=pdm_gnum_dtype) for loc in locs}
  for i_domain, src_part in enumerate(src_parts):
    # Compute global offsets for this domain
    for loc, array_idx in locs.items():
      dom_max = par_utils.arrays_max([src_part[2][array_idx]], comm)
      src_offsets[loc][i_domain+1] = src_offsets[loc][i_domain] + dom_max
    # Shift source arrays (inplace)
    for loc, array_idx in locs.items():
      src_part[2][array_idx] += src_offsets[loc][i_domain]


  tgt_offset, tgt_clouds = get_shifted_point_clouds(tgt_dom, location, comm)
  

  result = _mesh_location(src_parts, tgt_clouds, comm, reverse, loc_tolerance)

  # No need to shift back source data, because it was only a copy

  # Shift results and get domain ids
  direct_result = result[0] if reverse else result
  for tgt_result in direct_result:
    tgt_result['location_shifted'] = tgt_result.pop('location') #Rename key
    tgt_result['location'], tgt_result['domain'] = np_utils.shifted_to_local(
        tgt_result['location_shifted'], src_offsets['Cell'])
  if reverse:
    for src_result in result[1]:
      src_result['points_gnum_shifted'] = src_result.pop('points_gnum') #Rename key
      src_result['points_gnum'], src_result['domain'] = np_utils.shifted_to_local(
          src_result['points_gnum_shifted'], tgt_offset)

  # Reshape output to list of lists (as input domains)
  return result

def localize_points(src_tree, tgt_tree, location, comm, **options):
  """Localize points between two distributed trees.

  For all the points of the target tree matching the given location,
  search the cell of the source tree in which it is enclosed.
  The result, i.e. the gnum & domain number of the source cell (or -1 if the point is not localized),
  are stored in a ``DiscreteData_t`` container called "Localization" on the target zones.

  Source tree must be unstructured.

  Localization can be parametred thought the options kwargs:

  - ``loc_tolerance`` (default = 1E-6) -- Geometric tolerance for the method.

  Args:
    src_tree (CGNSTree): Source tree, partitionned. Only unstructured connectivities are managed.
    tgt_tree (CGNSTree): Target tree, partitionned.
    location ({'CellCenter', 'Vertex'}) : Target points to localize
    comm       (MPIComm): MPI communicator
    **options: Additional options related to location strategy

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #localize_points@start
        :end-before: #localize_points@end
        :dedent: 2
  """
  src_path = PT.predicates_to_paths(src_tree, 'CGNSBase_t/Zone_t')
  src_dom = [PT.get_node_from_path(src_tree, path) for path in src_path]
  tgt_dom = PT.get_children_from_predicates(tgt_tree, 'CGNSBase_t/Zone_t')

  located_data = _localize_points(src_dom, tgt_dom, location, comm, **options)

  dom_list = '\n'.join(src_path)
  
  for i_dom, tgt_part in enumerate(tgt_dom):
    sol = PT.update_child(tgt_part, "Localization", "DiscreteData_t")
    PT.new_GridLocation(location, sol)
    data = located_data[i_dom]
    n_tgts = data['located_ids'].size + data['unlocated_ids'].size,
    src_gnum = -np.ones(n_tgts, dtype=pdm_gnum_dtype) #Init with -1 to carry unlocated points
    src_dom  = -np.ones(n_tgts, dtype=np.int32)
    src_gnum[data['located_ids']] = data['location']
    src_dom [data['located_ids']] = data['domain']
    PT.new_DataArray("SrcId", src_gnum, parent=sol)
    PT.new_DataArray("DomId", src_dom,  parent=sol)
    PT.new_Descriptor("DomainList", dom_list, parent=sol)

