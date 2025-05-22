from mpi4py import MPI
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia          import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.utils    import py_utils, np_utils, par_utils

from maia.transfer import protocols as EP

from maia.algo.indexing                import pe_to_nface, ngon_to_edge_pe, get_pe_local
from maia.algo.dist.ngon_tools         import PDM_dfacecell_to_dcellface
from maia.algo.dist.connectivity_utils import entity_vtx_connectivity_elt, cell_vtx_connectivity_S
from maia.algo.dist.point_cloud_utils  import get_point_cloud

from maia.algo.part.localize import _mdom_mesh_location as _mdom_mesh_location_part

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

  vtx_distri  = MT.distribution_value(zone, 'Vertex')
  cell_distri = MT.distribution_value(zone, 'Cell')

  dcoords = PT.Zone.coordinates(zone)

  if use_geom:
    raise NotImplementedError
  else:
    cell_gnum = np.arange(cell_distri[0]+1, cell_distri[1]+1, dtype=pdm_gnum_dtype)

  if is_poly_3d_zone(zone):
    pe_to_nface(zone, comm) # Create NFace if not already existing
    ngon  = PT.Zone.NGonNode(zone)
    nface = PT.Zone.NFaceNode(zone)

    dface_vtx  = MT.Element.connectivity(ngon)
    dcell_face = MT.Element.connectivity(nface)
  
    # Compute part. like cell_face
    nface_distri = MT.distribution_value(nface, 'Element')
    if not use_geom and comm.allreduce(np.array_equal(cell_distri, nface_distri), MPI.LAND):
      pcell_face = dcell_face
    else:
      pcell_face = EP.block_to_part(dcell_face, nface_distri, cell_gnum-1, comm)
    face_gnum, inverse = np.unique(abs(pcell_face.values), return_inverse=True)
    face_gnum = face_gnum.astype(cell_gnum.dtype, copy=False)
    pcell_face_idx = pcell_face.displs.astype(np.int32, copy=False)
    pcell_face     = np.sign(pcell_face.values, dtype=np.int32) * np.arange(1, len(face_gnum)+1, dtype=np.int32)[inverse]
    
    # Compute part. like face_vtx
    ngon_distri = MT.distribution_value(ngon, 'Element')
    _pface_vtx = EP.block_to_part(dface_vtx, ngon_distri, face_gnum-1, comm)
    vtx_gnum, inverse = np.unique(_pface_vtx.values, return_inverse=True) # Unique preserve dtype
    vtx_gnum = vtx_gnum.astype(cell_gnum.dtype, copy=False)
    pface_vtx_idx = np_utils.sizes_to_indices(_pface_vtx.counts, dtype=np.int32)
    pface_vtx     = np.arange(1, len(vtx_gnum)+1, dtype=np.int32)[inverse]

    part_data = [pcell_face_idx, pcell_face, pface_vtx_idx, pface_vtx, \
        cell_gnum, face_gnum, vtx_gnum]
  

  elif is_poly_2d_zone(zone):

    edge  = MT.Zone.EdgeNode(zone)
    if PT.get_child_from_name(edge, 'ParentElements') is None:
      ngon_to_edge_pe(zone, comm)

    # Prepare dface_edge on distributed input
    local_pe = get_pe_local(edge).reshape(-1, order='C')

    _edge_distri = par_utils.partial_to_full_distribution(MT.distribution_value(edge, 'Element'), comm)
    _face_distri = par_utils.partial_to_full_distribution(cell_distri, comm)
    dface_edge = PDM_dfacecell_to_dcellface(comm, _edge_distri, _face_distri, local_pe)

    dedge_vtx = PT.get_child_from_name(edge, 'ElementConnectivity')[1]

    # Compute part. like face_edge
    if not use_geom:
      pface_edge = dface_edge
    else:
      pface_edge = EP.block_to_part(dface_edge, _face_distri, cell_gnum-1, comm)
    edge_gnum, inverse = np.unique(abs(pface_edge.values), return_inverse=True)
    pface_edge_idx = pface_edge.displs.astype(np.int32, copy=False)
    pface_edge     = np.sign(pface_edge.values, dtype=np.int32) * np.arange(1, len(edge_gnum)+1, dtype=np.int32)[inverse]

    # Compute part. like edge_vtx
    GI = EP.GlobalIndexer(_edge_distri, edge_gnum, comm, gnum_offset=1)
    pedge_vtx = GI.Take(dedge_vtx, count=2)
    vtx_gnum, inverse = np.unique(pedge_vtx, return_inverse=True) # Unique preserve dtype
    vtx_gnum = vtx_gnum.astype(cell_gnum.dtype, copy=False)
    pedge_vtx     = np.arange(1, len(vtx_gnum)+1, dtype=np.int32)[inverse]

    part_data = [pface_edge_idx, pface_edge, pedge_vtx, cell_gnum, vtx_gnum]

  else:
    # For elt and S zones: cell_vtx ('volumic' cells only), coords, cell_gnum, vtx_gnum

    if PT.Zone.Type(zone) == 'Structured':
      dcell_vtx = cell_vtx_connectivity_S(zone, dim)
    else:
      dcell_vtx = entity_vtx_connectivity_elt(zone, comm, dim, True)

    # Compute part. like cell_vtx
    if not use_geom:
      pcell_vtx = dcell_vtx
    else:
      pcell_vtx = EP.block_to_part(dcell_vtx, cell_distri, cell_gnum-1, comm)
    vtx_gnum, inverse = np.unique(pcell_vtx.values, return_inverse=True)
    vtx_gnum = vtx_gnum.astype(cell_gnum.dtype, copy=False)
    pcell_vtx_idx  = pcell_vtx.displs.astype(np.int32, copy=False)
    pcell_vtx      = np.arange(1, len(vtx_gnum)+1, dtype=np.int32)[inverse]


    part_data = [pcell_vtx_idx, pcell_vtx, cell_gnum, vtx_gnum]

  # Bring back coordinates
  pcoords = EP.block_to_part(dcoords._asdict(), vtx_distri, vtx_gnum-1, comm)
  pvtx_coords = np_utils.interweave_arrays(list(pcoords.values()))
  if is_poly_3d_zone(zone):
    part_data.insert(4, pvtx_coords)
  elif is_poly_2d_zone(zone):
    part_data.insert(3, pvtx_coords)
  else:
    part_data.insert(2, pvtx_coords)
  return part_data



def _mdom_mesh_location(src_parts, tgt_clouds, comm, reverse=False, loc_tolerance=1E-6):

  # Add a level in list to mimic partitions
  tgt_clouds_per_dom = [[c] for c in tgt_clouds]
  src_parts_per_dom  = [[p] for p in src_parts]
  
  result = _mdom_mesh_location_part(src_parts_per_dom, tgt_clouds_per_dom, comm, reverse, loc_tolerance)

  # Remove intermediate level
  if reverse:
    return py_utils.to_flat_list(result[0]), py_utils.to_flat_list(result[1])
  else:
    return py_utils.to_flat_list(result)
    
def _collect_source(src_doms, comm):
  # > Register source
  connectivity_t = None
  src_parts = []

  for zone in src_doms:

    dim = PT.Zone.CellDimension(zone)
    if PT.Zone.has_ngon_elements(zone):
      if connectivity_t=='Element':
        raise NotImplementedError("Source mesh must have NGon or Element connectivity but not both.")
      connectivity_t = 'Poly'
      selector = [0,1,5,2,3,6,4,7] if dim == 3 else [0,1,4,2,3,5]
    else:
      if connectivity_t == 'Poly':
        raise NotImplementedError("Source mesh must have NGon or Element connectivity but not both.")
      connectivity_t = 'Element'
      selector = [0,1,3,2,4]

    _part_data = minimal_partitioning(zone, comm, use_geom=False) # mesh_loc apply hilbert itself
    part_data = [_part_data[i] for i in selector] # Put in order expected by mesh_location
    src_parts.append((dim, connectivity_t, part_data))

  return src_parts

def _collect_target(tgt_doms, location, comm):
  """ Wraps get_point_cloud around multiple domains """
  return [get_point_cloud(zone, comm, location) for zone in tgt_doms]

def _localize_points(src_dom, tgt_dom, location, comm, \
    reverse=False, loc_tolerance=1E-6):
  """ Intermediate API who do not place output in tree.
  Inputs are list of size n_domain_src (resp. tgt) containing distributed zones (resp. clouds)
  """
  src_parts  = _collect_source(src_dom, comm)
  tgt_clouds = _collect_target(tgt_dom, location, comm)

  return _mdom_mesh_location(src_parts, tgt_clouds, comm, reverse, loc_tolerance)



def localize_points(src_tree, tgt_tree, location, comm, **options):
  """
  Distributed implementation of maia.algo.localize_points
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

