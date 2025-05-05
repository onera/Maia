import numpy as np
from mpi4py import MPI
from typing import overload

from maia.typing import *
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia                        import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.utils                  import py_utils, np_utils, par_utils
from maia.transfer               import utils as te_utils
from maia.utils                  import vstride as vs
from maia.factory.dist_from_part import get_parts_per_blocks

from .point_cloud_utils  import get_point_cloud
from .connectivity_utils import cell_vtx_connectivity, PDM_connectivity_transpose
import Pypdm.Pypdm as PDM

PointCloud = Tuple[NDArray, NDArray]
PartData = Tuple[int, str, List[NDArray]]
Result = Dict[str, NDArray]
InvResult = Dict[str, vs.VStrideArray]

def _get_part_data_ngon(part_zone: CGNSTree) -> List[NDArray]:
  dim = PT.Zone.CellDimension(part_zone)
  cx, cy, cz = PT.Zone.coordinates(part_zone)
  assert (cx is not None) and  (cy is not None) and (cz is not None)
  vtx_coords = np_utils.interweave_arrays([cx,cy,cz])

  vtx_ln_to_gn  = PT.get_np_value(MT.requestGlobalNumbering(part_zone, 'Vertex'))
  cell_ln_to_gn = PT.get_np_value(MT.requestGlobalNumbering(part_zone, 'Cell'))

  if dim == 3:
    ngon  = PT.Zone.NGonNode(part_zone)
    nface = PT.Zone.NFaceNode(part_zone)

    face_vtx  = MT.Element.connectivity(ngon)
    cell_face = MT.Element.connectivity(nface)

    face_ln_to_gn = PT.get_np_value(MT.requestGlobalNumbering(ngon, 'Element'))

    return [cell_face.displs, cell_face.values, cell_ln_to_gn, \
        face_vtx.displs, face_vtx.values, face_ln_to_gn, vtx_coords, vtx_ln_to_gn]

  elif dim == 2:
    edge  = MT.Zone.EdgeNode(part_zone)
    ngon  = PT.Zone.NGonNode(part_zone)

    edge_pe  = PT.get_np_value(PT.find_child_from_name(edge, "ParentElements")).reshape(-1, order='C') # Numpy will copy
    edge_vtx = PT.get_np_value(PT.find_child_from_name(edge, "ElementConnectivity"))

    # Convert edge_pe to face_edge
    if PT.Element.Range(ngon)[0] != 1:
      np_utils.shift_nonzeros(edge_pe, -PT.Element.Range(ngon)[0] + 1)
    edge_pe[1::2] *= -1 # Put sign on right edges
    is_internal = edge_pe != 0
    edge_face = edge_pe[is_internal]
    edge_counts = is_internal[0::2].astype(np.int32) + is_internal[1::2].astype(np.int32)
    edge_face = vs.from_counts(edge_counts, edge_face)
    face_edge = PDM_connectivity_transpose(PT.Element.Size(ngon), edge_face)

    return [face_edge.displs, face_edge.values, cell_ln_to_gn, edge_vtx, vtx_coords, vtx_ln_to_gn]

  else:
    raise RuntimeError("Unsupported dimension")
                


def _get_part_data_elts(part_zone: CGNSTree) -> List[NDArray]:
  # Actually works for elt of S meshes, for which we rebuild cell_vtx connectivity
  cx, cy, cz = PT.Zone.coordinates(part_zone)
  assert (cx is not None) and  (cy is not None) and (cz is not None)
  coords = [c.reshape(-1, order='F') for c in [cx,cy,cz]]
  vtx_coords = np_utils.interweave_arrays(coords)

  dim = PT.Zone.CellDimension(part_zone)
  cell_vtx = cell_vtx_connectivity(part_zone, dim)

  vtx_ln_to_gn, _, _, cell_ln_to_gn = te_utils.get_entities_numbering(part_zone)
  assert (vtx_ln_to_gn is not None) and (cell_ln_to_gn is not None)

  return [cell_vtx.displs, cell_vtx.values, cell_ln_to_gn, vtx_coords, vtx_ln_to_gn]
    

@overload
def _mesh_location(src_parts: List[PartData],
                   tgt_clouds: List[PointCloud],
                   comm: MPIComm, 
                   reverse: Literal[False],
                   loc_tolerance: float) -> List[Result]: ...
@overload
def _mesh_location(src_parts: List[PartData],
                   tgt_clouds: List[PointCloud],
                   comm: MPIComm, 
                   reverse:Literal[True],
                   loc_tolerance: float) -> Tuple[List[Result], List[InvResult]]: ...

def _mesh_location(src_parts: List[PartData],
                   tgt_clouds: List[PointCloud],
                   comm: MPIComm, 
                   reverse: bool,
                   loc_tolerance: float = 1E-6) -> Union[List[Result],
                                                         Tuple[List[Result], List[InvResult]]]:
  """ Wrapper of PDM mesh location
  For now, only 1 domain is supported so we expect source parts and target clouds
  as flat lists :
  Parts are tuple (dim, elt_kind, part_data)
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
    all_located_inv = []
    for i_src_part in range(n_part_src):
      _cell_vtx_result = mesh_loc.cell_vertex_get(i_src_part)
      _loc_result      = mesh_loc.points_in_elt_get(0, i_src_part)
      # Select only usefull results, and convert into vs array
      # Convert some result into vsarray
      loc_result = {
        'points_weights' : vs.from_displs(_loc_result['points_weights_idx'], _loc_result['points_weights']),
        'points_gnum'    : vs.from_displs(_loc_result['elt_pts_inside_idx'], _loc_result['points_gnum']),
        'cell_vtx'       : vs.from_displs(_cell_vtx_result['cell_vtx_idx'], _cell_vtx_result['cell_vtx'])
      }
      all_located_inv.append(loc_result)
    
    return all_target_data, all_located_inv
  else:
    return all_target_data

def _mdom_mesh_location(src_parts_per_dom:List[List[PartData]], 
                        tgt_clouds_per_dom:List[List[PointCloud]], 
                        comm:MPIComm, 
                        reverse:bool,
                        loc_tolerance:float =1E-6) -> Union[List[List[Result]],
                                                            Tuple[List[List[Result]], List[List[InvResult]]]]:
  """
  Wraps _mesh_location with multidomain support (with shifts)
  Input are similar to _mesh_location, but with nested lists by domains
  Input data must not be shifted, it will be done by this function
  """
  
  n_part_per_dom_src = [len(parts) for parts in src_parts_per_dom ]
  n_part_per_dom_tgt = [len(parts) for parts in tgt_clouds_per_dom]

  # Shift target data first; we dont do it inplace since src and target data
  # may share the same memory
  tgt_offset = np.zeros(len(tgt_clouds_per_dom)+1, dtype=pdm_gnum_dtype)
  for i_domain, clouds in enumerate(tgt_clouds_per_dom):
    # Compute global offsets for this domain
    dom_max = par_utils.arrays_max([cloud[1] for cloud in clouds], comm)
    tgt_offset[i_domain+1] = tgt_offset[i_domain] + dom_max
    # Shift source arrays (copy)
    tgt_clouds_per_dom[i_domain] = [(c[0], c[1] + tgt_offset[i_domain]) for c in clouds]

  tgt_clouds = py_utils.to_flat_list(tgt_clouds_per_dom)
  src_parts  = py_utils.to_flat_list(src_parts_per_dom)

  # Now shift source data. First we need to eetrieve loc, which should be the same for each partition.
  kind = ''
  if len(src_parts) > 0:
    dim, kind = src_parts[0][:2]
    if kind == 'Poly':
      kind += str(dim)
  kind = comm.allreduce(kind, MPI.MAX) # To let empty procs know the data

  locs = {'Poly2'   : {'Cell':2, 'Vtx':5},           
          'Poly3'   : {'Cell':2, 'Face':5, 'Vtx':7},
          'Element' : {'Cell':2, 'Vtx':4},         
         }[kind]

  # Effective shift
  src_offsets = {loc : np.zeros(len(src_parts_per_dom)+1, dtype=pdm_gnum_dtype) for loc in locs}
  for i_domain, src_parts_domain in enumerate(src_parts_per_dom):
    # Compute global offsets for this domain
    for loc, array_idx in locs.items():
      dom_max = par_utils.arrays_max([src_part[2][array_idx] for src_part in src_parts_domain], comm)
      src_offsets[loc][i_domain+1] = src_offsets[loc][i_domain] + dom_max
    # Shift source arrays (inplace)
    for src_part in src_parts_domain:
      for loc, array_idx in locs.items():
        src_part[2][array_idx] += src_offsets[loc][i_domain]

  if reverse:
    direct_result, inv_result = _mesh_location(src_parts, tgt_clouds, comm, True, loc_tolerance)
  else:
    direct_result = _mesh_location(src_parts, tgt_clouds, comm, False, loc_tolerance)

  # Shift back source data
  for i_domain, src_parts_domain in enumerate(src_parts_per_dom):
    for src_part in src_parts_domain:
      for loc, array_idx in locs.items():
        src_part[2][array_idx] -= src_offsets[loc][i_domain]

  # Shift results and get domain ids
  for tgt_result in direct_result:
    tgt_result['location_shifted'] = tgt_result.pop('location') #Rename key
    tgt_result['location'], tgt_result['domain'] = np_utils.shifted_to_local(
        tgt_result['location_shifted'], src_offsets['Cell'])
  if reverse:
    for src_result in inv_result:
      src_result['points_gnum_shifted'] = src_result.pop('points_gnum') #Rename key
      ini_gnum, domain = np_utils.shifted_to_local(src_result['points_gnum_shifted'].values, tgt_offset)
      src_result['points_gnum'] = vs.from_displs(src_result['points_gnum_shifted'].displs, ini_gnum)
      src_result['domain']      = vs.from_displs(src_result['points_gnum_shifted'].displs, domain)
  
  # Reshape output to list of lists (as input domains)
  if reverse:
    return py_utils.to_nested_list(direct_result, n_part_per_dom_tgt),\
           py_utils.to_nested_list(inv_result, n_part_per_dom_src)
  else:
    return py_utils.to_nested_list(direct_result, n_part_per_dom_tgt)

def _collect_source(src_parts_per_dom:List[List[CGNSPartTree]]) -> List[List[PartData]]:
  connectivity_t = None
  src_parts = []
  for src_part_zones in src_parts_per_dom:

    src_parts_domain = list()
    for src_part in src_part_zones:
      dim = PT.Zone.CellDimension(src_part)
      if PT.Zone.has_ngon_elements(src_part):
        if connectivity_t=='Element':
          raise NotImplementedError("Source mesh must have NGon or Element connectivity but not both.")
        connectivity_t = f'NGon{dim}D'
        src_parts_domain.append((dim, 'Poly', _get_part_data_ngon(src_part)))
      else:
        if connectivity_t is not None and 'NGON' in connectivity_t:
          raise NotImplementedError("Source mesh must have NGon or Element connectivity but not both.")
        connectivity_t = 'Element'
        src_parts_domain.append((dim, 'Element', _get_part_data_elts(src_part)))
    src_parts.append(src_parts_domain)

  return src_parts

def _collect_target(tgt_parts_per_dom:List[List[CGNSPartTree]], location:str) -> List[List[PointCloud]]:
  return [[get_point_cloud(part, location) for part in tgt_parts] \
          for tgt_parts in tgt_parts_per_dom]



@overload
def _localize_points(
  src_parts_per_dom: List[List[CGNSPartTree]], 
  tgt_parts_per_dom: List[List[CGNSPartTree]], 
  location: str,
  comm: MPIComm, 
  reverse: Literal[False],
  loc_tolerance: float) -> List[List[Result]]: ...
@overload
def _localize_points(
  src_parts_per_dom: List[List[CGNSPartTree]], 
  tgt_parts_per_dom: List[List[CGNSPartTree]], 
  location: str,
  comm: MPIComm, 
  reverse: Literal[True],
  loc_tolerance: float) -> Tuple[List[List[Result]], List[List[InvResult]]]: ...

def _localize_points(
  src_parts_per_dom: List[List[CGNSPartTree]], 
  tgt_parts_per_dom: List[List[CGNSPartTree]], 
  location: str,
  comm: MPIComm, 
  reverse: bool = False, 
  loc_tolerance: float = 1E-6) -> Union[List[List[Result]],
                                        Tuple[List[List[Result]], List[List[InvResult]]]]:

  """ Intermediate API who do not place output in tree.
  Inputs are list of size n_domain_src (resp. tgt) containing partitioned zones (resp. clouds)
  for each domain 
  """
  src_parts  = _collect_source(src_parts_per_dom)
  tgt_clouds = _collect_target(tgt_parts_per_dom, location)

  return _mdom_mesh_location(src_parts, tgt_clouds, comm, reverse, loc_tolerance)



def localize_points(src_tree: CGNSPartTree, 
                    tgt_tree: CGNSPartTree, 
                    location: str, 
                    comm: MPIComm, 
                    **options) -> None:
  """
  Partitioned implementation of maia.algo.localize_points
  """
  MT.check_cgns_part_tree(src_tree)
  MT.check_cgns_part_tree(tgt_tree)
  _src_parts_per_dom = get_parts_per_blocks(src_tree, comm)
  src_parts_per_dom = list(_src_parts_per_dom.values())
  tgt_parts_per_dom = list(get_parts_per_blocks(tgt_tree, comm).values())

  located_data = _localize_points(src_parts_per_dom, tgt_parts_per_dom, location, comm, **options)

  dom_list = '\n'.join(_src_parts_per_dom.keys())
  for i_dom, tgt_parts in enumerate(tgt_parts_per_dom):
    for i_part, tgt_part in enumerate(tgt_parts):
      sol = PT.update_child(tgt_part, "Localization", "DiscreteData_t")
      PT.new_GridLocation(location, sol)
      data = located_data[i_dom][i_part]
      n_tgts = data['located_ids'].size + data['unlocated_ids'].size,
      src_gnum = -np.ones(n_tgts, dtype=pdm_gnum_dtype) #Init with -1 to carry unlocated points
      src_dom  = -np.ones(n_tgts, dtype=np.int32)
      src_gnum[data['located_ids']] = data['location']
      src_dom [data['located_ids']] = data['domain']
      # For structured meshes, reshape result
      if PT.Zone.Type(tgt_part) == 'Structured':
        if n_tgts == PT.Zone.n_vtx(tgt_part):
          shape = PT.Zone.VertexSize(tgt_part)
        elif n_tgts == PT.Zone.n_cell(tgt_part):
          shape = PT.Zone.CellSize(tgt_part)
        else:
          raise RuntimeError("Unable to detect target location")
        src_gnum = src_gnum.reshape(shape, order='F')
        src_dom  = src_dom.reshape(shape, order='F')

      PT.new_DataArray("SrcId", src_gnum, parent=sol)
      PT.new_DataArray("DomId", src_dom,  parent=sol)
      PT.new_Descriptor("DomainList", dom_list, parent=sol)

