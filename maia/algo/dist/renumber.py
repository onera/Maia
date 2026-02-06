from mpi4py import MPI
import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.transfer import protocols as EP
from maia.utils import vstride as vs
from maia.utils import par_utils, np_utils

from .sections_tools import _concatenate_elt_sections, concatenate_elt_sections_if

from maia.typing import *

def is_elt_of_dim(dim:int) -> PT.pred.NodePredicate:
  return PT.pred.label_is('Elements_t') & PT.pred.NodePredicate(lambda e: PT.Element.Dimension(e)==dim)

def subdistri(distri:NDArray, start:int, end:int):
    intersect_starts = np.maximum(distri[:-1], start)
    intersect_ends = np.minimum(distri[1:], end)
    
    counts = np.maximum(intersect_ends - intersect_starts, 0)
    return np_utils.sizes_to_indices(counts)

def _local_bounds(ini_start_loc:int, ini_end_loc:int, g_start:int, g_end:int) -> Tuple[int, int]:
  ini_size = ini_end_loc - ini_start_loc
  r_start = max(ini_start_loc, g_start) - ini_start_loc
  r_end   = min(ini_end_loc, g_end) - ini_start_loc
  
  return min(r_start, ini_size), max(r_end, 0)

def local_bounds(distri:NDArray, g_start:int, g_end:int) -> Tuple[int, int]:
  """ Compute the local start/end indices that should be used to extract a slice of 
  a distributed array, restricted to the [g_start:g_end[ interval """
  return _local_bounds(distri[0], distri[1], g_start, g_end)
  


def _collect_shifted_pl_one(subset:CGNSTree, shift:int=0, donor:bool=False) -> NDArray:
  suffix = 'Donor' if donor else ''
  if (pl := PT.get_child_from_name(subset, f'PointList{suffix}')) is not None:
    _pl = PT.get_np_value(pl)[0]
  elif (pr := PT.get_child_from_name(subset, f'PointRange{suffix}')) is not None:
    distri = MT.Subset.distribution(subset)
    _pl = np_utils.single_dim_pr_to_pl(PT.get_np_value(pr), distri)[0]
  else:
    raise RuntimeError(f"Missing patch in subset node {PT.get_name(subset)}")
  return _pl + shift

def _collected_shifted_pl(zone:CGNSTree, loc:str, shift:int) -> List[NDArray]:
  return [_collect_shifted_pl_one(subset, shift) for subset in PT.iter_all_subsets(zone, loc)]

def _update_pl_one(subset:CGNSTree, new_pl:NDArray, shift:int=0, donor:bool=False):
  suffix = 'Donor' if donor else ''
  PT.rm_children_from_name(subset, f'PointList{suffix}')
  PT.rm_children_from_name(subset, f'PointRange{suffix}')
  PT.new_IndexArray(f'PointList{suffix}',
                    value=new_pl.reshape((1,-1), order='F') + shift,
                    parent=subset)

def _update_pl(zone:CGNSTree, loc:str, new_pl:List[NDArray], shift:int=0):
  for subset, _pl in zip(PT.iter_all_subsets(zone, loc), new_pl):
    _update_pl_one(subset, _pl, shift)
    # NB : PointListDonor of GCs will be copied afterward (under usual assumption that PL are symmetric) 

def _adapt_data_to_distri(data_in:NDArray, distri_in:NDArray, distri_out:NDArray, comm:MPIComm) -> NDArray:
  # Perform a BtB if necessary to have data distributed as distri_out
  _distri_in  = par_utils.auto_expand_distri(distri_in, comm)
  _distri_out = par_utils.auto_expand_distri(distri_out, comm)
  if not par_utils.is_same_distri(_distri_in, _distri_out, comm):
    return EP.block_to_block(data_in, _distri_in, _distri_out, comm)
  else:
    return data_in

def _update_point_lists(tree:CGNSDistTree, zone_path:CGNSPath,
                        new_id:NDArray, id_distri:NDArray,
                        loc:str, offset:int, comm:MPIComm):
  """
  Apply the entity renumbering to the PointLists of specified location
  PointList donor on opposite zones are updated as well
  """
  zone = PT.find_node_from_path(tree, zone_path)
  # Renumber PointLists

  old_pls = _collected_shifted_pl(zone, loc, -offset)
  new_pls = EP.block_to_part(new_id, id_distri, old_pls, comm)
  _update_pl(zone, loc, new_pls, offset)

  # Loop on others zones to update PLDonor
  GC_PRED = PT.pred.is_gc_of_kind(is_1to1=True) & PT.pred.has_location(f'*{loc}') # * is to catch hybrid GC
  for opp_base, opp_zone in PT.iter_children_from_predicates(tree, 'CGNSBase_t/Zone_t', ancestors=True):
    for gc in PT.iter_children_from_predicates(opp_zone, ['ZoneGridConnectivity_t', GC_PRED]):
      if PT.GridConnectivity.ZoneDonorPath(gc, PT.get_name(opp_base)) == zone_path:
        pld = _collect_shifted_pl_one(gc, -offset, True)
        new_pld = EP.block_to_part(new_id, id_distri, pld, comm)
        _update_pl_one(gc, new_pld, offset, True)

def _update_full_cellcenter_containers(zone:CGNSTree, new_id:NDArray, new_id_distri:NDArray, comm:MPIComm):
  """ Apply the entity renumbering to the full CellCenter containers"""
  cell_distri = MT.Zone.cell_distribution(zone)
  _cell_distri = par_utils.partial_to_full_distribution(cell_distri, comm)
  new_id_cell = _adapt_data_to_distri(new_id, new_id_distri, _cell_distri, comm)
  GI = EP.GlobalIndexer(_cell_distri, new_id_cell, comm)
  for array_n in PT.iter_children_from_predicates(zone, [MT.pred.FULL_CTN_CELL, 'DataArray_t']):
    array = PT.get_np_value(array_n)
    GI.Put(array, array)


def _renumber_std_sections_of_dim(zone, dim, input_distri_f, new_id, comm):
  # Renumber the standard elements of the specified dimension. They
  # will be concatened according to their kind

  pred = is_elt_of_dim(dim) # Concatenate elts of selected dim only
  concatenate_elt_sections_if(zone, pred, comm) #type:ignore[arg-type] # zone is distributed

  elts = sorted(PT.get_children_from_predicate(zone, pred), key=lambda e: PT.Element.Range(e)[0])
  offset = PT.Element.Range(elts[0])[0]

  input_distri = par_utils.full_to_partial_distribution(input_distri_f, comm)
  # Work (cat) section by (cat) section
  for elt in elts:
    distri = MT.Element.distribution(elt)
    global_start = PT.Element.Range(elt)[0] - offset
    global_end = global_start + PT.Element.Size(elt)
    _restrict = subdistri(input_distri_f, global_start, global_end)
    view_st, view_end = local_bounds(input_distri, global_start, global_end)
    new_id_loc = EP.block_to_block(new_id[view_st:view_end], _restrict, distri, comm)

    # Check that renumbering does not goes out of the current section
    low  = PT.Element.Range(elt)[0] - offset
    high = PT.Element.Range(elt)[1] - offset
    is_compatible = bool(np.all(low <= new_id_loc) and np.all(new_id_loc <= high))
    if not comm.allreduce(is_compatible, MPI.LAND):
      raise ValueError("Invalid permutation: elements of different type would be interlaced")

    # Do effective renumbering of elements
    GI = EP.GlobalIndexer(distri, new_id_loc-global_start, comm)
    elt_vtx_n = PT.find_child_from_name(elt, 'ElementConnectivity')
    elt_vtx = PT.get_np_value(elt_vtx_n)
    GI.Put(elt_vtx, elt_vtx, count=PT.Element.NVtx(elt))


def renumber_vertices(tree:CGNSDistTree, zone_path:CGNSPath, new_vtx_id:NDArray, comm:MPIComm):
  """
  Renumber vertices of the input zone.
  new_vtx_id is a distributed array, which associate
  to each old vtx it's new id (0-based)
  PointListDonor from other zones are also updated
  """
  zone = PT.find_node_from_path(tree, zone_path)

  # Ensure that new_vtx_id is distributed as 'ALL_VTX' distribution
  input_distri = par_utils.dn_to_distribution(new_vtx_id.size, comm)
  vtx_distri = MT.Zone.vtx_distribution(zone)
  new_vtx_id = _adapt_data_to_distri(new_vtx_id, input_distri, vtx_distri, comm)

  # Update Elements
  elts =  PT.get_children_from_predicate(zone, PT.pred.label_is('Elements_t')
                                            & ~PT.pred.is_element_of_type('NFACE_n'))
  ec_nodes = [PT.find_child_from_name(elt, 'ElementConnectivity') for elt in elts]
  ec_values = [PT.get_np_value(ec_node) - 1 for ec_node in ec_nodes]
  new_cnt_l = EP.block_to_part(new_vtx_id, vtx_distri, ec_values, comm)
  for ec_node, value in zip(ec_nodes, new_cnt_l):
    PT.set_value(ec_node, value+1)

  # Update PL data
  _update_point_lists(tree, zone_path, new_vtx_id, vtx_distri, 'Vertex', 1, comm)

  # Update full solutions (including coordinates)
  GI = EP.GlobalIndexer(vtx_distri, new_vtx_id, comm)

  # Coordinates
  for co in PT.Zone.coordinates(zone):
    if co is not None:
      GI.Put(co, co)
  # Full solutions
  for array_n in PT.iter_children_from_predicates(zone, [MT.pred.FULL_CTN_VTX, 'DataArray_t']):
    array = PT.get_np_value(array_n)
    GI.Put(array, array)


def renumber_edges(tree:CGNSDistTree, zone_path:CGNSPath, new_edge_id:NDArray, comm:MPIComm):
  """
  Renumber edges of the input zone.
  new_edge_id is a distributed array, which associates to each old edge it's new id (0-based)
  PointListDonor from other zones are also updated
  Note : if several edges sections are present in tree, we can not garantee that
  ordering does not interlace sections. Thus we concatenate edge section to a single one.
  """
  pred = is_elt_of_dim(1)
  zone = PT.find_node_from_path(tree, zone_path)

  is_native_dim = PT.Zone.CellDimension(zone) == 1

  edge_elts = PT.get_children_from_predicate(zone, pred)
  if len(edge_elts) == 0:
    return
  elif len(edge_elts) == 1:
    edge_elt = edge_elts[0]
  else:
    edge_elt = _concatenate_elt_sections(edge_elts, comm)
    PT.set_name(edge_elt, 'BAR_2')
    PT.rm_children_from_predicate(zone, pred)
    PT.add_child(zone, edge_elt)

  # Check if distribution of input new_edge_id and EdgeNode are identical
  # If not, get new_edge_id on element distribution
  input_distri = par_utils.dn_to_distribution(new_edge_id.size, comm)
  edge_distri  = MT.Element.distribution(edge_elt)
  _input_distri = par_utils.partial_to_full_distribution(input_distri, comm)
  _edge_distri  = par_utils.partial_to_full_distribution(edge_distri, comm)

  new_edge_id_elt = _adapt_data_to_distri(new_edge_id, _input_distri, _edge_distri, comm)

  # Now we can reorder edges
  edge_offset = PT.Element.Range(edge_elt)[0]
  GI = EP.GlobalIndexer(_edge_distri, new_edge_id_elt, comm)

  edge_vtx_n = PT.find_child_from_name(edge_elt, 'ElementConnectivity')
  edge_vtx = PT.get_np_value(edge_vtx_n)
  GI.Put(edge_vtx, edge_vtx, count=2)

  # Update PE if existing (inplace ok because distri did not change)
  if (pe_n := PT.get_child_from_name(edge_elt, 'ParentElements')) is not None:
    pe = PT.get_np_value(pe_n)
    GI.Put(pe[:,0], pe[:,0])
    GI.Put(pe[:,1], pe[:,1])

  # Update EdgeCenter PointList (no data to move, because full EdgeCenter data not allowed)
  loc = 'CellCenter' if is_native_dim else 'EdgeCenter'
  _update_point_lists(tree, zone_path, new_edge_id, _input_distri, loc, edge_offset, comm)

  # Update full solutions, if edges are native dim
  if is_native_dim:
    _update_full_cellcenter_containers(zone, new_edge_id, _input_distri, comm)




def renumber_faces(tree:CGNSDistTree, zone_path:CGNSPath, new_face_id:NDArray, comm:MPIComm):
  """
  Renumber faces of the input zone.
  new_face_id is a distributed array, which associates to each old face it's new id (0-based)
  PointListDonor from other zones are also updated
  Note for std meshes: if several faces sections are present in tree, we can not garantee that
  ordering does not interlace sections. Thus we concatenate faces section to a single one.
  In the permutation leads to interlacement of faces of different kind (tri, quad), an
  error is raised
  """
  zone = PT.find_node_from_path(tree, zone_path)

  is_native_dim = PT.Zone.CellDimension(zone) == 2

  input_distri = par_utils.dn_to_distribution(new_face_id.size, comm)
  _input_distri = par_utils.partial_to_full_distribution(input_distri, comm)

  if PT.pred.IS_POLY2D_ZONE(zone) or PT.pred.IS_POLY3D_ZONE(zone):

    if PT.Zone.has_ngon_elements(zone):
      face_offset = PT.Element.Range(PT.Zone.NGonNode(zone))[0]
    else: # Zone is poly2d with edges only
      face_offset = PT.Element.Range(MT.Zone.EdgeNode(zone))[1] + 1

    # If EdgeElements are present (poly2d zone), update ParentElement of edges
    # if existing since it indexes faces
    if PT.get_node_from_predicate(zone, PT.pred.is_element_of_type('BAR_2')) is not None:
      assert PT.Zone.CellDimension(zone) == 2
      ne = MT.Zone.EdgeNode(zone)
      pe_n = PT.get_child_from_name(ne, 'ParentElements')
      if pe_n is not None:
        pe = PT.get_np_value(pe_n)
        mask = (pe != 0)
        pe[mask] = EP.block_to_part(new_face_id, _input_distri, pe[mask]-face_offset, comm) + face_offset

    # If NG are present (poly2d or poly3d zone), move connectivity / parent elements
    if PT.Zone.has_ngon_elements(zone):
      ng = PT.Zone.NGonNode(zone)
      face_distri = MT.Element.distribution(ng)
      # Ensure that new_face_id is distributed as NG/Distribution
      new_face_id_elt = _adapt_data_to_distri(new_face_id, _input_distri, face_distri, comm)
      GI = EP.GlobalIndexer(face_distri, new_face_id_elt, comm)
      
      face_vtx_ini = MT.Element.connectivity(ng)
      face_vtx = vs.from_counts(*GI.Put_v((face_vtx_ini.counts, face_vtx_ini.values)))
      # Distribution of ElementConnectivity can change (the one of Element is fixed)
      elt_distri = par_utils.dn_to_distribution(face_vtx.dsize, comm) # JC TODO EXSCAN
      PT.update_child(ng, 'ElementStartOffset', value=face_vtx.displs + elt_distri[0].astype(face_vtx.displs.dtype))
      PT.update_child(ng, 'ElementConnectivity', value=face_vtx.values)

      # Update PE if existing (inplace ok because face distri did not change)
      if (pe_n := PT.get_child_from_name(ng, 'ParentElements')) is not None:
        pe = PT.get_np_value(pe_n)
        GI.Put(pe[:,0], pe[:,0])
        GI.Put(pe[:,1], pe[:,1])

      # Poly3d zone; update cell-->face connectivity if existing
      if PT.Zone.has_nface_elements(zone):
        nf = PT.Zone.NFaceNode(zone)
        cell_face_n = PT.find_child_from_name(nf, 'ElementConnectivity')
        cell_face = PT.get_np_value(cell_face_n)
        sign = np.sign(cell_face)
        val  = np.abs(cell_face)
        GI = EP.GlobalIndexer(face_distri, val-face_offset, comm)
        GI.Take(new_face_id_elt, cell_face)
        cell_face += face_offset
        cell_face *= sign

  else: # Standard elements
    _renumber_std_sections_of_dim(zone, 2, _input_distri, new_face_id, comm)
    face_offset = PT.Zone.get_elt_range_per_dim(zone)[2][0]
  
  # Update FaceCenter PointList (no data to move, because full FaceCenter data not allowed)
  loc = 'CellCenter' if is_native_dim else 'FaceCenter'
  _update_point_lists(tree, zone_path, new_face_id, _input_distri, loc, face_offset, comm)

  # Update full solutions, if edges are native dim
  if is_native_dim:
    _update_full_cellcenter_containers(zone, new_face_id, _input_distri, comm)


def renumber_cells(tree:CGNSDistTree, zone_path:CGNSPath, new_cell_id:NDArray, comm:MPIComm):
  """
  Renumber cells of the input zone.
  new_cell_id is a distributed array, which associates to each old cell it's new id (0-based)
  PointListDonor from other zones are also updated
  Note for std meshes: if several cells sections are present in tree, we can not garantee that
  ordering does not interlace sections. Thus we concatenate cells section to a single one.
  In the permutation leads to interlacement of cells of different kind (tetra, hexa), an
  error is raised
  """
  zone = PT.find_node_from_path(tree, zone_path)

  input_distri = par_utils.dn_to_distribution(new_cell_id.size, comm)
  _input_distri = par_utils.partial_to_full_distribution(input_distri, comm)

  if PT.pred.IS_POLY3D_ZONE(zone):

    if PT.Zone.has_nface_elements(zone):
      cell_offset = PT.Element.Range(PT.Zone.NFaceNode(zone))[0]
    else: # Zone is poly3d with faces only
      cell_offset = PT.Element.Range(PT.Zone.NGonNode(zone))[1] + 1

    # If ParentElements is present in NGON node, update it since it indexes cells
    ng = PT.Zone.NGonNode(zone)
    pe_n = PT.get_child_from_name(ng, 'ParentElements')
    if pe_n is not None:
      pe = PT.get_np_value(pe_n)
      mask = (pe != 0)
      pe[mask] = EP.block_to_part(new_cell_id, _input_distri, pe[mask]-cell_offset, comm) + cell_offset

    # Move cell_face connectivity
    if PT.Zone.has_nface_elements(zone):
      nf = PT.Zone.NFaceNode(zone)

      # Ensure that new_face_id is distributed as NG/Distribution
      cell_distri = MT.Element.distribution(nf)
      new_cell_id_elt = _adapt_data_to_distri(new_cell_id, _input_distri, cell_distri, comm)

      cell_face_ini = MT.Element.connectivity(nf)
      cell_face = EP.part_to_block(cell_face_ini, cell_distri, new_cell_id_elt, comm)

      # Distribution of ElementConnectivity can change (the one of Element is fixed)
      elt_distri = par_utils.dn_to_distribution(cell_face.dsize, comm) # JC TODO EXSCAN
      PT.update_child(nf, 'ElementStartOffset', value=cell_face.displs + elt_distri[0].astype(cell_face.displs.dtype))
      PT.update_child(nf, 'ElementConnectivity', value=cell_face.values)


  else: # Standard elements
    _renumber_std_sections_of_dim(zone, 3, _input_distri, new_cell_id, comm)
    cell_offset = PT.Zone.get_elt_range_per_dim(zone)[3][0]

  
  # Update CellCenter PointList
  _update_point_lists(tree, zone_path, new_cell_id, _input_distri, 'CellCenter', cell_offset, comm)

  # Update full solutions (cell always native dim)
  _update_full_cellcenter_containers(zone, new_cell_id, _input_distri, comm)

