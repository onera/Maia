import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.transfer import protocols as EP
from maia.utils import vstride as vs
from maia.utils import par_utils, np_utils

from .sections_tools import _concatenate_elt_sections

from maia.typing import *

def _collect_shifted_pl_one(subset:CGNSTree, shift:int=0, donor:bool=False) -> NDArray:
  suffix = 'Donor' if donor else ''
  if (pl := PT.get_child_from_name(subset, f'PointList{suffix}')) is not None:
    _pl = PT.get_np_value(pl)[0]
  elif (pr := PT.get_child_from_name(subset, f'PointRange{suffix}')) is not None:
    distri = MT.distribution_value(subset, 'Index')
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

def _renumber_pl_donor(tree, zone_path, loc, distri, new_id, pl_offset, comm):
  GC_PRED = PT.pred.is_gc_of_kind(is_1to1=True) & PT.pred.has_location(loc)
  for opp_base, opp_zone in PT.iter_children_from_predicates(tree, 'CGNSBase_t/Zone_t', ancestors=True):
    for gc in PT.iter_children_from_predicates(opp_zone, ['ZoneGridConnectivity_t', GC_PRED]):
      if PT.GridConnectivity.ZoneDonorPath(gc, PT.get_name(opp_base)) == zone_path:
        pld = _collect_shifted_pl_one(gc, -pl_offset, True)
        new_pld = EP.block_to_part(new_id, distri, pld, comm)
        _update_pl_one(gc, new_pld, pl_offset, True)


def renumber_vertices(tree, zone_path, new_vtx_id, comm):
  """
  Renumber vertices of the input zone.
  new_vtx_id is an array distributed as ALL_VTX, which associate
  to each old vtx it's new id (0-based)
  PointListDonor from other zones are also updated
  """
  zone = PT.find_node_from_path(tree, zone_path)
  vtx_distri = MT.distribution_value(zone, 'Vertex')

  GI = EP.GlobalIndexer(vtx_distri, new_vtx_id, comm)

  # Update Coordinates
  for co in PT.Zone.coordinates(zone):
    if co is not None:
      GI.Put(co, co)
  # Update full solutions
  for array_n in PT.iter_children_from_predicates(zone, [MT.pred.FULL_CTN_VTX, 'DataArray_t']):
    array = PT.get_np_value(array_n)
    GI.Put(array, array)

  # Update Elements
  ec_nodes = list()
  for elt in PT.iter_children_from_predicate(zone, 'Elements_t'):
    if PT.Element.Type(elt) == 'NFACE_n':
      continue
    ec_nodes.append(PT.find_child_from_name(elt, 'ElementConnectivity'))
  ec_values = [PT.get_np_value(ec_node) - 1 for ec_node in ec_nodes]
  new_cnt_l = EP.block_to_part(new_vtx_id, vtx_distri, ec_values, comm)
  for ec_node, value in zip(ec_nodes, new_cnt_l):
    PT.set_value(ec_node, value+1)

  # Update PL data
  old_pls = _collected_shifted_pl(zone, 'Vertex', -1)
  new_pls = EP.block_to_part(new_vtx_id, vtx_distri, old_pls, comm)
  _update_pl(zone, 'Vertex', new_pls, 1)

  # Loop on others zones to update PLDonor
  _renumber_pl_donor(tree, zone_path, 'Vertex', vtx_distri, new_vtx_id, 1, comm)



def renumber_edges(tree, zone_path, new_edge_id, comm):
  """
  Renumber edges of the input zone.
  new_edge_id is a distributed array, which associate to each old edge it's new id (0-based)
  PointListDonor from other zones are also updated
  Note : if several edges sections are present in tree, we can not garantee that
  ordering do not interlace sections. Thus we concatenate edge section to a single one.
  """
  pred = PT.pred.is_element_of_type('BAR_2')
  zone = PT.find_node_from_path(tree, zone_path)

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
  edge_distri  = MT.distribution_value(edge_elt, 'Element')
  _input_distri = par_utils.partial_to_full_distribution(input_distri, comm)
  _edge_distri  = par_utils.partial_to_full_distribution(edge_distri, comm)
  if not np.array_equal(_input_distri, _edge_distri):
    new_edge_id = EP.block_to_block(new_edge_id, _input_distri, _edge_distri, comm)

  # Now we can reorder edges
  edge_offset = PT.Element.Range(edge_elt)[0]
  GI = EP.GlobalIndexer(_edge_distri, new_edge_id, comm)

  edge_vtx_n = PT.find_child_from_name(edge_elt, 'ElementConnectivity')
  edge_vtx = PT.get_np_value(edge_vtx_n)
  GI.Put(edge_vtx, edge_vtx, count=2)

  # Update PE if existing (inplace ok because face distri did not change)
  if (pe_n := PT.get_child_from_name(edge_elt, 'ParentElements')) is not None:
    pe = PT.get_np_value(pe_n)
    GI.Put(pe[:,0], pe[:,0])
    GI.Put(pe[:,1], pe[:,1])

  # Update EdgeCenter PointList (no data to move, because full EdgeCenter data not allowed)
  old_pls = _collected_shifted_pl(zone, 'EdgeCenter', -edge_offset)
  new_pls = EP.block_to_part(new_edge_id, edge_distri, old_pls, comm)
  _update_pl(zone, 'EdgeCenter', new_pls, edge_offset)

  # Loop on others zones to update PLDonor
  _renumber_pl_donor(tree, zone_path, '*EdgeCenter', edge_distri, new_edge_id, edge_offset, comm)


def renumber_faces(tree, zone_path, new_face_id, comm):
  """
  Renumber faces of the input zone.
  new_face_id is an array distributed as ALL_FACES, which associate
  to each old face it's new id (0-based)
  Only NG zones are supported
  PointListDonor from other zones are also updated
  """
  zone = PT.find_node_from_path(tree, zone_path)
  if PT.pred.IS_POLY3D_ZONE(zone):
    # 1. NGon node : update face_vtx + pe (if existing)
    ng = PT.Zone.NGonNode(zone)
    ng_offset = PT.Element.Range(ng)[0]
    face_distri = MT.distribution_value(ng, 'Element')
    GI = EP.GlobalIndexer(face_distri, new_face_id, comm)
    
    face_vtx_ini = MT.Element.connectivity(ng)
    face_vtx = vs.from_counts(*GI.Put_v((face_vtx_ini.counts, face_vtx_ini.values)))
    # Distribution of ElementConnectivity can change (the one of Element is fixed)
    elt_distri = par_utils.dn_to_distribution(face_vtx.dsize, comm)
    PT.update_child(ng, 'ElementStartOffset', value=face_vtx.displs + elt_distri[0].astype(face_vtx.displs.dtype))
    PT.update_child(ng, 'ElementConnectivity', value=face_vtx.values)

    # Update PE if existing (inplace ok because face distri did not change)
    if (pe_n := PT.get_child_from_name(ng, 'ParentElements')) is not None:
      pe = PT.get_np_value(pe_n)
      GI.Put(pe[:,0], pe[:,0])
      GI.Put(pe[:,1], pe[:,1])

    MT.new_Distribution({'ElementConnectivity' : elt_distri}, parent=ng)

    # 2. NFACE node (if existing) : update cell->face connectivity
    if PT.Zone.has_nface_elements(zone):
      nf = PT.Zone.NFaceNode(zone)
      cell_face_n = PT.find_child_from_name(nf, 'ElementConnectivity')
      cell_face = PT.get_np_value(cell_face_n)
      sign = np.sign(cell_face)
      val  = np.abs(cell_face)
      GI = EP.GlobalIndexer(face_distri, val-ng_offset, comm)
      GI.Take(new_face_id, cell_face)
      cell_face += ng_offset
      cell_face *= sign

    # 3. Update FaceCenter PointList (no data to move, because full FaceCenter data not allowed)
    old_pls = _collected_shifted_pl(zone, 'FaceCenter', -ng_offset)
    new_pls = EP.block_to_part(new_face_id, face_distri, old_pls, comm)
    _update_pl(zone, 'FaceCenter', new_pls, ng_offset)

  else:
    pass

  # Loop on others zones to update PLDonor
  _renumber_pl_donor(tree, zone_path, '*FaceCenter', face_distri, new_face_id, ng_offset, comm)

