import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

import maia

from .point_cloud_utils  import get_point_cloud
from .localize           import minimal_partitioning
from .extract_surf_dmesh import extract_surf_from_bc

from maia.algo.part import closest_elt as pclosest_elt
from maia.transfer import protocols as EP
from maia.utils import np_utils

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

def detect_perio(dist_tree:CGNSDistTree, comm:MPIComm) -> Dict[str, List[PT.PeriodicValues]]:
  """
  Create dist_zone_path -> [periodicities] for input tree
  """
  grouped_zone_paths = PT.Tree.find_connected_zones(dist_tree)
  periodicities_per_path = dict()
  for group in grouped_zone_paths:
    fake_tree = PT.new_CGNSTree()
    fake_base = PT.new_CGNSBase(group[0].split('/')[0], parent=fake_tree)
    for zone_path in group:
      PT.add_child(fake_base, PT.get_node_from_path(dist_tree, zone_path))
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

def minimal_partitioning_poly2D(zone, comm):
  """ Adaptation of minimal_partitioning suitable for 
  2D trees without EdgeNode (+ direct return of face_vtx) """
  
  if not PT.Zone.has_ngon_elements(zone):
    maia.algo.edge_pe_to_ngon(zone, comm)
  ngon = PT.Zone.NGonNode(zone)

  face_distri = MT.Element.distribution(ngon)
  face_gnum = np.arange(face_distri[0]+1, face_distri[1]+1, dtype=face_distri.dtype)

  pface_vtx = MT.Element.connectivity(ngon)

  vtx_gnum, inverse = np.unique(pface_vtx.values, return_inverse=True)
  vtx_gnum = vtx_gnum.astype(face_gnum.dtype, copy=False)
  pface_vtx_idx = pface_vtx.displs.astype(np.int32, copy=False)
  pface_vtx_val = np.arange(1, len(vtx_gnum)+1, dtype=np.int32)[inverse]

  vtx_distri = MT.Zone.vtx_distribution(zone)
  dcoords = PT.Zone.coordinates(zone)
  dcoords = {key: data if data is not None else np.zeros_like(dcoords[0]) \
              for key, data in dcoords._asdict().items()}
  pcoords = EP.block_to_part(dcoords, vtx_distri, vtx_gnum-1, comm)
  pvtx_coords = np_utils.interweave_arrays(list(pcoords.values()))

  return pface_vtx_idx, pface_vtx_val, pvtx_coords, face_gnum, vtx_gnum
  
def update_closest_to_parent(surface_tree, points_tree, comm):
  # Update result (surface id) to refer to the corresponding parent face in volumic tree
  # with a PartToPart (part 1 = surfacic face, part 2 = closest face ids, data = parent)
  # Get part 1 + data (lngn shifted because of domains)
  face_parent_gnum_l = []
  face_ln_to_gn_l = []

  zone = PT.find_node_from_label(points_tree, 'Zone_t')
  domlist = PT.get_str_value(PT.find_node_from_path(zone, 'ClosestElement/DomainList')).split('\n')

  # Important : surface must be collected respecting DomainList order
  # to apply good offset
  ini_zone_offset = np.zeros(len(domlist)+1, int)
  for i, zone_path in enumerate(domlist):
    surf_zone = PT.find_node_from_path(surface_tree, zone_path)
    face_parent_gnum_l.append(PT.get_node_from_path(surf_zone, 'DiscreteData/Parent')[1])
    distri = MT.Zone.cell_distribution(surf_zone)
    gnum = np.arange(distri[0]+1+ini_zone_offset[i], distri[1]+1+ini_zone_offset[i])
    face_ln_to_gn_l.append(gnum) # -> Surface gnum for each partition of the surface, shifted ignoring periodics
    ini_zone_offset[i+1] = ini_zone_offset[i] + MT.Zone.n_cell(surf_zone)
  
  if ini_zone_offset[-1] == 0:
    # Early exit if ini_zone_offset = 0 (no surface in tree)
    return

  # Get part 2 (use same shift)
  closest_elt_gnum = []

  for part_zone in PT.get_all_Zone_t(points_tree):
    closest_dom = PT.find_node_from_path(part_zone, 'ClosestElement/ClosestEltDomId')[1]
    gnum = PT.find_node_from_path(part_zone, 'ClosestElement/ClosestEltGnum')[1]

    closest_elt_gnum.append(gnum + ini_zone_offset[closest_dom])

  # PartToPart to put back the ClosestEltGnum in volumic numbering (construct it only once)
  closest_parent_face = EP.part_to_part(face_parent_gnum_l, face_ln_to_gn_l, closest_elt_gnum, comm)

  for closest_parent, part_zone in zip(closest_parent_face, PT.get_all_Zone_t(points_tree)):
    fs_node = PT.get_child_from_name(part_zone, 'ClosestElement')
    PT.update_child(fs_node, "ClosestEltGnum", "DataArray_t", value=closest_parent)


def find_closest_element(src_dist_tree: CGNSDistTree,
                         tgt_dist_tree: CGNSDistTree,
                         location: str,
                         comm: MPIComm,
                         **options) -> None:
  """
  Distributed wrapping of find_closest_element, relying on light
  partitioning for surfacic mesh
  """
  all_clouds = [[get_point_cloud(zone, comm, location)] for zone in PT.iter_all_Zone_t(tgt_dist_tree)]

  surf_per_doms_part = dict()
  for zone_path in PT.predicates_to_paths(src_dist_tree, 'CGNSBase_t/Zone_t'):
    zone = PT.find_node_from_path(src_dist_tree, zone_path)
    assert PT.Zone.CellDimension(zone) <= 2
    if PT.pred.IS_POLY2D_ZONE(zone):
      face_vtx_idx, face_vtx, coords, face_gnum, vtx_gnum = minimal_partitioning_poly2D(zone, comm)
    else:
      face_vtx_idx, face_vtx, coords, face_gnum, vtx_gnum = minimal_partitioning(zone, comm)
    pzone = PT.new_Zone(f"{PT.get_name(zone)}.P{comm.rank}.N0", type='Unstructured')
    PT.set_value(pzone, [[vtx_gnum.size, face_gnum.size, 0]])
    PT.new_GridCoordinates(fields={f'Coordinate{d}': coords[i::3] for i,d in enumerate('XYZ')},
                           parent=pzone)
    MT.new_GlobalNumbering({'Vertex' : vtx_gnum, 'Cell' : face_gnum}, parent=pzone)
    if PT.Zone.CellDimension(zone) == 2:
      PT.new_NGonElements(erange=[1, face_gnum.size], eso=face_vtx_idx, ec=face_vtx, parent=pzone)
    else:
      PT.new_Elements(type='BAR_2', erange=[1, face_gnum.size], econn=face_vtx, parent=pzone)
    surf_per_doms_part[zone_path] = [pzone]
      
  periodicities = options.get('periodicities', dict())
  results = pclosest_elt.dist_surf_cloud_compute(surf_per_doms_part, all_clouds, periodicities, comm)

  dom_list = '\n'.join(surf_per_doms_part.keys())
  for dom_results, zone in zip(results, PT.get_all_Zone_t(tgt_dist_tree)):
    result = dom_results[0]
    # Retrieve location
    fs_node = pclosest_elt._create_output_container(zone, location, 'ClosestElement')
    for key, val in result.items():
      PT.update_child(fs_node, key, 'DataArray_t', val)
    PT.new_Descriptor("DomainList", dom_list, parent=fs_node)



def find_closest_boundary(src_dist_tree: CGNSDistTree,
                          src_tgt_tree: CGNSDistTree,
                          location: str,
                          comm: MPIComm,
                          surf_predicate: PT.pred.NodePredicate = IS_BND,
                          perio: bool = True) -> None:
  periodicities = detect_perio(src_dist_tree, comm) if perio else dict()
  bnd_tree = extract_surf_from_bc(src_dist_tree, surf_predicate, comm)
  find_closest_element(bnd_tree,
                       src_tgt_tree,
                       location,
                       comm,
                       periodicities=periodicities)

  update_closest_to_parent(bnd_tree, src_tgt_tree, comm)



def find_closest_boundary_propagation(dist_tree: CGNSDistTree,
                                      comm: MPIComm,
                                      surf_predicate: PT.pred.NodePredicate = IS_BND) -> None:

  if len(PT.get_all_Zone_t(dist_tree)) > 1:
    raise NotImplementedError("Wall_distance computation with method 'propagation' does not support multiple domains")

  bnd_tree = extract_surf_from_bc(dist_tree, surf_predicate, comm)

  # Light partitioning for bnd_tree (should be poly 2D)
  bnd_zone = PT.find_node_from_label(bnd_tree, 'Zone_t')
  assert PT.Zone.CellDimension(bnd_zone) == 2 and PT.pred.IS_POLY2D_ZONE(bnd_zone)
  face_vtx_idx, face_vtx, coords, face_gnum, vtx_gnum = minimal_partitioning_poly2D(bnd_zone, comm)
  pbnd_zone = PT.new_Zone(f"{PT.get_name(bnd_zone)}.P{comm.rank}.N0", type='Unstructured')
  PT.set_value(pbnd_zone, [[vtx_gnum.size, face_gnum.size, 0]])
  PT.new_GridCoordinates(fields={f'Coordinate{d}': coords[i::3] for i,d in enumerate('XYZ')},
                         parent=pbnd_zone)
  MT.new_GlobalNumbering({'Vertex' : vtx_gnum, 'Cell' : face_gnum}, parent=pbnd_zone)
  PT.new_NGonElements(erange=[1, face_gnum.size], eso=face_vtx_idx, ec=face_vtx, parent=pbnd_zone)
  
  # Light partitioning for volumic zone
  vol_zone = PT.find_node_from_label(dist_tree, 'Zone_t')
  cell_face_idx,  cell_face, face_vtx_idx, face_vtx, coords, \
    cell_gnum, face_gnum, vtx_gnum = minimal_partitioning(vol_zone, comm)
  pvol_zone = PT.new_Zone(f"{PT.get_name(vol_zone)}.P{comm.rank}.N0", type='Unstructured', size=[[vtx_gnum.size, cell_gnum.size, 0]])
  PT.new_GridCoordinates(fields={f'Coordinate{d}': coords[i::3] for i,d in enumerate('XYZ')},
                         parent=pvol_zone)
  ng = PT.new_NGonElements(erange=[1, face_gnum.size], eso=face_vtx_idx, ec=face_vtx, parent=pvol_zone)
  MT.new_GlobalNumbering({'Element' : face_gnum}, parent=ng)
  PT.new_NFaceElements(erange=[face_gnum.size+1, face_gnum.size+cell_gnum.size], eso=cell_face_idx, ec=cell_face, parent=pvol_zone)
  MT.new_GlobalNumbering({'Vertex' : vtx_gnum, 'Cell' : cell_gnum}, parent=pvol_zone)

  fields = pclosest_elt.dist_cell_center_surf_compute({'SingleDom' : [pbnd_zone]},
                                                      [pvol_zone],
                                                      comm)[0]
  
  fs_node = pclosest_elt._create_output_container(vol_zone, 'CellCenter', 'ClosestElement')
  for key, val in fields.items():
    PT.update_child(fs_node, key, 'DataArray_t', val)
  dom_path = PT.get_name(PT.find_child_from_label(dist_tree, 'CGNSBase_t')) + '/' +  PT.get_name(vol_zone)
  PT.new_Descriptor("DomainList", dom_path, parent=fs_node)

  update_closest_to_parent(bnd_tree, dist_tree, comm)
