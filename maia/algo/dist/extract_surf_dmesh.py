import numpy          as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils                import np_utils, par_utils
from maia.transfer.dist_to_part.index_exchange import collect_distributed_pl

from maia.transfer import protocols as EP
from maia.utils import vstride as vs

from .ngon_tools import cgns_connectivity_from_vs
from .connectivity_utils import entity_vtx_connectivity_elt
from .s_to_u import bc_s_to_bc_u, add_lowerdim_std_elements

from maia.typing import *

def extract_surf_from_bc_single(zone:CGNSDistTree,
                                bc_predicate:Callable[[CGNSTree], bool],
                                comm: MPIComm) -> CGNSDistTree:

  ztype = PT.get_np_value(zone).dtype
  zone_dim = PT.Zone.CellDimension(zone)
  wanted_loc = 'EdgeCenter' if zone_dim == 2 else 'FaceCenter'
  is_relevant_bc = PT.pred.NodePredicate(bc_predicate)

  # If input zone is structured => we create a unst. zone with only relevant
  # bcs + lowerdim elements as std elements
  s_parent_l = list()
  if PT.Zone.Type(zone) == 'Structured':
    u_size = np.prod(PT.get_np_value(zone), axis=0, dtype=ztype).reshape(1,-1)
    u_zone = PT.new_Zone(PT.get_name(zone), type='Unstructured', size=u_size)

    zonebc = PT.new_ZoneBC(parent=u_zone)
    for bc_s in PT.iter_children_from_predicates(zone, [PT.pred.label_is('ZoneBC_t'), is_relevant_bc]):
      # Work on a copy w/ BCDS
      shallow_bc = PT.new_BC(PT.get_name(bc_s), type=PT.get_str_value(bc_s), loc=PT.Subset.GridLocation(bc_s))
      PT.add_child(shallow_bc, PT.Subset.getPatch(bc_s))
      bc_u = bc_s_to_bc_u(shallow_bc, PT.Zone.VertexSize(zone), wanted_loc, comm.rank, comm.size)
      # > Get absolute gnum in all faces numbering for structured meshes
      s_parent_l.append(PT.get_np_value(PT.find_child_from_name(bc_u, 'PointList')))
      PT.add_child(zonebc, bc_u)

    elt = PT.new_Elements('DummyElt', erange=[0], parent=u_zone) # Trick to avoid offset of created element
    add_lowerdim_std_elements(u_zone, PT.Zone.VertexSize(zone), zone_dim, comm)
    PT.rm_child(u_zone, elt)
  else:
    u_zone = zone

  # Get selected faces or edge ids
  pred = [PT.pred.label_is('ZoneBC_t'), is_relevant_bc & PT.pred.has_location(wanted_loc)]
  selected_face_ids = collect_distributed_pl(u_zone, [pred])
  selected_distri = [MT.Subset.distribution(s) for s in PT.get_children_from_predicates(u_zone, pred)]


  # Extract face_vtx (or edge_vtx) connectivity --> depends on input connectivity, but in all
  # cases we output as poly mesh
  # Note : for now make parent start at 1 to be consistent with partitioned version.
  # This may change in the future (see #232)
  if PT.pred.IS_POLY3D_ZONE(u_zone):
    # We have input face_vtx --> get flagged faces and extract connectivity
    selected_face_ids = [ids[0] for ids in selected_face_ids]
    ngon_n = PT.Zone.NGonNode(u_zone)
    face_distri = MT.Element.distribution(ngon_n) 
    face_vtx = MT.Element.connectivity(ngon_n)
    GI = EP.GlobalIndexer(face_distri, selected_face_ids, comm, gnum_offset=PT.Element.Range(ngon_n)[0])
    flagged = np.flatnonzero(GI.access_counts > 0)
    parent = flagged.astype(ztype, copy=False) + face_distri[0] + 1
    ext_face_vtx = vs.take(face_vtx, flagged)
  else: # Std elts or Poly2D (in which case EdgeElements are defined)
    # Std elements *or* S : face extraction is managed direcly by entity_vtx_connectivity_elt
    # (note : this may not work if a face appears twice in selected_face_ids)
    # Here parent is simply the flagged id since connectivity is get following this order
    # but we first do a redistribution step to be // independant and to ensure better distribution
    
    start = 0
    loc_start = 0
    distri_tot = par_utils.uniform_distribution(sum(d[2] for d in selected_distri), comm)
    _parent = np.empty(distri_tot[1]-distri_tot[0], ztype)
    if PT.Zone.Type(zone) == 'Structured':
      parent = np.empty(distri_tot[1]-distri_tot[0], ztype)
    for j,ids in enumerate(selected_face_ids):
      distri_in  = selected_distri[j]
      distri_out = distri_in.copy()
      end = start + distri_in[2]
      distri_out[0] = max(min(distri_tot[0], end), start) - start
      distri_out[1] = max(min(distri_tot[1], end), start) - start
      btb = EP.BlockToBlock(distri_in, distri_out, comm)
      loc_end = loc_start + distri_out[1] - distri_out[0]
      btb.exchange_inplace(ids[0], _parent[loc_start:loc_end])
      if PT.Zone.Type(zone) == 'Structured':
        btb.exchange_inplace(s_parent_l[j][0], parent[loc_start:loc_end])
      start = end
      loc_start = loc_end

    ext_face_vtx = entity_vtx_connectivity_elt(u_zone, comm, zone_dim-1, False, _parent)

    if PT.Zone.Type(zone) != 'Structured':
      parent = _parent
      parent_shift = PT.Zone.get_elt_range_per_dim(u_zone)[zone_dim-1][0] - 1
      parent -= parent_shift

  ext_face_distri = par_utils.dn_to_distribution(len(ext_face_vtx), comm)

  # Flag selected vertices, renumber it and extract coordinates
  coords = PT.Zone.coordinates(zone)
  vtx_distri  = MT.Zone.vtx_distribution(zone)

  GI = EP.GlobalIndexer(vtx_distri, ext_face_vtx.values-1, comm)
  flagged_vtx = GI.access_counts > 0
  ext_vtx_distri = par_utils.dn_to_distribution(flagged_vtx.sum(), comm)
  new_vtx_id = np.empty(flagged_vtx.size, ext_face_vtx.dtype)
  new_vtx_id[flagged_vtx] = np.arange(ext_vtx_distri[0]+1, ext_vtx_distri[1]+1, dtype=new_vtx_id.dtype)
  ext_face_vtx._values = GI.Take(new_vtx_id) # Update using renumbered values
  ext_coords = {key: val[flagged_vtx] for key,val in coords._asdict().items() if val is not None}

  # Create extracted zone
  ext_size = np.array([[ext_vtx_distri[2], ext_face_distri[2], 0]], dtype=PT.get_np_value(zone).dtype)
  ext_zone = PT.new_Zone(PT.get_name(zone), type='Unstructured', size=ext_size)

  # Compute new vtx id
  PT.new_GridCoordinates(fields=ext_coords, parent=ext_zone)
  erange=np.array([1, ext_face_distri[2]], ext_face_vtx.dtype)
  if zone_dim == 3:
    eso, ec = cgns_connectivity_from_vs(ext_face_vtx, comm)
    ext_ng = PT.new_NGonElements(erange=erange, eso=eso, ec=ec, parent=ext_zone)
  else:
    ext_ng = PT.new_Elements('BAR', 'BAR_2', erange=erange, econn=ext_face_vtx.values, parent=ext_zone)
  PT.new_DiscreteData(loc='CellCenter', fields={'Parent' : parent}, parent=ext_zone)
  MT.new_Distribution({'Element' : ext_face_distri}, ext_ng)
  MT.new_Distribution({'Cell' : ext_face_distri, 'Vertex' : ext_vtx_distri}, parent=ext_zone)

  return ext_zone


def extract_surf_from_bc(dist_tree: CGNSDistTree, 
                         bc_predicate: Callable[[CGNSTree], bool], 
                         comm: MPIComm) -> CGNSDistTree:
  # Light / local version of extract_part for WallDistance

  ext_tree = PT.new_CGNSTree()
  for base in PT.iter_all_CGNSBase_t(dist_tree):
    ext_base = PT.new_CGNSBase(PT.get_name(base),
                               cell_dim=PT.Base.CellDimension(base)-1,
                               phy_dim=PT.Base.PhysicalDimension(base),
                               parent=ext_tree)
    for zone in PT.iter_all_Zone_t(base):
      PT.add_child(ext_base, extract_surf_from_bc_single(zone, bc_predicate, comm))
  
  return ext_tree
