from mpi4py import MPI
import numpy as np
import operator

import maia
import maia.pytree        as PT
import maia.pytree.maia   as MT

from   maia.factory                      import dist_from_part
from   maia.factory.partitioning.split_U import pdm_part_to_cgns_zone
from   maia.transfer                     import utils as TEU
from   maia.utils                        import np_utils, layouts
from   maia.utils                        import logging as mlog

from   .extraction_utils   import local_pl_offset, LOC_TO_DIM, DIMM_TO_DIMF,\
                                  get_partial_container_stride_and_order, discover_containers
from   .point_cloud_utils  import create_sub_numbering

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype

import Pypdm.Pypdm as PDM

# ExtractPart API changed between PDM2.6 and PDM2.7 (see !153), this switch allow to use good API
EP_OLD_API = hasattr(PDM.ExtractPart, 'extract_part_group_get')
PDM_EP_group_set   = PDM.ExtractPart.part_group_set         if EP_OLD_API else PDM.ExtractPart.group_set
PDM_EP_group_get   = PDM.ExtractPart.extract_part_group_get if EP_OLD_API else PDM.ExtractPart.group_get
PDM_EP_n_group_set = PDM.ExtractPart.part_n_group_set       if EP_OLD_API else PDM.ExtractPart.n_group_set

def _generate_entity_graph_comm(entity_gnum_l, comm, key):
  # Simplified version for manifold interfaces, waiting for
  # PDM.generate_entity_graph_comm wrapping
  from maia.transfer import protocols as EP
  from maia.utils import par_utils
  from maia.utils import vstride as vs

  n_part = len(entity_gnum_l)
  pn_l = [gn.size for gn in entity_gnum_l]
  stride_one_l = [np.ones(pn, np.int32) for pn in pn_l]
  # Avoid managing i_rank + i_part with a combinated 'rankpart' id
  part_distri = par_utils.dn_to_distribution(n_part, comm)
  part_distri_f = par_utils.partial_to_full_distribution(part_distri, comm)
  lid_l = [np.arange(1,pn+1, dtype=np.int32) for pn in pn_l]
  rank_l = [(part_distri[0]+i_part) * np.ones(pn, np.int32) for i_part, pn in enumerate(pn_l)]
  distri = par_utils.distribution_from_gnum(entity_gnum_l, comm, full=True)

  GI = EP.GlobalIndexer(distri, entity_gnum_l, comm, gnum_offset=1)
  gathered_lid  = GI.Put_v(list(zip(stride_one_l, lid_l)),  extend=True)
  gathered_rank = GI.Put_v(list(zip(stride_one_l, rank_l)), extend=True)

  # Put_v / Take_v pattern brings back rank and lid of entity sharing the same gnum on
  # other ranks
  lid  = [vs.from_counts(*data) for data in GI.Take_v(gathered_lid)]
  rank = [vs.from_counts(*data) for data in GI.Take_v(gathered_rank)]
  # Then we have to
  # - select duplicated entity
  # - remove self entity
  # - sort according to opp. rank order

  for i_part in range(n_part):
    is_jn = np.where(lid[i_part].counts > 1)[0]
    lid[i_part]  = vs.take(lid[i_part], is_jn)
    rank[i_part] = vs.take(rank[i_part], is_jn)

  non_manifold = any((lid[i_part].counts != 2).any() for i_part in range(n_part))
  if comm.allreduce(non_manifold, MPI.LOR):
    mlog.warning("Skip internal JNs reconstrution because extracted mesh is non-manifold")
    return [{f'np_{key}_part_bound_part_idx' : np.zeros(1, np.int32),
             f'np_{key}_part_bound' : np.empty(0, np.int32)} for i_part in range(n_part)]
  
  all_result = list()
  for i_part in range(n_part):
    rankpart = part_distri[0] + i_part
    opp_mask = rank[i_part].values != rankpart
    opp_lid = lid[i_part].values[opp_mask]
    opp_rank = rank[i_part].values[opp_mask]
    own_lid = lid[i_part].values[~opp_mask]
  
    # To ensure PL/PLD symmetry, we have to sort by rank, part and then by lid or opp_lid
    # within each rank (but we must use same lid for 2 sides of the join)
    # This is done using sorting_lid which is identical on both side + lexsort
    # (last key is used first)
    sorting_lid = (rankpart < opp_rank)*own_lid + (opp_rank < rankpart)*opp_lid
    sort_idx = np.lexsort([sorting_lid, opp_rank])
    opp_rank = opp_rank[sort_idx]
    _, counts = np_utils.unique_sorted(opp_rank, True)

    # Split opp_rank (agglomated) into rank + part with binsearch
    _opp_rank = np.searchsorted(part_distri_f, opp_rank, side='right') - 1
    _opp_part = opp_rank - part_distri_f[_opp_rank]

    np_part_bound_part_idx = np_utils.sizes_to_indices(counts)
    np_part_bound = np.empty(4*counts.sum(), np.int32)
    np_part_bound[0::4] = own_lid[sort_idx]
    np_part_bound[1::4] = _opp_rank #Already sorted
    np_part_bound[2::4] = _opp_part + 1 # Already sorted but must start at 1
    np_part_bound[3::4] = opp_lid[sort_idx]

    all_result.append({f'np_{key}_part_bound_part_idx' : np_part_bound_part_idx,
                       f'np_{key}_part_bound' : np_part_bound})
  return all_result

def exchange_field_one_domain_loc(part_zones, extract_zones, mesh_dim, exch_tool_box, container_name, comm):
  _grid_location    = {"Vertex" : "Vertex", "FaceCenter" : "Element", "CellCenter" : "Cell"}
  assert len(extract_zones) <= len(part_zones)

  partial_gnum = list()
  is_own_data = exch_tool_box['ExtractingCnt'] == container_name

  extract_zones_iter = iter(extract_zones)

  for i_part, part_zone in enumerate(part_zones):
    # Since empty extracted zones are not added in extracted tree, we do not
    # have len(part_zones) == len(extract_zones). We need to 'consume' the next
    # extracted zone only if it is not empty
    # On the other side i_part stills include empty extracted zones
    if exch_tool_box['parent_elt']['Vertex'][i_part].size == 0:
      continue # Extracted zone was empty

    extr_zone = next(extract_zones_iter) # Consume extr. zone

    container = PT.get_child_from_name(part_zone, container_name)
    if container is None:
      continue # Volumic zone has no fields


    grid_location = PT.Subset.GridLocation(container)
    assert grid_location in ['Vertex', 'FaceCenter', 'CellCenter']

    # > FlowSolution node def by zone
    if (mask_label := PT.get_label(container)) in ['FlowSolution_t', 'DiscreteData_t']:
      FS_ep = PT.new_FlowSolution(container_name, loc=DIMM_TO_DIMF[mesh_dim][grid_location], parent=extr_zone)
      PT.set_label(FS_ep, mask_label)
      pl_container = container
    elif PT.get_label(container) == 'ZoneSubRegion_t':
      FS_ep = PT.new_ZoneSubRegion(container_name, loc=DIMM_TO_DIMF[mesh_dim][grid_location], parent=extr_zone)
      pl_container = PT.find_node_from_path(part_zone, PT.Subset.ZSRExtent(container, part_zone))
    else:
      raise TypeError

    is_partial = PT.get_child_from_name(pl_container, 'PointList') is not None

    parent = exch_tool_box['parent_elt'][grid_location][i_part]

    elt_n = part_zone if grid_location != 'FaceCenter' else PT.Zone.NGonNode(part_zone)
    base_gnum = MT.globalnumbering_value(elt_n, _grid_location[grid_location])

    if is_partial:
      # If volumic container is partial, we need to retrieve the position of parent entity (given in gnum)
      # in the volumic pointlist (lnum)
      # We can do this with searchsorted if we convert the point_list (vol) in gnum before
      point_list_n = PT.find_node_from_name(pl_container, 'PointList')
      point_list   = PT.get_np_value(point_list_n)[0] - local_pl_offset(part_zone, LOC_TO_DIM[grid_location]) # Gnum start at 1

      point_list_gnum = base_gnum[point_list-1]
      
      sorter  = np.argsort(point_list_gnum)
      idx_tmp = np.searchsorted(point_list_gnum, parent, sorter=sorter)

      # Careful ! searchsorted always return a result, even if parent is not in
      # point_list_gnum which can happens when the volumic data is partial
      mask = np.take(point_list_gnum, idx_tmp, mode='clip') == parent
      idx = idx_tmp[mask]

      # Create PointList if input field is partial
      # NB : if the container *is* the one we are extracting from, then
      # extracted field will be full -> transform into FS
      if is_own_data:
        assert mask.all()
        assert PT.Subset.GridLocation(FS_ep) in ['CellCenter', 'Vertex']
        PT.set_label(FS_ep, 'FlowSolution_t')
      else:
        _extr_pl = np.where(mask)[0]
        extr_pl = _extr_pl + local_pl_offset(extr_zone, LOC_TO_DIM[grid_location]) + 1
        PT.new_IndexArray('PointList', value=extr_pl.reshape((1,-1), order='F'), parent=FS_ep)

        # To create gnum associated with PointList
        elt_n_ext = extr_zone if grid_location != 'FaceCenter' else PT.Zone.NGonNode(extr_zone)
        base_gnum_ext = MT.globalnumbering_value(elt_n_ext, _grid_location[grid_location])
        partial_gnum.append(base_gnum_ext[_extr_pl])

    else:
      # If volumic container is full, there is no pointlist indirection, but out parent entity is
      # still in gnum so we need to retrieve the local num too
      sorter = np.argsort(base_gnum)
      idx    = np.searchsorted(base_gnum, parent, sorter=sorter)
      
    # Extract fields and place in extracted container
    for field in PT.get_children_from_label(container, 'DataArray_t'):
      PT.new_DataArray(PT.get_name(field), PT.get_np_value(field)[idx], parent=FS_ep)

  # Update global numbering in extracted FS (only in partial case w/o is_own_data)
  # Again partial_gnum and extract_zones can have different len,
  # if somes zones have been skipped (no data on volumic zone)
  # so we need an index to get sub gnum only for with-field
  # extracted zones
  if not is_own_data:
    idx_read = 0
    sub_partial_gnum = create_sub_numbering(partial_gnum, comm)
    for extr_zone in extract_zones:
      if (FS_ep := PT.get_child_from_name(extr_zone, container_name)) is not None:
        if (pl:=PT.get_child_from_name(FS_ep, 'PointList')) is not None:
          MT.new_GlobalNumbering({'Index' : sub_partial_gnum[idx_read]}, FS_ep)
          idx_read += 1
          # Do cleaning at same time (remove container if PL is empty)
          if PT.Subset.n_elem(FS_ep) == 0:
            PT.rm_child(extr_zone, FS_ep)

def exchange_field_one_domain_req(part_zones, extract_zones, mesh_dim, exch_tool_box, container_name, comm):
  # > Retrieve fields name + GridLocation + PointList if container is not know by every partition
  mask_container, grid_location, partial_field = discover_containers(part_zones, container_name, 'PointList', 'IndexArray_t', comm)
  if mask_container is None:
    return
  assert grid_location in ['Vertex', 'FaceCenter', 'CellCenter']

  # When reequilibrate, each rank have at most one extracted zone
  extract_zone = extract_zones[0] if len(extract_zones) > 0 else None

  # > FlowSolution node def by zone
  if extract_zone is not None :
    if (mask_label := PT.get_label(mask_container)) in ['FlowSolution_t', 'DiscreteData_t']:
      FS_ep = PT.new_FlowSolution(container_name, loc=DIMM_TO_DIMF[mesh_dim][grid_location], parent=extract_zone)
      PT.set_label(FS_ep, mask_label)
    elif PT.get_label(mask_container) == 'ZoneSubRegion_t':
      FS_ep = PT.new_ZoneSubRegion(container_name, loc=DIMM_TO_DIMF[mesh_dim][grid_location], parent=extract_zone)
    else:
      raise TypeError
  

  # > Get PTP and parentElement for the good location
  ptp         = exch_tool_box['part_to_part'][grid_location]
  is_own_data = exch_tool_box['ExtractingCnt'] == container_name
  
  # LN_TO_GN
  _grid_location    = {"Vertex" : "Vertex", "FaceCenter" : "Element", "CellCenter" : "Cell"}
  
  if extract_zone is not None:
    elt_n            = extract_zone if grid_location!='FaceCenter' else PT.Zone.NGonNode(extract_zone)
    if elt_n is None :return
    part1_ln_to_gn   = [MT.globalnumbering_value(elt_n, _grid_location[grid_location])]

  # Get reordering informations if point_list
  # https://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array
  if partial_field:
    pl_gnum1, stride = get_partial_container_stride_and_order(part_zones, container_name, grid_location, ptp, comm)

  # > Field exchange
  for fld_node in PT.get_children_from_label(mask_container, 'DataArray_t'):
    fld_name = PT.get_name(fld_node)
    fld_path = f"{container_name}/{fld_name}"
    fld_dtype = PT.get_np_value(fld_node).dtype
    
    if partial_field:
      # Get field and organize it according to the gnum1_come_from arrays order
      fld_data = list()
      for i_part, part_zone in enumerate(part_zones) :
        fld_n = PT.get_node_from_path(part_zone,fld_path)
        fld_data_tmp = PT.get_np_value(fld_n) if fld_n is not None else np.empty(0, dtype=fld_dtype)
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

    # Interpolation and placement
    if extract_zone is not None:
      i_part = 0
      if part1_data[i_part].size!=0:
        PT.new_DataArray(fld_name, part1_data[i_part], parent=FS_ep)
  
  # Build PL with the last exchange stride
  if partial_field:
    if len(part1_data)!=0 and part1_data[0].size!=0:
      new_point_list = np.where(part1_stride[0]==1)[0] if part1_data[0].size!=0 else np.empty(0, dtype=np.int32)
      point_list = new_point_list + local_pl_offset(extract_zone, LOC_TO_DIM[grid_location])+1
      PT.new_IndexArray(name='PointList', value=point_list.reshape((1,-1), order='F'), parent=FS_ep)
      partial_part1_lngn = [part1_ln_to_gn[0][new_point_list]]
    else:
      partial_part1_lngn = []

    # Update global numbering in FS
    partial_gnum = create_sub_numbering(partial_part1_lngn, comm)
    if extract_zone is not None and len(partial_gnum)!=0:
      if is_own_data and PT.Subset.GridLocation(FS_ep) in ['CellCenter', 'Vertex']:
        # For owndata, output a FlowSolution without PL instead of keep a ZoneSubRegion
        assert (new_point_list == np.arange(point_list.size)).all()
        PT.set_label(FS_ep, 'FlowSolution_t')
        PT.rm_children_from_name(FS_ep, 'PointList')
      else:
        MT.new_GlobalNumbering({'Index' : partial_gnum[0]}, parent=FS_ep)

  if part1_data[0].size==0 and extract_zone is not None:
    PT.rm_child(extract_zone, FS_ep)

def exchange_field_one_domain(part_zones, extract_zones, mesh_dim, exch_tool_box, container_name, comm):
  equilibrate = len(exch_tool_box['part_to_part']) > 0
  if equilibrate:
    exchange_field_one_domain_req(part_zones, extract_zones, mesh_dim, exch_tool_box, container_name, comm)
  else:
    exchange_field_one_domain_loc(part_zones, extract_zones, mesh_dim, exch_tool_box, container_name, comm)


def exchange_field_u(part_tree, extract_part_tree, mesh_dim, exch_tool_box, container_names, comm) :
  # Get zones by domains (only one domain for now)
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm)

  # Get zone(s) from extractpart
  extract_zones = PT.get_all_Zone_t(extract_part_tree)

  for container_name in container_names:
    for dom_path, part_zones in part_tree_per_dom.items():
      exchange_field_one_domain(part_zones, extract_zones, mesh_dim, exch_tool_box[dom_path], \
          container_name, comm)


def extract_part_one_domain_u(part_zones, point_list, location, comm,
                              equilibrate=True,
                              graph_part_tool="hilbert"):
  """
  Prepare PDM extract_part object and perform the extraction of one domain.
  """
  dim = LOC_TO_DIM[location]

  n_part_in  = len(part_zones)
  n_part_out = 1 if equilibrate else n_part_in

  # In local mode, 'native' groups (eg face groups if we extract faces) are not yet supported by PDM
  # so we exclude them from set / get by using < instead of <= in bc parsing
  bc_op = operator.le if equilibrate else operator.lt
  
  kind = PDM._PDM_EXTRACT_PART_KIND_REEQUILIBRATE if equilibrate else PDM._PDM_EXTRACT_PART_KIND_LOCAL
  pdm_ep = PDM.ExtractPart(dim, # face/cells
                           n_part_in,
                           n_part_out,
                           kind,
                           eval(f"PDM._PDM_SPLIT_DUAL_WITH_{graph_part_tool.upper()}"),
                           True,
                           comm)

  # > Discover BCs
  dist_zone = PT.new_Zone('Zone')
  gdom_bcs_path_per_dim = {"CellCenter":None, "FaceCenter":None, "EdgeCenter":None, "Vertex":None}
  child_list = ['GridLocation', 'FamilyName_t', 'AdditionalFamilyName_t', 'Descriptor_t']
  for bc_type, dim_name in enumerate(gdom_bcs_path_per_dim):
    if bc_op(LOC_TO_DIM[dim_name], dim):
      is_dim_bc = PT.pred.is_bc_of_location(dim_name)
      dist_from_part.discover_nodes_from_matching(dist_zone, part_zones, ["ZoneBC_t", is_dim_bc], comm, child_list=child_list, get_value='leaf')
      gdom_bcs_path_per_dim[dim_name] = PT.predicates_to_paths(dist_zone, ['ZoneBC_t',is_dim_bc])
      n_gdom_bcs = len(gdom_bcs_path_per_dim[dim_name])
      PDM_EP_n_group_set(pdm_ep, bc_type+1, n_gdom_bcs)

  # Loop over domain zone : preparing extract part
  for i_part, part_zone in enumerate(part_zones):
    # Get NGon + NFac
    cx, cy, cz = PT.Zone.coordinates(part_zone)
    vtx_coords = np_utils.interweave_arrays([cx,cy,cz])
    
    ngon  = PT.Zone.NGonNode(part_zone)
    nface = PT.Zone.NFaceNode(part_zone)

    cell_face_idx = PT.get_child_from_name(nface, "ElementStartOffset" )[1]
    cell_face     = PT.get_child_from_name(nface, "ElementConnectivity")[1]
    face_vtx_idx  = PT.get_child_from_name(ngon,  "ElementStartOffset" )[1]
    face_vtx      = PT.get_child_from_name(ngon,  "ElementConnectivity")[1]

    vtx_ln_to_gn, _, face_ln_to_gn, cell_ln_to_gn = TEU.get_entities_numbering(part_zone)

    if EP_OLD_API:
      pdm_ep.part_set(i_part,
                      cell_ln_to_gn.shape[0], face_ln_to_gn.shape[0], 0, vtx_ln_to_gn.shape[0],
                      cell_face_idx, cell_face    ,
                      None, None, None,
                      face_vtx_idx , face_vtx     ,
                      cell_ln_to_gn, face_ln_to_gn,
                      None,
                      vtx_ln_to_gn , vtx_coords)
    else:
      pdm_ep.part_set(i_part,
                      cell_face_idx, cell_face    ,
                      None, None, None,
                      face_vtx_idx , face_vtx     ,
                      cell_ln_to_gn, face_ln_to_gn,
                      None,
                      vtx_ln_to_gn , vtx_coords)

    pdm_ep.selected_lnum_set(i_part, point_list[i_part][0] - local_pl_offset(part_zone, dim))


    # Add BCs info
    bc_type = 1
    for dim_name, gdom_bcs_path in gdom_bcs_path_per_dim.items():
      if bc_op(LOC_TO_DIM[dim_name], dim):
        for i_bc, bc_path in enumerate(gdom_bcs_path):
          bc_n  = PT.get_node_from_path(part_zone, bc_path)
          bc_pl = PT.get_value(PT.get_child_from_name(bc_n, 'PointList'))[0] \
                    if bc_n is not None else np.empty(0, np.int32)
          bc_gn = MT.globalnumbering_value(bc_n, 'Index') if bc_n is not None else np.empty(0, pdm_gnum_dtype)
          PDM_EP_group_set(pdm_ep, i_part, i_bc, bc_type, bc_pl-local_pl_offset(part_zone, LOC_TO_DIM[dim_name]) , bc_gn)
      bc_type +=1

  pdm_ep.compute()

  # > Compute edge data here (this is a global operation)
  # In addition we can not do a double get so we store some extracted data
  all_ep_vtx_ln_to_gn  = [pdm_ep.ln_to_gn_get(i_part,PDM._PDM_MESH_ENTITY_VTX)   for i_part in range(n_part_out)]
  if dim >= 2:
    all_ep_face_ln_to_gn = [pdm_ep.ln_to_gn_get(i_part, PDM._PDM_MESH_ENTITY_FACE) for i_part in range(n_part_out)]
    all_ep_face_vtx = [pdm_ep.connectivity_get(i_part, PDM._PDM_CONNECTIVITY_TYPE_FACE_VTX) for i_part in range(n_part_out)]
    if dim == 2:
      all_edge_data = PDM.compute_face_edge_from_face_vtx(comm, 
                                                          [t.size for t in all_ep_face_ln_to_gn],
                                                          [t.size for t in all_ep_vtx_ln_to_gn], 
                                                          [face_vtx[0] for face_vtx in all_ep_face_vtx], 
                                                          [face_vtx[1] for face_vtx in all_ep_face_vtx], 
                                                          all_ep_face_ln_to_gn,
                                                          all_ep_vtx_ln_to_gn)

  # > Reconstruction du maillage de l'extract part
  extract_zones = []
  for i_part in range(n_part_out):
    n_extract_cell = pdm_ep.n_entity_get(i_part, PDM._PDM_MESH_ENTITY_CELL)
    n_extract_face = pdm_ep.n_entity_get(i_part, PDM._PDM_MESH_ENTITY_FACE)
    n_extract_edge = pdm_ep.n_entity_get(i_part, PDM._PDM_MESH_ENTITY_EDGE)
    n_extract_vtx  = pdm_ep.n_entity_get(i_part, PDM._PDM_MESH_ENTITY_VTX )
    
    size_by_dim = {0: [[n_extract_vtx, 0             , 0]], # not yet implemented
                  1:   None                              , # not yet implemented
                  2: [[n_extract_vtx, n_extract_face, 0]],
                  3: [[n_extract_vtx, n_extract_cell, 0]] }

    # > ExtractPart zone construction
    extract_zone = PT.new_Zone(MT.conv.add_part_suffix('Zone', comm.Get_rank(), i_part),
                                size=size_by_dim[dim],
                                type='Unstructured')

    ep_vtx_ln_to_gn  = all_ep_vtx_ln_to_gn[i_part]
    MT.new_GlobalNumbering({"Vertex" : ep_vtx_ln_to_gn}, parent=extract_zone)

    # > Grid coordinates
    cx, cy, cz = layouts.interlaced_to_tuple_coords(pdm_ep.vtx_coord_get(i_part))
    extract_grid_coord = PT.new_GridCoordinates(parent=extract_zone)
    PT.new_DataArray('CoordinateX', cx, parent=extract_grid_coord)
    PT.new_DataArray('CoordinateY', cy, parent=extract_grid_coord)
    PT.new_DataArray('CoordinateZ', cz, parent=extract_grid_coord)

    if dim == 0:
      MT.new_GlobalNumbering({'Cell' : np.empty(0, dtype=ep_vtx_ln_to_gn.dtype)}, parent=extract_zone)

    # > NGON
    if dim >= 2:
      ep_face_vtx_idx, ep_face_vtx  = all_ep_face_vtx[i_part]
      ep_face_ln_to_gn = all_ep_face_ln_to_gn[i_part]

      nb_bar = 0
      if dim == 2:
        # Retrieve edges on 2D mesh
        edge_data = all_edge_data[i_part]

        nb_bar = edge_data['np_edge_ln_to_gn'].size
        bar_n = PT.new_Elements('EdgeElements', 'BAR_2', 
                                erange=[1, nb_bar], 
                                econn=edge_data['np_edge_vtx'], 
                                parent=extract_zone)
        MT.new_GlobalNumbering({'Element' : edge_data['np_edge_ln_to_gn']}, parent=bar_n)

      ngon_n = PT.new_NGonElements('NGonElements',
                                  erange  = [nb_bar+1, nb_bar+n_extract_face],
                                  ec      = ep_face_vtx,
                                  eso     = ep_face_vtx_idx,
                                  parent  = extract_zone)

      MT.new_GlobalNumbering({'Element' : ep_face_ln_to_gn}, parent=ngon_n)
      if dim == 2:
        MT.new_GlobalNumbering({'Cell' : ep_face_ln_to_gn}, parent=extract_zone)

    # > NFACES
    if dim == 3:
      ep_cell_face_idx, ep_cell_face = pdm_ep.connectivity_get(i_part, PDM._PDM_CONNECTIVITY_TYPE_CELL_FACE)
      nface_n = PT.new_NFaceElements('NFaceElements',
                                      erange  = [n_extract_face+1, n_extract_face+n_extract_cell],
                                      ec      = ep_cell_face,
                                      eso     = ep_cell_face_idx,
                                      parent  = extract_zone)

      ep_cell_ln_to_gn = pdm_ep.ln_to_gn_get(i_part, PDM._PDM_MESH_ENTITY_CELL)
      MT.new_GlobalNumbering({'Element' : ep_cell_ln_to_gn}, parent=nface_n)
      MT.new_GlobalNumbering({'Cell' : ep_cell_ln_to_gn}, parent=extract_zone)

      maia.algo.nface_to_pe(extract_zone, comm)

    # - Get BCs
    zonebc_n = PT.new_ZoneBC(parent=extract_zone)
    bc_type = 1
    for dim_name, gdom_bcs_path in gdom_bcs_path_per_dim.items():
      if bc_op(LOC_TO_DIM[dim_name], dim):
        for i_bc, bc_path in enumerate(gdom_bcs_path):
          bc_info = PDM_EP_group_get(pdm_ep, i_part, i_bc, bc_type)
          bc_pl = bc_info['group_entity']
          bc_gn = bc_info['group_entity_ln_to_gn']
          if bc_pl.size != 0:
            dist_bc = PT.get_node_from_path(dist_zone, bc_path)
            bc_name = bc_path.split('/')[-1]
            bc_val = PT.get_value(dist_bc) if PT.get_value(dist_bc) is not None else 'Null'
            bc_loc = 'CellCenter' if (dim_name == 'FaceCenter' and dim == 2) else dim_name
            if bc_loc == 'CellCenter' and dim == 2: # Offset BCs, because we put Edge elts first
              bc_pl += nb_bar
            bc_n = PT.new_BC(bc_name, bc_val, point_list=bc_pl.reshape((1,-1), order='F'), loc=bc_loc, parent=zonebc_n)
            for child in PT.get_children_from_predicate(dist_bc, ~PT.pred.name_is('GridLocation')):
              PT.add_child(bc_n, child)
            MT.new_GlobalNumbering({'Index':bc_gn}, parent=bc_n)
      bc_type +=1 

    extract_zones.append(extract_zone)

  # - Generate intrazones jns
  if dim >= 2:
    if dim == 2:
      data_l = _generate_entity_graph_comm([edge_data['np_edge_ln_to_gn'] for edge_data in all_edge_data], comm, 'edge')
    elif dim ==3:
      data_l = _generate_entity_graph_comm(all_ep_face_ln_to_gn, comm, 'face')

    for extr_zone, data in zip(extract_zones, data_l):
      pdm_part_to_cgns_zone.zgc_created_pdm_to_cgns(extr_zone, None, None, data, 'FaceCenter')
      zgc_n = PT.find_child_from_label(extr_zone, 'ZoneGridConnectivity_t')
      if len(PT.get_children(zgc_n)) == 0:
        PT.rm_child(extr_zone, zgc_n)

  # - Get PTP by vertex and cell
  ptp = dict()
  if equilibrate:
    ptp['Vertex']       = pdm_ep.part_to_part_get(PDM._PDM_MESH_ENTITY_VTX)
    if dim >= 2: # NGON
      ptp['FaceCenter'] = pdm_ep.part_to_part_get(PDM._PDM_MESH_ENTITY_FACE)
    if dim == 3: # NFACE
      ptp['CellCenter'] = pdm_ep.part_to_part_get(PDM._PDM_MESH_ENTITY_CELL)
    
  # - Get parent elt
  parent_elt = dict()
  parent_elt['Vertex']       = [pdm_ep.parent_ln_to_gn_get(i_part,PDM._PDM_MESH_ENTITY_VTX) for i_part in range(n_part_out)]
  if dim >= 2: # NGON
    parent_elt['FaceCenter'] = [pdm_ep.parent_ln_to_gn_get(i_part,PDM._PDM_MESH_ENTITY_FACE) for i_part in range(n_part_out)]
  if dim == 3: # NFACE
    parent_elt['CellCenter'] = [pdm_ep.parent_ln_to_gn_get(i_part,PDM._PDM_MESH_ENTITY_CELL) for i_part in range(n_part_out)]
  
  exch_tool_box = {'part_to_part' : ptp, 'parent_elt' : parent_elt}

  return extract_zones, exch_tool_box


