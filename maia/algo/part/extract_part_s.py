import maia
import maia.pytree      as PT
import maia.pytree.maia as MT
from   maia.factory  import dist_from_part
from   maia.factory.partitioning.split_S.part_zone import compute_face_gnum, compute_edge_gnum
from   maia.utils import s_numbering
from   .extraction_utils   import DIMM_TO_DIMF, build_intersection_numbering, discover_containers
from   maia import npy_pdm_gnum_dtype as pdm_gnum_dtype

import numpy as np


parent_lnum_path = {'Vertex'     :'parent_lnum_vtx',
                    'CellCenter' :'parent_lnum_cell'}

def exchange_field_one_domain(part_tree, extract_zones, mesh_dim, etb, container_name, comm) :

  src_dim, tgt_dim = mesh_dim
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm).values()
  assert len(part_tree_per_dom) == 1
  part_zones=list(part_tree_per_dom)[0]

  # > Retrieve fields name + GridLocation + PointRange if container
  #   is not know by every partition
  mask_container, grid_location, partial_field = discover_containers(part_zones, container_name, 'PointRange', 'IndexRange_t', comm)
  if mask_container is None:
    return
  if src_dim == 2:
    assert grid_location in ['Vertex', 'IEdgeCenter', 'JEdgeCenter', 'CellCenter']
  else:
    assert grid_location in ['Vertex', 'IFaceCenter', 'JFaceCenter', 'KFaceCenter', 'CellCenter']
  out_grid_location = DIMM_TO_DIMF[tgt_dim][grid_location]

  if partial_field:
    part1_pr, part1_gnum1, part1_in_part2 = build_intersection_numbering(part_tree, extract_zones, mesh_dim, container_name, grid_location, etb, comm)

  for i_zone, extract_zone in enumerate(extract_zones):
    
    zone_name = PT.get_name(extract_zone)
    part_zone = PT.get_node_from_name_and_label(part_tree, zone_name, 'Zone_t')
    is_own_data = etb['ExtractingCnt'] == container_name

    if partial_field and part1_gnum1[i_zone].size==0:
      continue # Pass if no recovering

    if (mask_label := PT.get_label(mask_container)) in ['FlowSolution_t', 'DiscreteData_t']:
      FS_ep = PT.new_FlowSolution(container_name, loc=out_grid_location, parent=extract_zone)
      PT.set_label(FS_ep, mask_label)
    elif PT.get_label(mask_container) == 'ZoneSubRegion_t':
      FS_ep = PT.new_ZoneSubRegion(container_name, loc=out_grid_location, parent=extract_zone)
    else:
      raise TypeError

    # Add partial numbering to node
    if partial_field:
      if tgt_dim < src_dim:
        # If output zone is 2D, we need to remove the useless direction in output PR
        extract_dir = etb['@@maia_extract_direction@@']
        part1_pr[i_zone] = np.delete(part1_pr[i_zone], extract_dir, axis=0)
      if is_own_data and out_grid_location in ['CellCenter', 'Vertex']:
        # For owndata, output a FlowSolution without PR instead of keep a ZoneSubRegion
        zsize = PT.Zone.CellSize(extract_zone) if out_grid_location == 'CellCenter' else \
                PT.Zone.VertexSize(extract_zone)
        assert (part1_pr[i_zone][:,0] == 1).all() and (part1_pr[i_zone][:,1] == zsize).all()
        PT.set_label(FS_ep, 'FlowSolution_t')
      else:
        PT.new_IndexRange(value=part1_pr[i_zone], parent=FS_ep)
        MT.new_GlobalNumbering({'Index' : part1_gnum1[i_zone]}, parent=FS_ep)

    for fld_node in PT.get_children_from_label(mask_container, 'DataArray_t'):
      fld_name = PT.get_name(fld_node)
      fld_path = f"{container_name}/{fld_name}"

      fld_data = PT.get_value(PT.get_node_from_path(part_zone,fld_path))

      if partial_field:
        extract_fld_data = fld_data.flatten(order='F')[part1_in_part2[i_zone]]
        if PT.get_label(FS_ep)=='FlowSolution_t':
          extract_fld_data = extract_fld_data.reshape(np.diff(part1_pr[i_zone])[:,0]+1, order='F')
      else:
        parent_part1_pl = etb[zone_name][parent_lnum_path[grid_location]]
        extract_fld_data = fld_data.flatten(order='F')[parent_part1_pl-1]
        if PT.get_label(FS_ep)=='FlowSolution_t':
          zone_elt_dim = PT.Zone.VertexSize(extract_zone) if grid_location=='Vertex' else PT.Zone.CellSize(extract_zone)
          extract_fld_data = extract_fld_data.reshape(zone_elt_dim, order='F')

      PT.new_DataArray(fld_name, extract_fld_data, parent=FS_ep)


def exchange_field_s(part_tree, extract_tree, dims, etb, containers_name, comm) :
  # Get zones by domains (only one domain for now)
  extract_part_tree_per_dom = dist_from_part.get_parts_per_blocks(extract_tree, comm)
  for container_name in containers_name:
    for i_domain, dom_ep_part_zones in enumerate(extract_part_tree_per_dom.items()):
      dom_path        = dom_ep_part_zones[0]
      extracted_zones = dom_ep_part_zones[1]
      exchange_field_one_domain(part_tree, extracted_zones, dims, etb[dom_path], container_name, comm)


def extract_part_one_domain_s(part_zones, point_range, dims, location, comm):
  extract_zones = list()
  lvtx_gn = list()
  lcell_gn = list()
  extract_pr_min_per_pzone_l = dict()

  for i_part, part_zone in enumerate(part_zones):
    pr = point_range[i_part]
    if pr.size!=0:
      extract_pr_min_per_pzone_l[PT.get_name(part_zone)] = np.minimum(pr[:,0], pr[:,1]) - 1
  extract_pr_min_per_pzone_l = comm.allgather(extract_pr_min_per_pzone_l)
  extract_pr_min_per_pzone_all = {k: v for d in extract_pr_min_per_pzone_l for k, v in d.items()}

  etb = dict()

  src_dim, tgt_dim = dims

  for i_part, part_zone in enumerate(part_zones):
    zone_name = PT.get_name(part_zone)
    zone_dim  = PT.get_value(part_zone)
    pr = np.copy(point_range[i_part])

    extract_zone = PT.new_Zone(zone_name, type='Structured', size=np.zeros((3,3), dtype=np.int32))
    
    if pr.size==0:
      lvtx_gn.append(np.empty(0, dtype=pdm_gnum_dtype))
      lcell_gn.append(np.empty(0, dtype=pdm_gnum_dtype))
      extract_zones.append(extract_zone)
      continue

    size_per_dim = np.diff(pr)[:,0]
    mask = np.ones(src_dim, dtype=bool)
    if location=='Vertex':
      idx = np.where(size_per_dim==0)[0]
      mask[idx] = False
      n_dim_pop = idx.size
      if n_dim_pop in [2,3]:
        raise NotImplementedError(f'Asked extraction is 0D or 1D (n_dim_pop={n_dim_pop})')
      extract_dir = idx[0]
    elif 'FaceCenter' in location or 'EdgeCenter' in location:
      extract_dir = PT.Subset.normal_axis(PT.new_BC(point_range=pr, loc=location))
      mask[extract_dir] = False
      n_dim_pop = 1
    else:
      n_dim_pop = 0
    if location!='Vertex':
      pr[mask,1]+=1
      size_per_dim+=1

    if location != 'CellCenter':
      etb['@@maia_extract_direction@@'] = extract_dir
    # n_dim_pop = 0
    extract_zone_dim = np.zeros((src_dim-n_dim_pop,3), dtype=np.int32)
    extract_zone_dim[:,0] = size_per_dim[mask]+1 # size_per_dim[mask]+1
    extract_zone_dim[:,1] = size_per_dim[mask]   # size_per_dim[mask]
    PT.set_value(extract_zone, extract_zone_dim)

    # > Get coordinates
    cx, cy, cz = PT.Zone.coordinates(part_zone)
    slice_x = slice(pr[0,0]-1, pr[0,1])
    slice_y = slice(pr[1,0]-1, pr[1,1])
    if src_dim == 3:
      slice_z = slice(pr[2,0]-1, pr[2,1])
      extract_cx = cx[slice_x, slice_y, slice_z]
      extract_cy = cy[slice_x, slice_y, slice_z]
      extract_cz = cz[slice_x, slice_y, slice_z]
    else:
      extract_cx = cx[slice_x, slice_y]
      extract_cy = cy[slice_x, slice_y]
      extract_cz = cz[slice_x, slice_y] if cz is not None else None

    extract_cx = np.reshape(extract_cx, size_per_dim[mask]+1)
    extract_cy = np.reshape(extract_cy, size_per_dim[mask]+1)
    if cz is not None:
      extract_cz = np.reshape(extract_cz, size_per_dim[mask]+1)
    PT.new_GridCoordinates(fields={'CoordinateX':extract_cx,
                                   'CoordinateY':extract_cy,
                                   'CoordinateZ':extract_cz}, # Cz will be removed after if None
                           parent=extract_zone)

    # > Set GlobalNumbering
    vtx_per_dir  = zone_dim[:,0]
    cell_per_dir = zone_dim[:,1]

    keys = ['Vertex', 'Edge', 'Cell'] if src_dim == 2 else ['Vertex', 'Face', 'Cell']
    gn_entities = {key: MT.globalnumbering_value(part_zone, key) for key in keys}

    _pr = pr.copy()
    if n_dim_pop > 0:
      # If we are extracting a 2D mesh, pr (stored in vertices) has one constant line [1,1]
      # When generating cell_range, we need to add one, otherwise we have an empty range
      _pr[extract_dir, 1] = max(pr[extract_dir]) + 1
      _pr[extract_dir, 0] = min(pr[extract_dir])
    i_ar_cell = np.arange(min(_pr[0]), max(_pr[0]))
    j_ar_cell = np.arange(min(_pr[1]), max(_pr[1])).reshape(-1,1)
    if src_dim > 2:
      k_ar_cell = np.arange(min(_pr[2]), max(_pr[2])).reshape(-1,1,1)

    # Get the parent gnum of Cell or Face/Edge depending of n_dim_pop
    # (if n_dim_pop=1, we are extracting faces from 3D or edges from 2D)
    # The local (idx_from_loc) to acces gnum array must be computed from the relevant location
    face_or_edge = 'Edge' if src_dim == 2 else 'Face'
    _entity = face_or_edge if n_dim_pop == 1 else 'Cell'
    entity = 'IJK'[extract_dir] + f'{face_or_edge}Center' if _entity == face_or_edge else 'CellCenter'
    if src_dim == 2:
      locnum_cell = s_numbering.ij_to_index_from_loc(i_ar_cell, j_ar_cell, entity, vtx_per_dir).flatten()
    else:
      locnum_cell = s_numbering.ijk_to_index_from_loc(i_ar_cell, j_ar_cell, k_ar_cell, entity, vtx_per_dir).flatten()

    lcell_gn.append(gn_entities[_entity][locnum_cell-1])

    i_ar_vtx = np.arange(min(pr[0]), max(pr[0])+1)
    j_ar_vtx = np.arange(min(pr[1]), max(pr[1])+1).reshape(-1,1)
    if src_dim > 2:
      k_ar_vtx = np.arange(min(pr[2]), max(pr[2])+1).reshape(-1,1,1)
    if src_dim == 2:
      locnum_vtx = s_numbering.ij_to_index_from_loc(i_ar_vtx, j_ar_vtx, 'Vertex', vtx_per_dir).flatten()
    else:
      locnum_vtx = s_numbering.ijk_to_index_from_loc(i_ar_vtx, j_ar_vtx, k_ar_vtx, 'Vertex', vtx_per_dir).flatten()
    lvtx_gn.append(gn_entities['Vertex'][locnum_vtx-1])

    etb[zone_name] = {'parent_lnum_vtx' :locnum_vtx,
                      'parent_lnum_cell':locnum_cell}

    # > Get joins without post-treating PRs
    for zgc_n in PT.get_children_from_label(part_zone, 'ZoneGridConnectivity_t'):
      extract_zgc = PT.new_ZoneGridConnectivity(PT.get_name(zgc_n), parent=extract_zone)
      for gc_n in PT.get_children_from_predicate(zgc_n, MT.pred.is_gc_of_kind(is_intra=True)):
        gc_pr = PT.get_value(PT.get_child_from_name(gc_n,"PointRange"))
        intersection = maia.factory.partitioning.split_S.part_zone.intersect_pr(gc_pr, pr)
        if intersection is not None:

          transform = PT.GridConnectivity.Transform(gc_n)
          new_gc_prd = PT.utils.gc_transform_window(gc_n, intersection)
          
          # > Update joins PRs
          min_cur = extract_pr_min_per_pzone_all[zone_name]
          try:
              min_opp = extract_pr_min_per_pzone_all[PT.get_value(gc_n)]
          except KeyError:
              min_opp = None
          
          new_gc_pr = np.copy(intersection)
          new_gc_pr -= min_cur.reshape((-1,1))

          if min_opp is not None:
            new_gc_prd -= min_opp.reshape((-1,1))
            
          if tgt_dim < src_dim:
            new_gc_pr  = np.delete(new_gc_pr , extract_dir, 0)
            new_gc_prd = np.delete(new_gc_prd, transform[extract_dir]-1, 0)

          gc_name = PT.get_name(gc_n)
          gc_donorname = PT.get_value(gc_n)
          extract_gc_n = PT.new_GridConnectivity1to1(gc_name, donor_name=gc_donorname, point_range=new_gc_pr, point_range_donor=new_gc_prd, parent=extract_zgc)

    extract_zones.append(extract_zone)

  # > Create GlobalNumbering
  partial_gnum_vtx  = maia.algo.part.point_cloud_utils.create_sub_numbering(lvtx_gn, comm)
  partial_gnum_cell = maia.algo.part.point_cloud_utils.create_sub_numbering(lcell_gn, comm)
  if src_dim == tgt_dim:
    cell_size = dist_from_part._recover_dist_block_size(extract_zones, comm)

  if len(partial_gnum_vtx)!=0:
    for i_part, extract_zone in enumerate(extract_zones):
      MT.new_GlobalNumbering({'Vertex' : partial_gnum_vtx [i_part],
                              'Cell'   : partial_gnum_cell[i_part]}, parent=extract_zone)
      
      # > Retrive missing gnum if 3d
      if tgt_dim==src_dim and PT.Zone.n_cell(extract_zone)!=0:
        if src_dim == 2:
          cell_ij    = s_numbering.index_to_ij(partial_gnum_cell[i_part], cell_size[:,1])
          cell_range = np.array([[min(cell_ij[0]),max(cell_ij[0])],
                                 [min(cell_ij[1]),max(cell_ij[1])]])
        else:
          cell_ijk   = s_numbering.index_to_ijk(partial_gnum_cell[i_part], cell_size[:,1])
          cell_range = np.array([[min(cell_ijk[0]),max(cell_ijk[0])],
                                 [min(cell_ijk[1]),max(cell_ijk[1])],
                                 [min(cell_ijk[2]),max(cell_ijk[2])]])
        cell_window = cell_range
        cell_window[:,1] +=1

        dist_cell_per_dir = cell_size[:,1]

        gn_node = MT.find_GlobalNumbering(extract_zone)
        PT.new_DataArray("CellRange", cell_range, parent=gn_node)
        PT.new_DataArray("CellSize", cell_size[:,1], parent=gn_node)
        if src_dim == 2:
          PT.new_DataArray('Edge', compute_edge_gnum(dist_cell_per_dir, cell_window), parent=gn_node)
        else:
          PT.new_DataArray('Face', compute_face_gnum(dist_cell_per_dir, cell_window), parent=gn_node)

  else:
    assert len(partial_gnum_cell)==0

  return extract_zones,etb

