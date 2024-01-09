import copy, time
import mpi4py.MPI as MPI

import maia
import maia.pytree as PT
from   maia.factory  import dist_from_part
from   maia.factory.partitioning.split_S.part_zone import compute_face_gnum
from   maia.algo.dist.s_to_u import compute_transform_matrix, apply_transform_matrix, guess_bnd_normal_index
from   maia.utils import s_numbering
from   .extraction_utils   import local_pl_offset, LOC_TO_DIM, DIMM_TO_DIMF, build_intersection_numbering
from   maia import npy_pdm_gnum_dtype as pdm_gnum_dtype

import numpy as np


def exchange_field_one_domain(part_tree, extract_zones, mesh_dim, etb, container_name, comm) :

  loc_correspondance = {'Vertex'    : 'Vertex',
                        'FaceCenter': 'Cell',
                        'CellCenter': 'Cell'}

  # > Retrieve fields name + GridLocation + PointList if container
  #   is not know by every partition
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm).values()
  assert len(part_tree_per_dom) == 1
  part_zones=list(part_tree_per_dom)[0]
  
  mask_zone = ['MaskedZone', None, [], 'Zone_t']
  dist_from_part.discover_nodes_from_matching(mask_zone, part_zones, container_name, comm, \
      child_list=['GridLocation', 'BCRegionName', 'GridConnectivityRegionName'])
  
  fields_query = lambda n: PT.get_label(n) in ['DataArray_t', 'IndexRange_t']
  dist_from_part.discover_nodes_from_matching(mask_zone, part_zones, [container_name, fields_query], comm)
  mask_container = PT.get_child_from_name(mask_zone, container_name)
  if mask_container is None:
    raise ValueError("[maia-extract_part] asked container for exchange is not in tree")
  if PT.get_child_from_label(mask_container, 'DataArray_t') is None:
    return

  # > Manage BC and GC ZSR
  ref_zsr_node    = mask_container
  bc_descriptor_n = PT.get_child_from_name(mask_container, 'BCRegionName')
  gc_descriptor_n = PT.get_child_from_name(mask_container, 'GridConnectivityRegionName')
  assert not (bc_descriptor_n and gc_descriptor_n)
  if bc_descriptor_n is not None:
    bc_name      = PT.get_value(bc_descriptor_n)
    dist_from_part.discover_nodes_from_matching(mask_zone, part_zones, ['ZoneBC_t', bc_name], comm, child_list=['PointRange', 'GridLocation_t'])
    ref_zsr_node = PT.get_child_from_predicates(mask_zone, f'ZoneBC_t/{bc_name}')
  elif gc_descriptor_n is not None:
    gc_name      = PT.get_value(gc_descriptor_n)
    dist_from_part.discover_nodes_from_matching(mask_zone, part_zones, ['ZoneGridConnectivity_t', gc_name], comm, child_list=['PointRange', 'GridLocation_t'])
    ref_zsr_node = PT.get_child_from_predicates(mask_zone, f'ZoneGridConnectivity_t/{gc_name})')

  grid_location = PT.Subset.GridLocation(ref_zsr_node)
  partial_field = PT.get_child_from_name(ref_zsr_node, 'PointRange') is not None
  assert grid_location in ['Vertex', 'IFaceCenter', 'JFaceCenter', 'KFaceCenter', 'CellCenter']

  parent_lnum_path = {'Vertex'     :'parent_lnum_vtx',
                      'IFaceCenter':'parent_lnum_cell',
                      'JFaceCenter':'parent_lnum_cell',
                      'KFaceCenter':'parent_lnum_cell',
                      'CellCenter' :'parent_lnum_cell'}

  if partial_field:
    part1_pr, part1_gnum1, part1_in_part2 = build_intersection_numbering(part_tree, extract_zones, mesh_dim, container_name, grid_location, etb, comm)

  for i_zone, extract_zone in enumerate(extract_zones):
    
    zone_name = PT.get_name(extract_zone)
    part_zone = PT.get_node_from_name_and_label(part_tree, zone_name, 'Zone_t')

    if partial_field and part1_gnum1[i_zone].size==0:
      continue # Pass if no recovering

    if PT.get_label(mask_container) == 'FlowSolution_t':
      FS_ep = PT.new_FlowSolution(container_name, loc=DIMM_TO_DIMF[mesh_dim][grid_location], parent=extract_zone)
    elif PT.get_label(mask_container) == 'ZoneSubRegion_t':
      FS_ep = PT.new_ZoneSubRegion(container_name, loc=DIMM_TO_DIMF[mesh_dim][grid_location], parent=extract_zone)
    else:
      raise TypeError

    # Add partial numbering to node
    if partial_field:
      pr_n = PT.new_PointRange(value=part1_pr[i_zone], parent=FS_ep)
      gn_n = PT.maia.newGlobalNumbering({'Index' : part1_gnum1[i_zone]}, parent=FS_ep)

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


def exchange_field_s(part_tree, extract_tree, mesh_dim, etb, container_names, comm) :
  # Get zones by domains (only one domain for now)
  extract_part_tree_per_dom = dist_from_part.get_parts_per_blocks(extract_tree, comm)
  for container_name in container_names:
    for i_domain, dom_ep_part_zones in enumerate(extract_part_tree_per_dom.items()):
      dom_path        = dom_ep_part_zones[0]
      extracted_zones = dom_ep_part_zones[1]
      exchange_field_one_domain(part_tree, extracted_zones, mesh_dim, etb[dom_path], container_name, comm)


def extract_part_one_domain_s(part_zones, point_range, dim, comm, equilibrate=False):
  extract_zones = list()
  lvtx_gn = list()
  lcell_gn = list()
  extract_pr_min_per_pzone_l = dict()

  for i_part, part_zone in enumerate(part_zones):
    pr = point_range[i_part]
    if pr.size!=0:
      extract_pr_min_per_pzone_l[PT.get_name(part_zone)] = np.array([min(pr[0,0],pr[0,1]) - 1,
                                                                     min(pr[1,0],pr[1,1]) - 1,
                                                                     min(pr[2,0],pr[2,1]) - 1])
  extract_pr_min_per_pzone_l = comm.allgather(extract_pr_min_per_pzone_l)
  extract_pr_min_per_pzone_all = {k: v for d in extract_pr_min_per_pzone_l for k, v in d.items()}

  etb = dict()

  for i_part, part_zone in enumerate(part_zones):
    zone_name = PT.get_name(part_zone)
    zone_dim  = PT.get_value(part_zone)
    pr = copy.deepcopy(point_range[i_part])

    extract_zone = PT.new_Zone(zone_name, type='Structured', size=np.zeros((3,3), dtype=np.int32))
    
    if pr.size==0:
      lvtx_gn.append(np.empty(0, dtype=pdm_gnum_dtype))
      lcell_gn.append(np.empty(0, dtype=pdm_gnum_dtype))
      extract_zones.append(extract_zone)
      continue

    n_dim_pop = 0
    size_per_dim = np.diff(pr)[:,0]
    mask = np.ones(3, dtype=bool)
    if dim<3:
      idx = np.where(size_per_dim==0)[0]
      mask[idx] = False
      n_dim_pop = idx.size
    if dim>0:
      pr[mask,1]+=1
      size_per_dim+=1

    # n_dim_pop = 0
    extract_zone_dim = np.zeros((3-n_dim_pop,3), dtype=np.int32)
    extract_zone_dim[:,0] = size_per_dim[mask]+1 # size_per_dim[mask]+1
    extract_zone_dim[:,1] = size_per_dim[mask]   # size_per_dim[mask]
    PT.set_value(extract_zone, extract_zone_dim)

    # > Get coordinates
    cx, cy, cz = PT.Zone.coordinates(part_zone)
    extract_cx = cx[pr[0,0]-1:pr[0,1], pr[1,0]-1:pr[1,1], pr[2,0]-1:pr[2,1]]
    extract_cy = cy[pr[0,0]-1:pr[0,1], pr[1,0]-1:pr[1,1], pr[2,0]-1:pr[2,1]]
    extract_cz = cz[pr[0,0]-1:pr[0,1], pr[1,0]-1:pr[1,1], pr[2,0]-1:pr[2,1]]
    extract_cx = np.reshape(extract_cx, size_per_dim[mask]+1)
    extract_cy = np.reshape(extract_cy, size_per_dim[mask]+1)
    extract_cz = np.reshape(extract_cz, size_per_dim[mask]+1)
    PT.new_GridCoordinates(fields={'CoordinateX':extract_cx,
                                   'CoordinateY':extract_cy,
                                   'CoordinateZ':extract_cz},
                           parent=extract_zone)

    # > Set GlobalNumbering
    vtx_per_dir  = zone_dim[:,0]
    cell_per_dir = zone_dim[:,1]

    gn = PT.get_child_from_name(part_zone, ':CGNS#GlobalNumbering')
    gn_vtx  = PT.get_value(PT.get_node_from_name(gn, 'Vertex'))
    gn_face = PT.get_value(PT.get_node_from_name(gn, 'Face'))
    gn_cell = PT.get_value(PT.get_node_from_name(gn, 'Cell'))

    i_ar_cell = np.arange(min(pr[0]), max(pr[0]))
    j_ar_cell = np.arange(min(pr[1]), max(pr[1])).reshape(-1,1)
    k_ar_cell = np.arange(min(pr[2]), max(pr[2])).reshape(-1,1,1)

    if dim<3:
      DIM_TO_LOC = {0:'Vertex', 1:'EdgeCenter', 2:'FaceCenter', 3:'CellCenter'}
      ijk_to_faceIndex = [s_numbering.ijk_to_faceiIndex, s_numbering.ijk_to_facejIndex, s_numbering.ijk_to_facekIndex]

      extract_dir = maia.algo.dist.s_to_u.guess_bnd_normal_index(pr, DIM_TO_LOC[dim])

      locnum_cell = ijk_to_faceIndex[extract_dir](i_ar_cell, j_ar_cell, k_ar_cell, \
                            cell_per_dir, vtx_per_dir).flatten()
      lcell_gn.append(gn_face[locnum_cell-1])
    else:
      locnum_cell = s_numbering.ijk_to_index(i_ar_cell, j_ar_cell, k_ar_cell, cell_per_dir).flatten()
      lcell_gn.append(gn_cell[locnum_cell-1])

    i_ar_vtx = np.arange(min(pr[0]), max(pr[0])+1)
    j_ar_vtx = np.arange(min(pr[1]), max(pr[1])+1).reshape(-1,1)
    k_ar_vtx = np.arange(min(pr[2]), max(pr[2])+1).reshape(-1,1,1)
    locnum_vtx = s_numbering.ijk_to_index(i_ar_vtx, j_ar_vtx, k_ar_vtx, vtx_per_dir).flatten()
    lvtx_gn.append(gn_vtx[locnum_vtx - 1])

    etb[zone_name] = {'parent_lnum_vtx' :locnum_vtx,
                      'parent_lnum_cell':locnum_cell}

    # > Get joins without post-treating PRs
    for zgc_n in PT.get_children_from_label(part_zone, 'ZoneGridConnectivity_t'):
      extract_zgc = PT.new_ZoneGridConnectivity(PT.get_name(zgc_n), parent=extract_zone)
      for gc_n in PT.get_children_from_predicate(zgc_n, lambda n: PT.maia.conv.is_intra_gc(PT.get_name(n))):
        gc_pr = PT.get_value(PT.get_child_from_name(gc_n,"PointRange"))
        intersection = maia.factory.partitioning.split_S.part_zone.intersect_pr(gc_pr, pr)
        if intersection is not None:

          pr_of_gc  = PT.get_value(PT.get_child_from_name(gc_n, 'PointRange'))
          prd_of_gc = PT.get_value(PT.get_child_from_name(gc_n, 'PointRangeDonor'))
          

          transform = PT.get_value(PT.get_node_from_predicates(gc_n, 'Transform'))
          T = compute_transform_matrix(transform)
          apply_t1 = apply_transform_matrix(intersection[:,0], pr_of_gc[:,0], prd_of_gc[:,0], T)
          apply_t2 = apply_transform_matrix(intersection[:,1], pr_of_gc[:,0], prd_of_gc[:,0], T)
          new_gc_prd = [[apply_t1[dim], apply_t2[dim]] for dim in range(3)]
          
          # > Update joins PRs
          min_cur = extract_pr_min_per_pzone_all[zone_name]
          try:
              min_opp = extract_pr_min_per_pzone_all[PT.get_value(gc_n)]
          except KeyError:
              min_opp = None
          
          new_gc_pr = gc_pr
          new_gc_pr[0,:] -= min_cur[0]
          new_gc_pr[1,:] -= min_cur[1]
          new_gc_pr[2,:] -= min_cur[2]

          if min_opp is not None:
            new_gc_pr[0,:] -= min_opp[0]
            new_gc_pr[1,:] -= min_opp[1]
            new_gc_pr[2,:] -= min_opp[2]
          if dim<3:
            new_gc_pr  = np.delete(new_gc_pr , extract_dir, 0)
            new_gc_prd = np.delete(new_gc_prd, transform[extract_dir]-1, 0)

          gc_name = PT.get_name(gc_n)
          gc_donorname = PT.get_value(gc_n)
          extract_gc_n = PT.new_GridConnectivity1to1(gc_name, donor_name=gc_donorname, point_range=new_gc_pr, point_range_donor=new_gc_prd, parent=extract_zgc)

    extract_zones.append(extract_zone)

  # > Create GlobalNumbering
  partial_gnum_vtx  = maia.algo.part.point_cloud_utils.create_sub_numbering(lvtx_gn, comm)
  partial_gnum_cell = maia.algo.part.point_cloud_utils.create_sub_numbering(lcell_gn, comm)
  if  dim==3:
    cell_size = dist_from_part._recover_dist_block_size(extract_zones, comm)

  if len(partial_gnum_vtx)!=0:
    for i_part, extract_zone in enumerate(extract_zones):
      PT.maia.newGlobalNumbering({'Vertex' : partial_gnum_vtx [i_part],
                                  'Cell'   : partial_gnum_cell[i_part]},
                                 parent=extract_zone)
      
      # > Retrive missing gnum if 3d
      if  dim==3 and PT.Zone.n_cell(extract_zone)!=0:
        cell_ijk   = s_numbering.index_to_ijk(partial_gnum_cell[i_part], cell_size[:,1])
        cell_range = np.array([[min(cell_ijk[0]),max(cell_ijk[0])],
                               [min(cell_ijk[1]),max(cell_ijk[1])],
                               [min(cell_ijk[2]),max(cell_ijk[2])]])
        cell_window = cell_range
        cell_window[:,1] +=1

        dist_cell_per_dir = cell_size[:,1]
        face_lntogn = compute_face_gnum(dist_cell_per_dir, cell_window)
        
        gn_node = PT.maia.getGlobalNumbering(extract_zone)
        PT.new_node("CellRange", "IndexRange_t", cell_range, parent=gn_node)
        PT.new_DataArray("CellSize", cell_size[:,1], parent=gn_node)
        PT.new_DataArray("Face", face_lntogn, parent=gn_node)
  else:
    assert len(partial_gnum_cell)==0

  return extract_zones,etb

