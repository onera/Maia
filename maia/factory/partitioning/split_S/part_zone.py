from mpi4py import MPI
import numpy              as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia import npy_pdm_gnum_dtype as pdm_dtype
from maia.utils import np_utils, s_numbering
from .                               import split_cut_tree as SCT
from maia.transfer.dist_to_part import data_exchange  as BTP
from maia.algo.dist.s_to_u import guess_bnd_normal_index, \
                                       compute_transform_matrix, \
                                       apply_transform_matrix, \
                                       n_face_per_dir

idx_to_dir = {0:'x', 1:'y', 2:'z'}
dir_to_idx = {'x':0, 'y':1, 'z':2}
min_max_as_int = lambda st : 0 if 'min' in st else 1

def zone_cell_range(zone):
  """ Return the size of a point_range 2d array """
  n_cell = PT.Zone.CellSize(zone)
  zone_range = np.empty((n_cell.shape[0], 2), n_cell.dtype)
  zone_range[:,0] = 1
  zone_range[:,1] = n_cell
  return zone_range

def collect_S_bnd_per_dir(zone):
  """
  Group the bc and 1to1 gc founds in a structured zone according to
  the geometric boundary they belong (xmin, xmax, ymin, ... , zmax).
  Return a dictionnary storing the dist boundaries for
  each geometric boundary.
  """
  base_bound = {k : [] for k in ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]}

  bnd_queries = [['ZoneBC_t', 'BC_t'],
      ['ZoneGridConnectivity_t', lambda n : PT.get_label(n) in ['GridConnectivity1to1_t', 'GridConnectivity_t']],
      ['ZoneBC_t', 'BC_t', 'BCDataSet_t']]
  for bnd_query in bnd_queries:
    for nodes in PT.iter_children_from_predicates(zone, bnd_query, ancestors=True):
      bnd = nodes[-1]
      grid_loc    = PT.Subset.GridLocation(bnd)
      point_range_n = PT.get_node_from_name(bnd, 'PointRange')
      if point_range_n is not None: #Skip BC/GC defined by a PointList -- they will be updated after
        point_range = point_range_n[1]
        bnd_normal_index = guess_bnd_normal_index(point_range, grid_loc)

        if PT.get_label(bnd) == 'BCDataSet_t':
          bcds_path = '/'.join([PT.get_name(n) for n in nodes[:-1]])
          PT.update_child(bnd, '__maia::dspath', 'Descriptor_t', bcds_path)

        pr_val = point_range[bnd_normal_index,0]
        extr = 'min' if pr_val == 1 else 'max'

        base_bound[idx_to_dir[bnd_normal_index] + extr].append(bnd)

  return base_bound

def intersect_pr(prA, prB):
  """
  Compute the intersection of two numpy array of shape(n,2)
  Return None if arrays are disjoints
  """
  assert prA.ndim == 2 and prA.shape[1] == 2
  assert prB.shape == prA.shape

  sub_pr = np.empty_like(prA)
  for d in range(prA.shape[0]):
    if prB[d,0] <= prA[d,0] and prA[d,0] <= prB[d,1]:
      sub_pr[d,0] = prA[d,0]
    elif prA[d,0] <= prB[d,0] and prB[d,0] <= prA[d,1]:
      sub_pr[d,0] = prB[d,0]
    else:
      return None
    sub_pr[d,1] = min(prA[d,1], prB[d,1])

  return sub_pr

def pr_to_cell_location(pr, normal_idx, original_loc, bnd_is_max, reverse=False):
  """
  Shift the input point_range to change its location from original_loc to cells
  (if reverse = False) or from cells to its original_loc (if reverse = True)
  """
  not_normal_idx = np.where(np.arange(pr.shape[0]) != normal_idx)[0]
  sign = -1 if reverse else 1
  if not original_loc == 'CellCenter':
    pr[normal_idx,:] -= sign*int(bnd_is_max)
  if original_loc == 'Vertex':
    pr[not_normal_idx,1] -= sign*1

def pr_to_global_num(pr, zone_offset, reverse=False):
  """
  Shift the input point_range to have it in the global numbering of the dist_zone
  (if reverse = False) or to go back in the local (if reverse = True)
  """
  sign = -1 if reverse else 1
  pr += sign*(zone_offset.reshape((-1,1)) - 1)

def create_bcs(d_zone, p_zone, p_zone_offset):
  """
  """
  zbc = PT.new_ZoneBC(parent=p_zone)
  for geo_bnd, dist_bnds in collect_S_bnd_per_dir(d_zone).items():

    normal_idx = dir_to_idx[geo_bnd[0]]
    extr       = min_max_as_int(geo_bnd) #0 if min, 1 if max

    # Get the part boundary in absolute cell numbering
    range_part_bc_g = zone_cell_range(p_zone)
    range_part_bc_g[normal_idx, 1-extr] = range_part_bc_g[normal_idx, extr]
    pr_to_global_num(range_part_bc_g, p_zone_offset)

    #Check if part boundary is internal or comes from an old BC or GC
    is_old_bc = range_part_bc_g[normal_idx,extr] == zone_cell_range(d_zone)[normal_idx,extr]
    if is_old_bc:
      dirs = np.where(np.arange(range_part_bc_g.shape[0]) != normal_idx)[0]
      for dist_bc in dist_bnds:
        dist_bc_pr_n = PT.get_child_from_name(dist_bc, 'PointRange')
        range_dist_bc = np.copy(dist_bc_pr_n[1])
        grid_loc      = PT.Subset.GridLocation(dist_bc)

        #Swap because some gc are allowed to be reversed and convert to cell
        dir_to_swap     = (range_dist_bc[:,1] < range_dist_bc[:,0])
        range_dist_bc[dir_to_swap, 0], range_dist_bc[dir_to_swap, 1] = \
                range_dist_bc[dir_to_swap, 1], range_dist_bc[dir_to_swap, 0]
        pr_to_cell_location(range_dist_bc, normal_idx, grid_loc, extr)

        inter = intersect_pr(range_part_bc_g[dirs,:], range_dist_bc[dirs,:])
        if inter is not None:
          sub_pr = np.empty((3,2), dtype=np.int32)
          sub_pr[dirs,:] = inter
          sub_pr[normal_idx,:] = range_part_bc_g[normal_idx,:]

          #Move back to local numbering, original location and unswapped
          pr_to_global_num(sub_pr, p_zone_offset, reverse=True)
          pr_to_cell_location(sub_pr, normal_idx, grid_loc, extr, reverse=True)
          sub_pr[dir_to_swap, 0], sub_pr[dir_to_swap, 1] = \
                  sub_pr[dir_to_swap, 1], sub_pr[dir_to_swap, 0]

          #Effective creation of BC in part zone
          if PT.get_label(dist_bc) == 'BCDataSet_t':
            path_node = PT.get_child_from_name(dist_bc, '__maia::dspath')
            parent_path   = PT.get_value(path_node)
            PT.rm_child(dist_bc, path_node) #Cleanup
            #BC should have been created before so its ok
            parent = PT.get_node_from_path(p_zone, parent_path)
            part_bc = PT.new_node(PT.get_name(dist_bc), 'BCDataSet_t', parent=parent)
            PT.new_PointRange(value=sub_pr, parent=part_bc)
            sub_pr_loc = np.copy(sub_pr)
            sub_pr_loc[0,:] += range_part_bc_g[0,0] - range_dist_bc[0,0]
            sub_pr_loc[1,:] += range_part_bc_g[1,0] - range_dist_bc[1,0]
            sub_pr_loc[2,:] += range_part_bc_g[2,0] - range_dist_bc[2,0]
            sub_pr_loc[normal_idx,:] = 1

            i_ar  = np.arange(sub_pr_loc[0,0], sub_pr_loc[0,1]+1, dtype=pdm_dtype)
            j_ar  = np.arange(sub_pr_loc[1,0], sub_pr_loc[1,1]+1, dtype=pdm_dtype).reshape(-1,1)
            k_ar  = np.arange(sub_pr_loc[2,0], sub_pr_loc[2,1]+1, dtype=pdm_dtype).reshape(-1,1,1)
            bcds_lntogn = s_numbering.ijk_to_index(i_ar, j_ar, k_ar, PT.PointRange.SizePerIndex(dist_bc_pr_n)).flatten()
            assert bcds_lntogn.size == PT.Subset.n_elem(part_bc)
            MT.newGlobalNumbering({'Index' : bcds_lntogn}, part_bc)
          else: #GC are put with bc and treated afterward
            part_bc = PT.new_BC(PT.get_name(dist_bc), point_range=sub_pr, parent=zbc)
          PT.set_value(part_bc, PT.get_value(dist_bc))
          PT.new_GridLocation(grid_loc, parent=part_bc)
          PT.add_child(part_bc, PT.get_child_from_name(dist_bc, 'Transform'))
          PT.add_child(part_bc, PT.get_child_from_label(dist_bc, 'GridConnectivityType_t'))
          PT.add_child(part_bc, PT.get_child_from_label(dist_bc, 'GridConnectivityProperty_t'))
          if PT.get_child_from_name(dist_bc, 'GridConnectivityDonorName') is not None:
            PT.add_child(part_bc, PT.get_child_from_name(dist_bc, 'GridConnectivityDonorName'))
            PT.new_child(part_bc, 'distPR', 'IndexRange_t', PT.get_child_from_name(dist_bc, 'PointRange')[1])
            PT.new_child(part_bc, 'distPRDonor', 'IndexRange_t', PT.get_child_from_name(dist_bc, 'PointRangeDonor')[1])
            PT.new_DataArray('zone_offset', p_zone_offset, parent=part_bc)

def create_internal_gcs(d_zone, p_zones, p_zones_offset, comm):
  """
  """
  # 1. Collect : for each partition, select the boundary corresponding to new (internal)
  # joins. We store the PR the boundary in (cell) global numbering
  jn_list = [ [] for i in range(len(p_zones))]
  for geo_bnd in ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]:
    normal_idx = dir_to_idx[geo_bnd[0]]
    extr       = min_max_as_int(geo_bnd) #0 if min, 1 if max
    for i_part, (p_zone, p_zone_offset) in enumerate(zip(p_zones, p_zones_offset)):
      range_part_bc_g = zone_cell_range(p_zone)
      range_part_bc_g[normal_idx, 1-extr] = range_part_bc_g[normal_idx, extr]
      pr_to_global_num(range_part_bc_g, p_zone_offset)
      #Check if part boundary is internal or comes from an old BC or GC
      is_old_bc = range_part_bc_g[normal_idx,extr] == zone_cell_range(d_zone)[normal_idx,extr]
      if not is_old_bc:
        jn_list[i_part].append(PT.new_PointRange(geo_bnd, range_part_bc_g))

  # 2. Exchange. We will need to compare to other parts joins
  all_offset_list = comm.allgather(p_zones_offset)
  all_jn_list     = comm.allgather(jn_list)

  # 3. Process
  for i_part, p_zone in enumerate(p_zones):
    zgc = PT.new_node('ZoneGridConnectivity', 'ZoneGridConnectivity_t', parent=p_zone)
    for jn in jn_list[i_part]:
      # Get data for the current join
      normal_idx = dir_to_idx[PT.get_name(jn)[0]]
      extr       = min_max_as_int(PT.get_name(jn)[1:])
      shift = 1 - 2*extr #1 si min, -1 si max
      dirs  = np.where(np.arange(3) != normal_idx)[0]
      my_pr = PT.get_value(jn)

      # Check opposite joins
      for j_proc, opp_parts in enumerate(all_jn_list):
        for j_part, opp_part in enumerate(opp_parts):
          for opp_jn in opp_part:
            opp_normal_idx = dir_to_idx[PT.get_name(opp_jn)[0]]
            opp_extr       = min_max_as_int(PT.get_name(opp_jn)[1:])
            opp_pr         = PT.get_value(opp_jn)
            is_admissible  = opp_normal_idx == normal_idx \
                             and opp_extr != extr \
                             and my_pr[normal_idx,0] - shift == opp_pr[normal_idx,0]
            if is_admissible:
              inter = intersect_pr(my_pr[dirs,:], opp_pr[dirs,:])
              if inter is not None:
                sub_pr = np.empty((3,2), dtype=np.int32)
                sub_pr[dirs,:] = inter
                sub_pr[normal_idx,:] = my_pr[normal_idx,:]

                sub_pr_d = np.copy(sub_pr)
                sub_pr_d[normal_idx] -= shift
                #Global to local
                pr_to_global_num(sub_pr, p_zones_offset[i_part], reverse=True)
                pr_to_global_num(sub_pr_d, all_offset_list[j_proc][j_part], reverse=True)
                #Restore location
                pr_to_cell_location(sub_pr, normal_idx, 'Vertex', extr, reverse=True)
                pr_to_cell_location(sub_pr_d, normal_idx, 'Vertex', 1-extr, reverse=True)

                #Effective creation of GC in part zone
                gc_name  = MT.conv.name_intra_gc(comm.Get_rank(), i_part, j_proc, j_part)
                opp_zone = MT.conv.add_part_suffix(PT.get_name(d_zone), j_proc, j_part)
                part_gc = PT.new_GridConnectivity1to1(gc_name, opp_zone, transform=[1,2,3], parent=zgc)
                PT.new_PointRange('PointRange',      sub_pr,   parent=part_gc)
                PT.new_PointRange('PointRangeDonor', sub_pr_d, parent=part_gc)

def split_original_joins_S(all_part_zones, comm):
  """
  Intersect the original joins of the meshes. Such joins must are stored in ZoneBC_t node
  at the end of S partitioning with needed information stored inside
  """
  ori_jn_to_pr  = dict()
  zones_offsets = dict()
  for part in all_part_zones:
    dzone_name = MT.conv.get_part_prefix(part[0])
    for jn in PT.iter_children_from_predicates(part, 'ZoneBC_t/BC_t'):
      if PT.get_child_from_name(jn, 'GridConnectivityDonorName') is not None:
        p_zone_offset = PT.get_child_from_name(jn, 'zone_offset')[1]
        pr_n = PT.new_PointRange(part[0], np.copy(PT.get_child_from_name(jn, 'PointRange')[1]))
        key = dzone_name + '/' + jn[0] #TODO : Be carefull if multibase ; this key may clash
        # Pr dans la num globale de la zone
        pr_to_global_num(pr_n[1], p_zone_offset)
        try:
          ori_jn_to_pr[key].append(pr_n)
        except KeyError:
          ori_jn_to_pr[key] = [pr_n]
        zones_offsets[PT.get_name(part)] = p_zone_offset

  #Gather and create dic jn -> List of PR
  ori_jn_to_pr_glob = dict()
  all_offset_zones  = dict()
  for ori_jn_to_pr_rank in comm.allgather(ori_jn_to_pr):
    for key, value in ori_jn_to_pr_rank.items():
      if key in ori_jn_to_pr_glob:
        ori_jn_to_pr_glob[key].extend(value)
      else:
        ori_jn_to_pr_glob[key] = value
  for zones_offsets_rank in comm.allgather(zones_offsets):
    all_offset_zones.update(zones_offsets_rank)

  for part in all_part_zones:
    zone_gc = PT.update_child(part, 'ZoneGridConnectivity', 'ZoneGridConnectivity_t')
    to_delete = []
    for jn in PT.iter_children_from_predicates(part, 'ZoneBC_t/BC_t'):
      if PT.get_child_from_name(jn, 'GridConnectivityDonorName') is not None:
        dist_pr = PT.get_child_from_name(jn, 'distPR')[1]
        dist_prd = PT.get_child_from_name(jn, 'distPRDonor')[1]
        transform  = PT.get_child_from_name(jn, 'Transform')[1]
        T_matrix = compute_transform_matrix(transform)
        assert PT.Subset.GridLocation(jn) == 'Vertex'

        #Jn dans la num globale de la dist_zone
        p_zone_offset = PT.get_child_from_name(jn, 'zone_offset')[1]
        pr = np.copy(PT.get_child_from_name(jn, 'PointRange')[1])
        pr_to_global_num(pr, p_zone_offset)

        #Jn dans la num globale de la dist_zone opposée
        pr_in_opp_abs = np.empty((3,2), dtype=pr.dtype)
        pr_in_opp_abs[:,0] = apply_transform_matrix(pr[:,0], dist_pr[:,0], dist_prd[:,0], T_matrix)
        pr_in_opp_abs[:,1] = apply_transform_matrix(pr[:,1], dist_pr[:,0], dist_prd[:,0], T_matrix)

        #Jn dans la zone opposée et en cellules
        normal_idx = guess_bnd_normal_index(pr_in_opp_abs, 'Vertex')
        dirs       = np.where(np.arange(3) != normal_idx)[0]
        bnd_is_max = pr_in_opp_abs[normal_idx,0] != 1 #Sommets
        dir_to_swap     = (pr_in_opp_abs[:,1] < pr_in_opp_abs[:,0])
        pr_in_opp_abs[dir_to_swap, 0], pr_in_opp_abs[dir_to_swap, 1] = \
                pr_in_opp_abs[dir_to_swap, 1], pr_in_opp_abs[dir_to_swap, 0]
        pr_to_cell_location(pr_in_opp_abs, normal_idx, 'Vertex', bnd_is_max)

        opp_jn_key = PT.get_value(jn).split('/')[-1] + '/' + PT.get_value(PT.get_child_from_name(jn, 'GridConnectivityDonorName'))
        opposed_joins = ori_jn_to_pr_glob[opp_jn_key]

        to_delete.append(jn)
        i_sub_jn = 0
        for opposed_join in opposed_joins:
          pr_opp_abs = np.copy(PT.get_value(opposed_join))
          # Also swap opposed jn (using same directions)
          pr_opp_abs[dir_to_swap, 0], pr_opp_abs[dir_to_swap, 1] = \
                  pr_opp_abs[dir_to_swap, 1], pr_opp_abs[dir_to_swap, 0]
          pr_to_cell_location(pr_opp_abs, normal_idx, 'Vertex', bnd_is_max)
          inter = intersect_pr(pr_in_opp_abs[dirs,:], pr_opp_abs[dirs,:])
          if inter is not None:
            sub_prd = np.empty((3,2), dtype=np.int32)
            sub_prd[dirs,:] = inter
            sub_prd[normal_idx,:] = pr_in_opp_abs[normal_idx,:]
            # Go back to vertex and invert swap
            pr_to_cell_location(sub_prd, normal_idx, 'Vertex', bnd_is_max, reverse=True)
            sub_prd[dir_to_swap, 0], sub_prd[dir_to_swap, 1] = \
                    sub_prd[dir_to_swap, 1], sub_prd[dir_to_swap, 0]
            # Go back to dist_zone
            sub_pr = np.empty((3,2), dtype=pr.dtype)
            sub_pr[:,0] = apply_transform_matrix(sub_prd[:,0], dist_prd[:,0], dist_pr[:,0], T_matrix.T)
            sub_pr[:,1] = apply_transform_matrix(sub_prd[:,1], dist_prd[:,0], dist_pr[:,0], T_matrix.T)
            # Go back to local numbering
            pr_to_global_num(sub_pr, p_zone_offset, reverse=True)
            p_zone_offset_opp = all_offset_zones[PT.get_name(opposed_join)]
            pr_to_global_num(sub_prd, p_zone_offset_opp, reverse=True)

            #Effective creation of GC in part zone
            gc_name  = PT.get_name(jn) + '.' + str(i_sub_jn)
            # Catch opposite base if present
            opp_path = PT.get_value(jn)
            opp_base = opp_path.split('/')[0] + '/' if '/' in opp_path else ''
            opp_zone = PT.get_name(opposed_join)
            part_gc = PT.new_GridConnectivity1to1(gc_name, opp_base + opp_zone, transform=transform, parent=zone_gc)
            PT.new_PointRange('PointRange',      sub_pr,  parent=part_gc)
            PT.new_PointRange('PointRangeDonor', sub_prd, parent=part_gc)
            PT.add_child(part_gc, PT.get_child_from_label(jn, 'GridConnectivityProperty_t'))
            PT.add_child(part_gc, PT.get_child_from_name(jn, 'GridConnectivityDonorName'))
            i_sub_jn += 1
      elif PT.get_child_from_label(jn, 'GridConnectivityType_t') is not None:
        #This is a join, but not 1to1. So we just move it with other jns
        PT.set_label(jn, 'GridConnectivity_t')
        PT.rm_children_from_label(jn, 'GridConnectivityType_t') # Will be added after by post_split
        PT.add_child(zone_gc, jn)
        to_delete.append(jn)
    #Cleanup
    zbc = PT.get_child_from_label(part, 'ZoneBC_t') #All jns are stored under ZBC
    for node in to_delete:
      PT.rm_child(zbc, node)

def create_zone_gnums(cell_window, dist_zone_cell_size, dtype=pdm_dtype):
  """
  Create the vertex, face and cell global numbering for a partitioned zone
  from the cell_window array, a (3,2) shaped array indicating where starts and ends the
  partition cells (semi open, start at 1) and the dist_zone_cell_size (ie number of cells
  of the original dist_zone in each direction)
  """

  dist_cell_per_dir = dist_zone_cell_size
  dist_vtx_per_dir  = dist_zone_cell_size + 1
  dist_face_per_dir = n_face_per_dir(dist_vtx_per_dir, dist_cell_per_dir)

  part_cell_per_dir = cell_window[:,1] - cell_window[:,0]
  part_face_per_dir = n_face_per_dir(part_cell_per_dir+1, part_cell_per_dir)

  # Vertex
  i_ar  = np.arange(cell_window[0,0], cell_window[0,1]+1, dtype=dtype)
  j_ar  = np.arange(cell_window[1,0], cell_window[1,1]+1, dtype=dtype).reshape(-1,1)
  k_ar  = np.arange(cell_window[2,0], cell_window[2,1]+1, dtype=dtype).reshape(-1,1,1)
  vtx_lntogn = s_numbering.ijk_to_index(i_ar, j_ar, k_ar, dist_vtx_per_dir).flatten()

  # Cell
  i_ar  = np.arange(cell_window[0,0], cell_window[0,1], dtype=dtype)
  j_ar  = np.arange(cell_window[1,0], cell_window[1,1], dtype=dtype).reshape(-1,1)
  k_ar  = np.arange(cell_window[2,0], cell_window[2,1], dtype=dtype).reshape(-1,1,1)
  cell_lntogn = s_numbering.ijk_to_index(i_ar, j_ar, k_ar, dist_cell_per_dir).flatten()

  # Faces
  shifted_nface_p = np_utils.sizes_to_indices(part_face_per_dir)
  ijk_to_faceIndex = [s_numbering.ijk_to_faceiIndex, s_numbering.ijk_to_facejIndex, s_numbering.ijk_to_facekIndex]
  face_lntogn = np.empty(shifted_nface_p[-1], dtype=dtype)
  for idir in range(3):
    i_ar  = np.arange(cell_window[0,0], cell_window[0,1]+(idir==0), dtype=dtype)
    j_ar  = np.arange(cell_window[1,0], cell_window[1,1]+(idir==1), dtype=dtype).reshape(-1,1)
    k_ar  = np.arange(cell_window[2,0], cell_window[2,1]+(idir==2), dtype=dtype).reshape(-1,1,1)
    face_lntogn[shifted_nface_p[idir]:shifted_nface_p[idir+1]] = ijk_to_faceIndex[idir](i_ar, j_ar, k_ar, \
        dist_cell_per_dir, dist_vtx_per_dir).flatten()

  return vtx_lntogn, face_lntogn, cell_lntogn

def part_s_zone(d_zone, d_zone_weights, comm):

  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  n_part_this_zone = np.array(len(d_zone_weights), dtype=np.int32)
  n_part_each_proc = np.empty(n_rank, dtype=np.int32)
  comm.Allgather(n_part_this_zone, n_part_each_proc)

  my_weights = np.asarray(d_zone_weights, dtype=np.float64)
  all_weights = np.empty(n_part_each_proc.sum(), dtype=np.float64)
  comm.Allgatherv(my_weights, [all_weights, n_part_each_proc])

  all_parts = SCT.split_S_block(PT.Zone.CellSize(d_zone), len(all_weights), all_weights)

  my_start = n_part_each_proc[:i_rank].sum()
  my_end   = my_start + n_part_this_zone
  my_parts = all_parts[my_start:my_end]

  part_zones = []
  idx_dim = PT.get_value(d_zone).shape[0]
  for i_part, part in enumerate(my_parts):
    #Get dim and setup zone
    cell_bounds = np.asarray(part, dtype=np.int32) + 1 #Semi open, but start at 1
    n_cells = np.diff(cell_bounds)
    pzone_name = MT.conv.add_part_suffix(PT.get_name(d_zone), i_rank, i_part)
    pzone_dims = np.hstack([n_cells+1, n_cells, np.zeros((idx_dim,1), dtype=np.int32)])
    part_zone  = PT.new_Zone(pzone_name, size=pzone_dims, type='Structured')

    vtx_lntogn, face_lntogn, cell_lntogn = create_zone_gnums(cell_bounds, PT.Zone.CellSize(d_zone))
    gn_node = MT.newGlobalNumbering({'Vertex' : vtx_lntogn, 'Face' : face_lntogn, 'Cell' : cell_lntogn},
                                    parent=part_zone)
    _cell_bounds = np.copy(cell_bounds, order='F')
    _cell_bounds[:,1] -= 1
    PT.new_node("CellRange", "IndexRange_t", _cell_bounds, parent=gn_node)
    PT.new_DataArray("CellSize", PT.Zone.CellSize(d_zone), parent=gn_node)

    create_bcs(d_zone, part_zone, cell_bounds[:,0])

    part_zones.append(part_zone)

  BTP.dist_coords_to_part_coords(d_zone, part_zones, comm)

  parts_offset = [np.asarray(part, dtype=np.int32)[:,0] + 1 for part in my_parts]
  create_internal_gcs(d_zone, part_zones, parts_offset, comm)

  return part_zones
