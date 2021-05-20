from mpi4py import MPI
import numpy              as np

import Converter.Internal as I
import maia.sids.Internal_ext as IE

from maia.sids import sids as SIDS
from maia.sids import conventions as conv
from .                               import split_cut_tree as SCT
from maia.tree_exchange.dist_to_part import data_exchange  as BTP
from maia.transform.dist_tree.convert_s_to_u import guess_bnd_normal_index, \
                                                    compute_transform_matrix, \
                                                    apply_transformation
from maia.sids import sids

idx_to_dir = {0:'x', 1:'y', 2:'z'}
dir_to_idx = {'x':0, 'y':1, 'z':2}
min_max_as_int = lambda st : 0 if 'min' in st else 1

def ijk_to_index(i,j,k,n_elmt):
  return i+(j-1)*n_elmt[0]+(k-1)*n_elmt[0]*n_elmt[1]

def zone_cell_range(zone):
  """ Return the size of a point_range 2d array """
  n_cell = SIDS.CellSize(zone)
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

  for bnd_path in ['ZoneBC_t/BC_t', 'ZoneGridConnectivity_t/GridConnectivity1to1_t']:
    for bnd in IE.getNodesFromTypeMatching(zone, bnd_path):
      grid_loc    = SIDS.GridLocation(bnd)
      point_range = I.getNodeFromName(bnd, 'PointRange')[1]
      bnd_normal_index = guess_bnd_normal_index(point_range, grid_loc)

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
  zbc = I.newZoneBC(parent=p_zone)
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
        range_dist_bc = np.copy(I.getNodeFromName1(dist_bc, 'PointRange')[1])
        grid_loc      = SIDS.GridLocation(dist_bc)

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
          part_bc = I.newBC(I.getName(dist_bc), sub_pr, parent=zbc)
          I.setValue(part_bc, I.getValue(dist_bc))
          I.newGridLocation(grid_loc, parent=part_bc)
          if I.getNodeFromName1(dist_bc, 'Ordinal') is not None:
            I._addChild(part_bc, I.getNodeFromName1(dist_bc, 'Ordinal'))
            I._addChild(part_bc, I.getNodeFromName1(dist_bc, 'OrdinalOpp'))
            I._addChild(part_bc, I.getNodeFromName1(dist_bc, 'Transform'))
            I.createChild(part_bc, 'distPR', 'IndexRange_t', I.getNodeFromName1(dist_bc, 'PointRange')[1])
            I.createChild(part_bc, 'distPRDonor', 'IndexRange_t', I.getNodeFromName1(dist_bc, 'PointRangeDonor')[1])
            I.newDataArray('zone_offset', p_zone_offset, parent=part_bc)

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
        jn_list[i_part].append(I.newPointRange(geo_bnd, range_part_bc_g))

  # 2. Exchange. We will need to compare to other parts joins
  all_offset_list = comm.allgather(p_zones_offset)
  all_jn_list     = comm.allgather(jn_list)

  # 3. Process
  for i_part, p_zone in enumerate(p_zones):
    zgc = I.newZoneGridConnectivity(parent=p_zone)
    for jn in jn_list[i_part]:
      # Get data for the current join
      normal_idx = dir_to_idx[I.getName(jn)[0]]
      extr       = min_max_as_int(I.getName(jn)[1:])
      shift = 1 - 2*extr #1 si min, -1 si max
      dirs  = np.where(np.arange(3) != normal_idx)[0]
      my_pr = I.getValue(jn)

      # Check opposite joins
      for j_proc, opp_parts in enumerate(all_jn_list):
        for j_part, opp_part in enumerate(opp_parts):
          for opp_jn in opp_part:
            opp_normal_idx = dir_to_idx[I.getName(opp_jn)[0]]
            opp_extr       = min_max_as_int(I.getName(opp_jn)[1:])
            opp_pr         = I.getValue(opp_jn)
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
                gc_name  = conv.name_intra_gc(comm.Get_rank(), i_part, j_proc, j_part)
                opp_zone = conv.add_part_suffix(I.getName(d_zone), j_proc, j_part)
                part_gc = I.newGridConnectivity1to1(gc_name, opp_zone,
                                                    pointRange=sub_pr, pointRangeDonor=sub_pr_d,
                                                    transform = [1,2,3], parent=zgc)
                I.newGridLocation('Vertex', parent=part_gc)

def split_original_joins_S(all_part_zones, comm):
  """
  Intersect the original joins of the meshes. Such joins must are stored in ZoneBC_t node
  at the end of S partitioning with needed information stored inside
  """
  ordinal_to_pr = dict()
  zones_offsets = dict()
  for part in all_part_zones:
    for jn in IE.getNodesFromTypeMatching(part, 'ZoneBC_t/BC_t'):
      if I.getNodeFromName1(jn, 'Ordinal') is not None:
        p_zone_offset = I.getNodeFromName1(jn, 'zone_offset')[1]
        pr_n = I.newPointRange(part[0], np.copy(I.getNodeFromName1(jn, 'PointRange')[1]))
        # Pr dans la num globale de la zone
        pr_to_global_num(pr_n[1], p_zone_offset)
        try:
          ordinal_to_pr[I.getNodeFromName1(jn, 'Ordinal')[1][0]].append(pr_n)
        except KeyError:
          ordinal_to_pr[I.getNodeFromName1(jn, 'Ordinal')[1][0]] = [pr_n]
        zones_offsets[I.getName(part)] = p_zone_offset

  #Gather and create dic ordinal -> List of PR
  ordinal_to_pr_glob = dict()
  all_offset_zones   = dict()
  for ordinal_to_pr_rank in comm.allgather(ordinal_to_pr):
    for key, value in ordinal_to_pr_rank.items():
      if key in ordinal_to_pr_glob:
        ordinal_to_pr_glob[key].extend(value)
      else:
        ordinal_to_pr_glob[key] = value
  for zones_offsets_rank in comm.allgather(zones_offsets):
    all_offset_zones.update(zones_offsets_rank)

  for part in all_part_zones:
    zone_gc = I.createUniqueChild(part, 'ZoneGridConnectivity', 'ZoneGridConnectivity_t')
    to_delete = []
    for jn in IE.getNodesFromTypeMatching(part, 'ZoneBC_t/BC_t'):
      if I.getNodeFromName1(jn, 'Ordinal') is not None:
        dist_pr = I.getNodeFromName1(jn, 'distPR')[1]
        dist_prd = I.getNodeFromName1(jn, 'distPRDonor')[1]
        transform  = I.getNodeFromName1(jn, 'Transform')[1]
        T_matrix = compute_transform_matrix(transform)
        assert sids.GridLocation(jn) == 'Vertex'

        #Jn dans la num globale de la dist_zone
        p_zone_offset = I.getNodeFromName1(jn, 'zone_offset')[1]
        pr = np.copy(I.getNodeFromName1(jn, 'PointRange')[1])
        pr_to_global_num(pr, p_zone_offset)

        #Jn dans la num globale de la dist_zone opposée
        pr_in_opp_abs = np.empty((3,2), dtype=pr.dtype)
        pr_in_opp_abs[:,0] = apply_transformation(pr[:,0], dist_pr[:,0], dist_prd[:,0], T_matrix)
        pr_in_opp_abs[:,1] = apply_transformation(pr[:,1], dist_pr[:,0], dist_prd[:,0], T_matrix)

        #Jn dans la zone opposée et en cellules
        normal_idx = guess_bnd_normal_index(pr_in_opp_abs, 'Vertex')
        dirs       = np.where(np.arange(3) != normal_idx)[0]
        bnd_is_max = pr_in_opp_abs[normal_idx,0] != 1 #Sommets
        dir_to_swap     = (pr_in_opp_abs[:,1] < pr_in_opp_abs[:,0])
        pr_in_opp_abs[dir_to_swap, 0], pr_in_opp_abs[dir_to_swap, 1] = \
                pr_in_opp_abs[dir_to_swap, 1], pr_in_opp_abs[dir_to_swap, 0]
        pr_to_cell_location(pr_in_opp_abs, normal_idx, 'Vertex', bnd_is_max)

        opposed_joins = ordinal_to_pr_glob[I.getNodeFromName1(jn, 'OrdinalOpp')[1][0]]

        to_delete.append(jn)
        i_sub_jn = 0
        for opposed_join in opposed_joins:
          pr_opp_abs = np.copy(I.getValue(opposed_join))
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
            sub_pr[:,0] = apply_transformation(sub_prd[:,0], dist_prd[:,0], dist_pr[:,0], T_matrix.T)
            sub_pr[:,1] = apply_transformation(sub_prd[:,1], dist_prd[:,0], dist_pr[:,0], T_matrix.T)
            # Go back to local numbering
            pr_to_global_num(sub_pr, p_zone_offset, reverse=True)
            p_zone_offset_opp = all_offset_zones[I.getName(opposed_join)]
            pr_to_global_num(sub_prd, p_zone_offset_opp, reverse=True)

            #Effective creation of GC in part zone
            gc_name  = I.getName(jn) + '.' + str(i_sub_jn)
            opp_zone = I.getName(opposed_join)
            part_gc = I.newGridConnectivity1to1(gc_name, opp_zone,
                                                pointRange=sub_pr, pointRangeDonor=sub_prd,
                                                transform = transform, parent=zone_gc)
            I.newGridLocation('Vertex', parent=part_gc)
            i_sub_jn += 1
    #Cleanup
    for node in to_delete:
      I._rmNode(part, node)


def part_s_zone(d_zone, d_zone_weights, comm):

  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  n_part_this_zone = np.array(len(d_zone_weights), dtype=np.int32)
  n_part_each_proc = np.empty(n_rank, dtype=np.int32)
  comm.Allgather(n_part_this_zone, n_part_each_proc)

  my_weights = np.asarray(d_zone_weights, dtype=np.float64)
  all_weights = np.empty(n_part_each_proc.sum(), dtype=np.float64)
  comm.Allgatherv(my_weights, [all_weights, n_part_each_proc])

  all_parts = SCT.split_S_block(SIDS.CellSize(d_zone), len(all_weights), all_weights)

  my_start = n_part_each_proc[:i_rank].sum()
  my_end   = my_start + n_part_this_zone
  my_parts = all_parts[my_start:my_end]

  part_zones = []
  for i_part, part in enumerate(my_parts):
    #Get dim and setup zone
    cell_bounds = np.asarray(part, dtype=np.int32) + 1 #Semi open, but start at 1
    n_cells = np.diff(cell_bounds)
    pzone_name = conv.add_part_suffix(I.getName(d_zone), i_rank, i_part)
    pzone_dims = np.hstack([n_cells+1, n_cells, np.zeros((3,1), dtype=np.int32)])
    part_zone  = I.newZone(pzone_name, pzone_dims, ztype='Structured')

    #Get ln2gn : following convention i, j, k increasing. Add 1 to end for vtx
    lngn_zone = IE.newGlobalNumbering(parent=part_zone)
    i_ar  = np.arange(cell_bounds[0,0], cell_bounds[0,1]+1, dtype=np.int32)
    j_ar  = np.arange(cell_bounds[1,0], cell_bounds[1,1]+1, dtype=np.int32).reshape(-1,1)
    k_ar  = np.arange(cell_bounds[2,0], cell_bounds[2,1]+1, dtype=np.int32).reshape(-1,1,1)
    vtx_lntogn = ijk_to_index(i_ar, j_ar, k_ar, SIDS.VertexSize(d_zone)).flatten()
    I.newDataArray('Vertex', vtx_lntogn, parent=lngn_zone)
    i_ar  = np.arange(cell_bounds[0,0], cell_bounds[0,1], dtype=np.int32)
    j_ar  = np.arange(cell_bounds[1,0], cell_bounds[1,1], dtype=np.int32).reshape(-1,1)
    k_ar  = np.arange(cell_bounds[2,0], cell_bounds[2,1], dtype=np.int32).reshape(-1,1,1)
    cell_lntogn = ijk_to_index(i_ar, j_ar, k_ar, SIDS.CellSize(d_zone)).flatten()
    I.newDataArray('Cell', cell_lntogn, parent=lngn_zone)

    create_bcs(d_zone, part_zone, cell_bounds[:,0])

    part_zones.append(part_zone)

  BTP.dist_coords_to_part_coords(d_zone, part_zones, comm)

  parts_offset = [np.asarray(part, dtype=np.int32)[:,0] + 1 for part in my_parts]
  create_internal_gcs(d_zone, part_zones, parts_offset, comm)

  return part_zones
