from mpi4py import MPI
import numpy              as np

import Converter.Internal as I

from maia.sids import sids as SIDS
from .                               import split_cut_tree as SCT
from maia.tree_exchange.dist_to_part import grid_coords
from maia.transform.dist_tree.convert_s_to_u import guess_bnd_normal_index

def ijk_to_index(i,j,k,n_elmt):
  return i+(j-1)*n_elmt[0]+(k-1)*n_elmt[0]*n_elmt[1] 

def merge_dist_boundaries(dist_zone):
  base_bound = {k:[] for k in ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]}
  idx_to_dir = {0: 'x', 1: 'y', 2:'z'}

  for zone_bc in I.getNodesFromType1(dist_zone, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zone_bc, 'BC_t'):
      grid_loc_n = I.getNodeFromName(bc, 'GridLocation')
      grid_loc   = I.getValue(grid_loc_n) if grid_loc_n else 'Vertex'
      point_range = I.getNodeFromName(bc, 'PointRange')[1]
      bnd_normal_index = guess_bnd_normal_index(point_range, grid_loc)

      pr_val = point_range[bnd_normal_index,0]

      extr = 'min' if pr_val == 1 else 'max'
      base_bound[idx_to_dir[bnd_normal_index] + extr].append(I.getName(bc))

  for zone_gc in I.getNodesFromType1(dist_zone, 'ZoneGridConnectivity_t'):
    for gc in I.getNodesFromType1(zone_gc, 'GridConnectivity1to1_t'):
      grid_loc_n = I.getNodeFromName(gc, 'GridLocation')
      point_range = I.getNodeFromName(gc, 'PointRange')[1]
      bnd_normal_index = guess_bnd_normal_index(point_range, 'Vertex')

      pr_val = point_range[bnd_normal_index,0]

      extr = 'min' if pr_val == 1 else 'max'
      base_bound[idx_to_dir[bnd_normal_index] + extr].append(I.getName(gc))

  return base_bound


def create_bcs(zone, part, comm):
  i_rank = comm.Get_rank()
  dist_bound = merge_dist_boundaries(zone)
  dir_to_idx = {'x':0, 'y':1, 'z':2}

  zbc = I.newZoneBC(parent=part)
  part_bounds = list()

  abs_dims = I.getNodeFromName1(part, 'AbsDims')[1]

  for bnd in dist_bound.keys():

    normal_idx = dir_to_idx[bnd[0]]
    extr = 0 if 'min' in bnd else 1

    local_range = np.ones((3,2), dtype=np.int32)
    local_range[:,1] = SIDS.CellSize(part)
    local_range[normal_idx, 1-extr] = local_range[normal_idx, extr]

    global_range = local_range + abs_dims[:,0].reshape((3,1)) - 1

    global_range_dist = np.ones((3,2), dtype=np.int32)
    global_range_dist[:,1] = SIDS.CellSize(zone)

    #Ancienne BC/GC ou nouveau raccord
    is_old_bc = global_range[normal_idx,extr] == global_range_dist[normal_idx,extr]

    if is_old_bc:
      dirs = np.where(np.arange(3) != normal_idx)[0]
      for dist_bc_name in dist_bound[bnd]:
        dist_bc       = I.getNodeFromName(zone, dist_bc_name)
        range_dist_bc = I.getNodeFromName1(dist_bc, 'PointRange')[1]
        grid_loc_n    = I.getNodeFromType1(dist_bc, 'GridLocation_t')
        grid_loc      = I.getValue(grid_loc_n) if grid_loc_n else 'Vertex'
        #Convert dist bc to cell

        #Swap if gc is reversed
        dir_to_swap     = (range_dist_bc[:,1] < range_dist_bc[:,0])
        range_dist_bc[dir_to_swap, 0], range_dist_bc[dir_to_swap, 1] = \
                range_dist_bc[dir_to_swap, 1], range_dist_bc[dir_to_swap, 0]
        if not grid_loc == 'CellCenter':
          range_dist_bc[normal_idx,:] -= int(range_dist_bc[normal_idx,0] != 1)
        if grid_loc == 'Vertex':
          range_dist_bc[dirs,1] -= 1

        sub_pr = np.empty((3,2), dtype=np.int32)
        for d in dirs:
          if range_dist_bc[d,0] <= global_range[d,0] and global_range[d,0] <= range_dist_bc[d,1]:
            sub_pr[d,0] = global_range[d,0]
          elif global_range[d,0] <= range_dist_bc[d,0] and range_dist_bc[d,0] <= global_range[d,1]:
            sub_pr[d,0] = range_dist_bc[d,0]
          else:
            break
          sub_pr[d,1] = min(global_range[d,1], range_dist_bc[d,1])
        sub_pr[normal_idx,:] = global_range[normal_idx,:]
        #Global to local
        sub_pr = sub_pr - abs_dims[:,0].reshape((3,1)) + 1
        #Restore location
        if not grid_loc == 'CellCenter':
          sub_pr[normal_idx,:] += int('max' in bnd)
        if grid_loc == 'Vertex':
          sub_pr[dirs,1] += 1
        sub_pr[dir_to_swap, 0], sub_pr[dir_to_swap, 1] = \
                sub_pr[dir_to_swap, 1], sub_pr[dir_to_swap, 0]

        part_bc = I.newBC(dist_bc_name + 'suff', sub_pr, parent=zbc)
        I.newGridLocation(grid_loc, parent=part_bc)
    else:
      I.newBC('JN_' + bnd, parent=zbc)



def part_s_zone(zone, zone_weights, comm):
  # n_part_this_zone = len(dzone_to_weighted_parts[I.getName(zone)])
  # weights_tot = comm.allgather(dzone_to_weighted_parts[I.getName(zone)])
  # n_part_per_proc = [len(w) for w in weights_tot]
  # fl_weights_tot = [item for sublist in weights_tot for item in sublist] #Flatten
  #print("Rank", comm.Get_rank(), "zone", zone[0], 'npart', n_part_this_zone, 'wtot', fl_weights_tot)

  # all_parts = SCT.split_S_block(zone[1][:,1], len(fl_weights_tot), fl_weights_tot)
  # my_start = sum([len(k) for k in weights_tot[:i_rank]])
  # my_end   = my_start + n_part_this_zone
  # my_parts = all_parts[my_start:my_end]
  # print("({0})".format(zone[0]), "Rank", comm.Get_rank(), "my parts", my_parts, "\n")

  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  n_part_this_zone = np.array(len(zone_weights), dtype=np.int32)
  n_part_each_proc = np.empty(n_rank, dtype=np.int32)
  comm.Allgather(n_part_this_zone, n_part_each_proc)

  my_weights = np.asarray(zone_weights, dtype=np.float64)
  all_weights = np.empty(n_part_each_proc.sum(), dtype=np.float64)
  comm.Allgatherv(my_weights, [all_weights, n_part_each_proc])

  all_parts = SCT.split_S_block(zone[1][:,1], len(all_weights), all_weights)

  my_start = n_part_each_proc[:i_rank].sum()
  my_end   = my_start + n_part_this_zone
  my_parts = all_parts[my_start:my_end]

  part_zones = []
  for i_part, part in enumerate(my_parts):
    #Get dim and setup zone
    cell_bounds = np.asarray(part, dtype=np.int32) + 1 #Semi open, but start at 1
    n_cells = np.diff(cell_bounds)
    pzone_name = '{0}.P{1}.N{2}'.format(zone[0], i_rank, i_part)
    pzone_dims = np.hstack([n_cells+1, n_cells, np.zeros((3,1), dtype=np.int32)])
    part_zone  = I.newZone(pzone_name, pzone_dims, ztype='Structured')

    absolute_cells = np.copy(cell_bounds)
    absolute_cells[:,1] -= 1
    I.createNode('AbsDims', 'UserDefinedData_t', absolute_cells, parent=part_zone)

    #Get ln2gn : following convvention i, j, k increasing. Add 1 to end for vtx
    lngn_zone = I.createNode(':CGNS#Lntogn', 'UserDefinedData_t', parent=part_zone)
    i_ar  = np.arange(cell_bounds[0,0], cell_bounds[0,1]+1, dtype=np.int32)
    j_ar  = np.arange(cell_bounds[1,0], cell_bounds[1,1]+1, dtype=np.int32).reshape(-1,1)
    k_ar  = np.arange(cell_bounds[2,0], cell_bounds[2,1]+1, dtype=np.int32).reshape(-1,1,1)
    vtx_lntogn = ijk_to_index(i_ar, j_ar, k_ar, zone[1][:,0]).flatten()
    I.newDataArray('Vertex', vtx_lntogn, parent=lngn_zone)
    i_ar  = np.arange(cell_bounds[0,0], cell_bounds[0,1], dtype=np.int32)
    j_ar  = np.arange(cell_bounds[1,0], cell_bounds[1,1], dtype=np.int32).reshape(-1,1)
    k_ar  = np.arange(cell_bounds[2,0], cell_bounds[2,1], dtype=np.int32).reshape(-1,1,1)
    cell_lntogn = ijk_to_index(i_ar, j_ar, k_ar, zone[1][:,1]).flatten()
    I.newDataArray('Cell', cell_lntogn, parent=lngn_zone)

    part_zones.append(part_zone)

  grid_coords.dist_coords_to_part_coords2(zone, part_zones, comm)
  for part in part_zones:
    create_bcs(zone, part, comm)

  return part_zones
