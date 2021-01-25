from mpi4py import MPI
import numpy              as np

import Converter.Internal as I

from .                               import split_cut_tree as SCT
from maia.tree_exchange.dist_to_part import grid_coords

def ijk_to_index(i,j,k,n_elmt):
  return i+(j-1)*n_elmt[0]+(k-1)*n_elmt[0]*n_elmt[1] 

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

  return part_zones
