import mpi4py.MPI as MPI

import maia
import maia.pytree as PT
from   maia.algo.part.extraction_utils import LOC_TO_DIM
from   maia.factory  import dist_from_part

import numpy as np

comm = MPI.COMM_WORLD

'''
QUESTIONS:
  - les PR peuvent etre inversées ?
'''


class Extractor:
  def __init__(self,
               part_tree, point_range, location, comm, 
               equilibrate=False):
    self.part_tree = part_tree
    self.comm      = comm

    # Get zones by domains
    part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm).values()
    # Check : monodomain
    assert len(part_tree_per_dom) == 1

    self.dim = LOC_TO_DIM[location]
    assert self.dim in [0,2,3], "[MAIA] Error : dimensions 1 not yet implemented"
    cell_dim = 3 if location == 'Vertex' else self.dim 
    
    # ExtractPart CGNSTree
    extracted_tree = PT.new_CGNSTree()
    extracted_base = PT.new_CGNSBase('Base', cell_dim=cell_dim, phy_dim=3, parent=extracted_tree)

    for i_domain, part_zones in enumerate(part_tree_per_dom):
      extracted_zones = extract_part_one_domain(part_zones, point_range[i_domain], self.dim, comm,
                                                    equilibrate=False)
      for zone in extracted_zones:
        if PT.Zone.n_vtx(zone)!=0:
          PT.add_child(extracted_base, zone)
    
    self.extracted_tree = extracted_tree
  
  def get_extract_part_tree(self):
    return self.extracted_tree


def extract_part_one_domain(part_zones, point_range, dim, comm, equilibrate=False):
  extract_zones = list()
  for i_part, part_zone in enumerate(part_zones):
    # PT.print_tree(part_zone)
    zone_dim = PT.get_value(part_zone)
    pr = point_range[i_part]

    extract_zone = PT.new_Zone(PT.get_name(part_zone), type='Structured', size=np.zeros((3,3), dtype=np.int32))
    
    if pr.size==0:
      extract_zones.append(extract_zone)
      continue

    size_per_dim = np.diff(pr)[:,0]

    n_dim_pop = 0
    # n_dim_pop = idx.size
    extract_zone_dim = np.zeros((3-n_dim_pop,3), dtype=np.int32)
    extract_zone_dim[:,0] = size_per_dim+1 # size_per_dim[mask]+1
    extract_zone_dim[:,1] = size_per_dim   # size_per_dim[mask]
    PT.set_value(extract_zone, extract_zone_dim)

    # > Get coordinates
    cx, cy, cz = PT.Zone.coordinates(part_zone)
    extract_cx = cx[pr[0,0]-1:pr[0,1], pr[1,0]-1:pr[1,1], pr[2,0]-1:pr[2,1]]
    extract_cy = cy[pr[0,0]-1:pr[0,1], pr[1,0]-1:pr[1,1], pr[2,0]-1:pr[2,1]]
    extract_cz = cz[pr[0,0]-1:pr[0,1], pr[1,0]-1:pr[1,1], pr[2,0]-1:pr[2,1]]
    # extract_cx = np.reshape(extract_cx, (10,10))
    # extract_cy = np.reshape(extract_cy, (10,10))
    # extract_cz = np.reshape(extract_cz, (10,10))
    PT.new_GridCoordinates(fields={'CoordinateX':extract_cx,
                                   'CoordinateY':extract_cy,
                                   'CoordinateZ':extract_cz},
                           parent=extract_zone)

    PT.print_tree(extract_zone)
    extract_zones.append(extract_zone)
  return extract_zones


def extract_part_s_from_bc_name(part_tree, bc_name, comm):
  
  # > Define extracted tree
  extracted_tree = PT.new_CGNSTree()
  base_name = PT.get_name(PT.get_node_from_label(part_tree, 'CGNSBase_t'))
  extracted_base = PT.new_CGNSBase(base_name, cell_dim = 2, phy_dim = 3, parent = extracted_tree) #faire un truc general pour cellDim et phy_Dim
    

  local_part_tree   = PT.shallow_copy(part_tree)
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(local_part_tree, comm)

  point_range = list()
  location = ''
  for domain, part_zones in part_tree_per_dom.items():
    point_range_domain = list()
    for part_zone in part_zones:
      bc_n = PT.get_node_from_name_and_label(part_zone, bc_name, 'BC_t') 
      if bc_n is not None:
        pr_n = PT.get_child_from_name(bc_n, 'PointRange')
        point_range_domain.append(PT.get_value(pr_n))
        location = PT.Subset.GridLocation(bc_n)
      else:
        point_range_domain.append(np.empty(0, np.int32))
    point_range.append(point_range_domain)

  # Get location if proc has no zsr
  location = comm.allreduce(location, op=MPI.MAX)
  

  extractor = Extractor(part_tree, point_range, location, comm)

  extracted_tree = extractor.get_extract_part_tree()

  return extracted_tree