import time
import mpi4py.MPI as MPI

import maia
import maia.pytree as PT
from   maia.algo.part.extraction_utils import LOC_TO_DIM
from   maia.factory  import dist_from_part
from   maia.utils import s_numbering
import maia.utils.logging as mlog
from   maia import npy_pdm_gnum_dtype as pdm_gnum_dtype
from   maia.algo.dist.s_to_u import compute_transform_matrix, apply_transform_matrix,\
                                    gc_is_reference, guess_bnd_normal_index
from   .extraction_utils   import local_pl_offset, LOC_TO_DIM, get_partial_container_stride_and_order
import numpy as np

comm = MPI.COMM_WORLD

DIMM_TO_DIMF = { 0: {'Vertex':'Vertex'},
               # 1: {'Vertex': None,    'EdgeCenter':None, 'FaceCenter':None, 'CellCenter':None},
                 2: {'Vertex':'Vertex', 'EdgeCenter':'EdgeCenter', 'FaceCenter':'CellCenter'},
                 3: {'Vertex':'Vertex', 'EdgeCenter':'EdgeCenter', 'FaceCenter':'FaceCenter', 'CellCenter':'CellCenter'}}

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
    if location == 'Vertex':
      cell_dim = -1
      for domain_prs in point_range:
        for part_pr in domain_prs:
          if part_pr.size!=0:
            size_per_dim = np.diff(part_pr)[:,0]
            idx = np.where(size_per_dim!=0)[0]
            cell_dim = idx.size
      cell_dim = comm.allreduce(cell_dim, op=MPI.MAX)
    else:
      cell_dim = self.dim 

    
    # ExtractPart CGNSTree
    extract_tree = PT.new_CGNSTree()
    extract_base = PT.new_CGNSBase('Base', cell_dim=cell_dim, phy_dim=3, parent=extract_tree)

    for i_domain, part_zones in enumerate(part_tree_per_dom):
      extract_zones = extract_part_one_domain(part_zones, point_range[i_domain], self.dim, comm,
                                                    equilibrate=False)
      for zone in extracted_zones:
        if PT.Zone.n_vtx(zone)!=0:
          PT.add_child(extracted_base, zone)
    
    self.extract_tree = extract_tree
  
  def exchange_fields(self, fs_container):
    # Get zones by domains (only one domain for now)
    part_tree_per_dom = dist_from_part.get_parts_per_blocks(self.part_tree, self.comm).values()

    # Get zone from extractpart
    extract_zones = PT.get_all_Zone_t(self.extract_tree)

    for container_name in fs_container:
      for i_domain, part_zones in enumerate(part_tree_per_dom):
        exchange_field_one_domain(self.part_tree, extract_zones, self.dim, \
            container_name, self.comm)

  def get_extract_part_tree(self):
    return self.extract_tree


def exchange_field_one_domain(part_tree, extract_zones, mesh_dim, container_name, comm) :

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
  
  fields_query = lambda n: PT.get_label(n) in ['DataArray_t', 'IndexArray_t']
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
    dist_from_part.discover_nodes_from_matching(mask_zone, part_zones, ['ZoneBC_t', bc_name], comm, child_list=['PointList', 'GridLocation_t'])
    ref_zsr_node = PT.get_child_from_predicates(mask_zone, f'ZoneBC_t/{bc_name}')
  elif gc_descriptor_n is not None:
    gc_name      = PT.get_value(gc_descriptor_n)
    dist_from_part.discover_nodes_from_matching(mask_zone, part_zones, ['ZoneGridConnectivity_t', gc_name], comm, child_list=['PointList', 'GridLocation_t'])
    ref_zsr_node = PT.get_child_from_predicates(mask_zone, f'ZoneGridConnectivity_t/{gc_name})')
  
  grid_location = PT.Subset.GridLocation(ref_zsr_node)
  partial_field = PT.get_child_from_name(ref_zsr_node, 'PointList') is not None
  assert grid_location in ['Vertex', 'FaceCenter', 'CellCenter']


  parent_lnum_path = {'Vertex'    :':maia#exchutils/parent_lnum_vtx',
                      'FaceCenter':':maia#exchutils/parent_lnum_cell'}

  for extract_zone in extract_zones:
    
    zone_name = PT.get_name(extract_zone)
    part_zone = PT.get_node_from_name_and_label(part_tree, zone_name, 'Zone_t')

    if PT.get_label(mask_container) == 'FlowSolution_t':
      FS_ep = PT.new_FlowSolution(container_name, loc=DIMM_TO_DIMF[mesh_dim][grid_location], parent=extract_zone)
    elif PT.get_label(mask_container) == 'ZoneSubRegion_t':
      FS_ep = PT.new_ZoneSubRegion(container_name, loc=DIMM_TO_DIMF[mesh_dim][grid_location], parent=extract_zone)
    else:
      raise TypeError
    if   grid_location=='Vertex':
      zone_elt_size = PT.Zone.VertexSize(extract_zone)
    elif grid_location=='FaceCenter':
      zone_elt_size = PT.Zone.FaceSize(extract_zone)
    elif grid_location=='CellCenter':
      zone_elt_size = PT.Zone.CellSize(extract_zone)
    
    part2_pl = PT.get_value(PT.get_node_from_path(extract_zone, parent_lnum_path[grid_location]))

    for fld_node in PT.get_children_from_label(mask_container, 'DataArray_t'):
      fld_name = PT.get_name(fld_node)
      fld_path = f"{container_name}/{fld_name}"

      fld_data = PT.get_value(PT.get_node_from_path(part_zone,fld_path))
      extract_fld_data = fld_data.flatten(order='F')[part2_pl-1]

      PT.new_DataArray(fld_name, extract_fld_data.reshape(zone_elt_size, order='F'), parent=FS_ep)

    if PT.Zone.n_vtx(extract_zone)==0:
      continue
def extract_part_one_domain(part_zones, point_range, dim, comm, equilibrate=False):
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
  for i_part, part_zone in enumerate(part_zones):
    # PT.print_tree(part_zone)
    zone_name = PT.get_name(part_zone)
    zone_dim  = PT.get_value(part_zone)
    pr = point_range[i_part]
    extract_zone = PT.new_Zone(zone_name, type='Structured', size=np.zeros((3,3), dtype=np.int32))
    
    if pr.size==0:
      lvtx_gn.append(np.empty(0, dtype=pdm_gnum_dtype))
      lcell_gn.append(np.empty(0, dtype=pdm_gnum_dtype))
      extract_zones.append(extract_zone)
      continue

    size_per_dim = np.diff(pr)[:,0]
    print(f'size_per_dim = {size_per_dim}')
    mask = np.isin(size_per_dim, 0, invert=True)
    idx = np.where(not(mask).all())[0]
    print(f'size_per_dim = {size_per_dim}')
    print(f'mask = {mask}')
    print(f'idx = {idx}')
    print(f'size_per_dim[mask]+1 = {size_per_dim[mask]+1}')

    # n_dim_pop = 0
    n_dim_pop = idx.size
    extract_zone_dim = np.zeros((3-n_dim_pop,3), dtype=np.int32)
    print(f'extract_zone_dim = {extract_zone_dim}')
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
    gn = PT.get_child_from_name(part_zone, ':CGNS#GlobalNumbering')
    gn_vtx  = PT.get_value(PT.get_node_from_name(gn, 'Vertex'))
    gn_face = PT.get_value(PT.get_node_from_name(gn, 'Face'))

    print(f'pr = {pr}')
    DIM_TO_LOC = {0:'Vertex', 1:'EdgeCenter', 2:'FaceCenter', 3:'CellCenter'}
    ijk_to_faceIndex = [s_numbering.ijk_to_faceiIndex, s_numbering.ijk_to_facejIndex, s_numbering.ijk_to_facekIndex]
    
    extract_dir = maia.algo.dist.s_to_u.guess_bnd_normal_index(pr, DIM_TO_LOC[dim])
    print(f'extract_dir = {extract_dir}')

    i_ar_cell = np.arange(min(pr[0]), max(pr[0]))                 if min(pr[0])!=max(pr[0]) else np.array([max(pr[0])], dtype=np.int32)
    j_ar_cell = np.arange(min(pr[1]), max(pr[1])).reshape(-1,1)   if min(pr[1])!=max(pr[1]) else np.array([max(pr[1])], dtype=np.int32)
    k_ar_cell = np.arange(min(pr[2]), max(pr[2])).reshape(-1,1,1) if min(pr[2])!=max(pr[2]) else np.array([max(pr[2])], dtype=np.int32)
    print(f'i_ar_cell = {i_ar_cell}')
    print(f'j_ar_cell = {j_ar_cell}')
    print(f'k_ar_cell = {k_ar_cell}')

    vtx_per_dir  = zone_dim[:,0]
    cell_per_dir = zone_dim[:,1]
    print(f'vtx_per_dir  = {vtx_per_dir}')
    print(f'cell_per_dir = {cell_per_dir}')
    locnum_face = ijk_to_faceIndex[2](i_ar_cell, j_ar_cell, k_ar_cell, \
                          cell_per_dir, vtx_per_dir).flatten()
    print(f'locnum_face = {locnum_face.size}')
    lcell_gn.append(gn_face[locnum_face -1])

    i_ar_vtx = np.arange(min(pr[0]), max(pr[0])+1)
    j_ar_vtx = np.arange(min(pr[1]), max(pr[1])+1).reshape(-1,1)
    k_ar_vtx = np.arange(min(pr[2]), max(pr[2])+1).reshape(-1,1,1)
    locnum_vtx = s_numbering.ijk_to_index(i_ar_vtx, j_ar_vtx, k_ar_vtx, vtx_per_dir).flatten()
    lvtx_gn.append(gn_vtx[locnum_vtx - 1])
    

    exchutils_n = PT.new_node(':maia#exchutils', label='UserDefinedData_t', parent=extract_zone)
    PT.new_DataArray('parent_lnum_vtx', value=locnum_vtx, parent=exchutils_n)
    PT.new_DataArray('parent_lnum_cell', value=locnum_face, parent=exchutils_n)

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
          new_gc_pr  = np.delete(new_gc_pr , extract_dir, 0)
          new_gc_prd = np.delete(new_gc_prd, transform[extract_dir]-1, 0)

          gc_name = PT.get_name(gc_n)
          gc_donorname = PT.get_value(gc_n)
          extract_gc_n = PT.new_GridConnectivity1to1(gc_name, donor_name=gc_donorname, point_range=new_gc_pr, point_range_donor=new_gc_prd, parent=extract_zgc)
    extract_zones.append(extract_zone)
  # > Create GlobalNumbering
  partial_gnum_vtx  = maia.algo.part.point_cloud_utils.create_sub_numbering(lvtx_gn, comm)
  partial_gnum_cell = maia.algo.part.point_cloud_utils.create_sub_numbering(lcell_gn, comm)
  
  if len(partial_gnum_vtx)!=0:
    for i_part, extract_zone in enumerate(extract_zones):
      PT.maia.newGlobalNumbering({'Vertex' : partial_gnum_vtx [i_part],
                                  'Cell'   : partial_gnum_cell[i_part]},
                                 parent=extract_zone)
  else:
    assert len(partial_gnum_cell)!=0

  # return extract_zones, exch_tool_box
  return extract_zones


def extract_part_s_from_bc_name(part_tree, bc_name, comm,
                                transfer_dataset=True,
                                containers_name=[],
                                **options):
  # > Define extracted tree
  extract_tree = PT.new_CGNSTree()
  base_name = PT.get_name(PT.get_node_from_label(part_tree, 'CGNSBase_t'))
  extract_base = PT.new_CGNSBase(base_name, cell_dim = 2, phy_dim = 3, parent = extract_tree) #faire un truc general pour cellDim et phy_Dim
    

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
  
  start = time.time()

  extractor = Extractor(part_tree, point_range, location, comm)

  l_containers_name = [name for name in containers_name]
  if l_containers_name:
    extractor.exchange_fields(l_containers_name)
  end = time.time()
  
  # Print some light stats
  elts_kind = ['vtx', 'edges', 'faces', 'cells'][extractor.dim]
  if extractor.dim == 0:
    n_cell = sum([PT.Zone.n_vtx(zone) for zone in PT.iter_all_Zone_t(extract_tree)])
  else:
    n_cell = sum([PT.Zone.n_cell(zone) for zone in PT.iter_all_Zone_t(extract_tree)])
  n_cell_all = comm.allreduce(n_cell, MPI.SUM)
  
  mlog.info(f"Extraction from BC \"{bc_name}\" completed ({end-start:.2f} s) -- "
            f"Extracted tree has locally {mlog.size_to_str(n_cell)} {elts_kind} "
            f"(Σ={mlog.size_to_str(n_cell_all)})")

  extract_tree = extractor.get_extract_part_tree()



  return extract_tree