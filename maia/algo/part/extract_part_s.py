import copy, time
import mpi4py.MPI as MPI

import maia
import maia.pytree as PT
from   maia.algo.part.extraction_utils import LOC_TO_DIM
from   maia.factory  import dist_from_part
from   maia.utils import np_utils, s_numbering
import maia.utils.logging as mlog
from   maia import npy_pdm_gnum_dtype as pdm_gnum_dtype
from   maia.algo.dist.s_to_u import compute_transform_matrix, apply_transform_matrix,\
                                    gc_is_reference, guess_bnd_normal_index, n_face_per_dir
from   .extraction_utils   import local_pl_offset, LOC_TO_DIM, build_intersection_numbering
import numpy as np

comm = MPI.COMM_WORLD

DIMM_TO_DIMF = { 0: {'Vertex':'Vertex'},
               # 1: {'Vertex': None,    'EdgeCenter':None, 'FaceCenter':None, 'CellCenter':None},
                 2: {'Vertex'     :'Vertex',
                     'EdgeCenter' :'EdgeCenter',
                     'FaceCenter' :'CellCenter', 'IFaceCenter':'CellCenter', 'JFaceCenter':'CellCenter', 'KFaceCenter':'CellCenter'},
                 3: {'Vertex':'Vertex', 'EdgeCenter':'EdgeCenter', 'FaceCenter':'FaceCenter', 'CellCenter':'CellCenter'}}

'''
QUESTIONS:
  - les PR peuvent etre inversées ?
'''
def set_transfer_dataset(bc_n, zsr_bc_n):
  there_is_dataset = False
  assert PT.get_child_from_predicates(bc_n, 'BCDataSet_t/IndexArray_t') is None,\
                 'BCDataSet_t with PointList aren\'t managed'
  ds_arrays = PT.get_children_from_predicates(bc_n, 'BCDataSet_t/BCData_t/DataArray_t')
  for ds_array in ds_arrays:
    PT.new_DataArray(name=PT.get_name(ds_array), value=PT.get_value(ds_array), parent=zsr_bc_n)
  if len(ds_arrays) != 0:
    there_is_dataset = True
    # PL and Location is needed for data exchange, but this should be done in ZSR func
    for name in ['PointRange', 'GridLocation']:
      PT.add_child(zsr_bc_n, PT.get_child_from_name(bc_n, name))
  return there_is_dataset

class Extractor:
  def __init__(self,
               part_tree, point_range, location, comm, 
               equilibrate=False):
    self.part_tree     = part_tree
    self.exch_tool_box = dict()
    self.comm          = comm

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
      extract_zones, etb = extract_part_one_domain(part_zones, point_range[i_domain], self.dim, comm,
                                                    equilibrate=False)

      # > Retrive missing gnum
      '''
      Retrieve CellSize, CellRange and Face
      recover_dist_block_size -> CellSize
      CellSize + Cell --ijk_to_index--> CellRange
      CellSize + CellRange + Cell --create_zone_gnums--> Face
      '''
      if self.dim==3:
        from maia.factory.dist_from_part import _recover_dist_block_size
        cell_size = _recover_dist_block_size(extract_zones, comm)

        for zone in extract_zones:
          if PT.Zone.n_cell(zone)!=0:
            cell_gnum  = PT.maia.getGlobalNumbering(zone, 'Cell')[1]
            cell_ijk   = s_numbering.index_to_ijk(cell_gnum, cell_size[:,1])
            cell_range = np.array([[min(cell_ijk[0]),max(cell_ijk[0])],
                                   [min(cell_ijk[1]),max(cell_ijk[1])],
                                   [min(cell_ijk[2]),max(cell_ijk[2])]])
            cell_window = cell_range
            cell_window[:,1] +=1
            dist_cell_per_dir = cell_size[:,1]
            part_cell_per_dir = cell_window[:,1] - cell_window[:,0]
            dist_vtx_per_dir = dist_cell_per_dir+1
            dist_face_per_dir = n_face_per_dir(dist_vtx_per_dir, dist_cell_per_dir)
            part_face_per_dir = n_face_per_dir(part_cell_per_dir+1, part_cell_per_dir)
            shifted_nface_p = np_utils.sizes_to_indices(part_face_per_dir)
            ijk_to_faceIndex = [s_numbering.ijk_to_faceiIndex, s_numbering.ijk_to_facejIndex, s_numbering.ijk_to_facekIndex]
            face_lntogn = np.empty(shifted_nface_p[-1], dtype=pdm_gnum_dtype)
            for idir in range(3):
              i_ar  = np.arange(cell_window[0,0], cell_window[0,1]+(idir==0), dtype=pdm_gnum_dtype)
              j_ar  = np.arange(cell_window[1,0], cell_window[1,1]+(idir==1), dtype=pdm_gnum_dtype).reshape(-1,1)
              k_ar  = np.arange(cell_window[2,0], cell_window[2,1]+(idir==2), dtype=pdm_gnum_dtype).reshape(-1,1,1)
              face_lntogn[shifted_nface_p[idir]:shifted_nface_p[idir+1]] = ijk_to_faceIndex[idir](i_ar, j_ar, k_ar, \
                  dist_cell_per_dir, dist_vtx_per_dir).flatten()
            
            gn_node = PT.maia.getGlobalNumbering(zone)
            PT.new_node("CellRange", "IndexRange_t", cell_range, parent=gn_node)
            PT.new_DataArray("CellSize", cell_size[:,1], parent=gn_node)
            PT.new_DataArray("Face", face_lntogn, parent=gn_node)

      self.exch_tool_box.update(etb)
      for zone in extract_zones:
        if PT.Zone.n_vtx(zone)!=0:
          PT.add_child(extract_base, zone)

      # > Clean orphan GC
      all_zone_name_l = PT.get_names(PT.get_children_from_label(extract_base, 'Zone_t'))
      # for zone_n in extract_zones:
      #   for zgc_n in PT.get_children_from_label(zone_n, 'ZoneGridConnectivity_t'):
      #     for gc_n in PT.get_children_from_label(zgc_n, 'GridConnectivity1to1_t'):
      #       all_gc_name_l.append(PT.get_name(gc_n))
      all_zone_name_l = comm.allgather(all_zone_name_l)
      all_zone_name = list(np.concatenate(all_zone_name_l))
      print(f'all_zone_name = {all_zone_name}')
      # all_gc_name_all = {k: v for d in extract_pr_min_per_pzone_l for k, v in d.items()}


      for zone_n in PT.get_children_from_label(extract_base, 'Zone_t'):
        for zgc_n in PT.get_children_from_label(zone_n, 'ZoneGridConnectivity_t'):
          for gc_n in PT.get_children_from_label(zgc_n, 'GridConnectivity1to1_t'):
            # matching_zone_name = PT.get_value(gc_n)
            # if PT.get_child_from_name_and_label(extract_base, matching_zone_name, 'Zone_t') is None:
            matching_zone_name = PT.get_value(gc_n)
            # print(f'{matching_zone_name} {matching_zone_name not in all_gc_name} ')
            if matching_zone_name not in all_zone_name:
              print(f'removing {matching_zone_name}')
              PT.rm_child(zgc_n, gc_n)
          if len(PT.get_children_from_label(zgc_n, 'GridConnectivity1to1_t'))==0:
            PT.rm_child(zone_n, zgc_n)
        # PT.print_tree(zone_n)
    self.extract_tree = extract_tree
  
  def exchange_fields(self, fs_container):
    # Get zones by domains (only one domain for now)
    part_tree_per_dom = dist_from_part.get_parts_per_blocks(self.part_tree, self.comm).values()

    # Get zone from extractpart -> Assume unique domain
    extract_zones = PT.get_all_Zone_t(self.extract_tree)

    for container_name in fs_container:
      for i_domain, part_zones in enumerate(part_tree_per_dom):
        exchange_field_one_domain(self.part_tree, extract_zones, self.dim, self.exch_tool_box,\
            container_name, self.comm)

  def get_extract_part_tree(self):
    return self.extract_tree


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
    part1_pr, part1_gnum1, part1_in_part2 = build_intersection_numbering(part_tree, extract_zones, container_name, grid_location, etb, comm)

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
  
  if len(partial_gnum_vtx)!=0:
    for i_part, extract_zone in enumerate(extract_zones):
      PT.maia.newGlobalNumbering({'Vertex' : partial_gnum_vtx [i_part],
                                  'Cell'   : partial_gnum_cell[i_part]},
                                 parent=extract_zone)
  else:
    assert len(partial_gnum_cell)!=0

  # return extract_zones, exch_tool_box
  return extract_zones,etb


def extract_part_s_from_zsr(part_tree, zsr_name, comm,
                          transfer_dataset=True,
                          containers_name=[], **options):
  """Extract the submesh defined by the provided ZoneSubRegion from the input volumic
  partitioned tree.

  Dimension of the output mesh is set up accordingly to the GridLocation of the ZoneSubRegion.
  Submesh is returned as an independant partitioned CGNSTree and includes the relevant connectivities.

  Fields found under the ZSR node are transfered to the extracted mesh if ``transfer_dataset`` is set to True.
  In addition, additional containers specified in ``containers_name`` list are transfered to the extracted tree.
  Containers to be transfered can be either of label FlowSolution_t or ZoneSubRegion_t.

  Args:
    part_tree       (CGNSTree)    : Partitioned tree from which extraction is computed. Only U-NGon
      connectivities are managed.
    zsr_name        (str)         : Name of the ZoneSubRegion_t node
    comm            (MPIComm)     : MPI communicator
    transfer_dataset(bool)        : Transfer (or not) fields stored in ZSR to the extracted mesh (default to ``True``)
    containers_name (list of str) : List of the names of the fields containers to transfer
                                    on the output extracted tree.
    **options: Options related to the extraction.
  Returns:
    extracted_tree (CGNSTree)  : Extracted submesh (partitioned)

  Extraction can be controled by the optional kwargs:

    - ``graph_part_tool`` (str) -- Partitioning tool used to balance the extracted zones.
      Admissible values are ``hilbert, parmetis, ptscotch``. Note that
      vertex-located extractions require hilbert partitioning. Defaults to ``hilbert``.
  
  Important:
    - Input tree must be unstructured and have a ngon connectivity.
    - Partitions must come from a single initial domain on input tree.
  
  See also:
    :func:`create_extractor_from_zsr` takes the same parameters, excepted ``containers_name``,
    and returns an Extractor object which can be used to exchange containers more than once through its
    ``Extractor.exchange_fields(container_name)`` method.
  
  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #extract_from_zsr@start
      :end-before:  #extract_from_zsr@end
      :dedent: 2
  """

  start = time.time()
  extractor = create_extractor_from_zsr(part_tree, zsr_name, comm, **options)

  l_containers_name = [name for name in containers_name]
  if transfer_dataset and zsr_name not in l_containers_name:
    l_containers_name += [zsr_name]
  if l_containers_name:
    extractor.exchange_fields(l_containers_name)
  end = time.time()

  extracted_tree = extractor.get_extract_part_tree()

  # Print some light stats
  elts_kind = ['vtx', 'edges', 'faces', 'cells'][extractor.dim]
  if extractor.dim == 0:
    n_cell = sum([PT.Zone.n_vtx(zone) for zone in PT.iter_all_Zone_t(extracted_tree)])
  else:
    n_cell = sum([PT.Zone.n_cell(zone) for zone in PT.iter_all_Zone_t(extracted_tree)])
  n_cell_all = comm.allreduce(n_cell, MPI.SUM)
  mlog.info(f"Extraction from ZoneSubRegion \"{zsr_name}\" completed ({end-start:.2f} s) -- "
            f"Extracted tree has locally {mlog.size_to_str(n_cell)} {elts_kind} "
            f"(Σ={mlog.size_to_str(n_cell_all)})")

  return extracted_tree


def create_extractor_from_zsr(part_tree, zsr_path, comm, **options):
  """Same as extract_part_from_zsr, but return the extractor object."""
  # Get zones by domains

  part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm)

  # Get range for each partitioned zone and group it by domain
  point_range = list()
  location = ''
  for domain, part_zones in part_tree_per_dom.items():
    point_range_domain = list()
    for part_zone in part_zones:
      zsr_node = PT.get_node_from_path(part_zone, zsr_path)
      if zsr_node is not None:
        related_node = PT.getSubregionExtent(zsr_node, part_zone)
        zsr_node     = PT.get_node_from_path(part_zone, related_node)
        point_range_domain.append(PT.get_child_from_name(zsr_node, "PointRange")[1])
        location = PT.Subset.GridLocation(zsr_node)
      else:
        point_range_domain.append(np.empty(0, np.int32))
    point_range.append(point_range_domain)

  # Get location if proc has no zsr
  location = comm.allreduce(location, op=MPI.MAX)

  return Extractor(part_tree, point_range, location, comm)


def extract_part_s_from_bc_name(part_tree, bc_name, comm,
                                transfer_dataset=True,
                                containers_name=[],
                                **options):
  """Extract the submesh defined by the provided BC name from the input volumic
  partitioned tree.

  Behaviour and arguments of this function are similar to those of :func:`extract_part_from_zsr`:
  ``zsr_name`` becomes ``bc_name`` and optional ``transfer_dataset`` argument allows to 
  transfer BCDataSet from BC to the extracted mesh (default to ``True``).

  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #extract_from_bc_name@start
      :end-before:  #extract_from_bc_name@end
      :dedent: 2
  """

  # > Define extracted tree
  l_containers_name = [name for name in containers_name]
  local_part_tree   = PT.shallow_copy(part_tree)
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(local_part_tree, comm)
  
  # Adding ZSR to tree
  there_is_bcdataset = False
  for domain, part_zones in part_tree_per_dom.items():
    for part_zone in part_zones:
      bc_n = PT.get_node_from_name_and_label(part_zone, bc_name, 'BC_t') 
      if bc_n is not None:
        zsr_bc_n  = PT.new_ZoneSubRegion(name=bc_name, bc_name=bc_name, parent=part_zone)
        if transfer_dataset:
          there_is_bcdataset = set_transfer_dataset(bc_n, zsr_bc_n)

  if transfer_dataset and comm.allreduce(there_is_bcdataset, MPI.LOR):
    l_containers_name.append(bc_name) # not to change the initial containers_name list

  return extract_part_s_from_zsr(local_part_tree, bc_name, comm,
                                 transfer_dataset=False,
                                 containers_name=l_containers_name,
                               **options)