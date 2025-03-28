from mpi4py import MPI
import numpy              as np

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.typing         import *
import maia.pytree       as PT
import maia.pytree.utils as PTu
import maia.pytree.maia  as MT

from maia.utils     import np_utils, par_utils, s_numbering
from maia.transfer  import utils     as te_utils
from maia.transfer  import protocols as EP

LOC_TO_GN = {'Vertex': 'Vertex', 'FaceCenter': 'Face', 'CellCenter': 'Cell',
             'IEdgeCenter': 'Edge', 'JEdgeCenter': 'Edge',
             'IFaceCenter': 'Face', 'JFaceCenter': 'Face', 'KFaceCenter': 'Face'}

def create_part_pl_gnum_unique(part_zones: CGNSPartTree, 
                               node_path: CGNSTree,
                               comm: MPIComm) -> None:
  """
  Create a global numbering index for a given node, assuming that entity in
  this node are not duplicated over partitions.
  Node must contain a partitioned pointList.
  The global numbering is just a shift over the partitions.
  """
  # Collect the size of PL for each part
  n_elems = np.empty(len(part_zones), dtype=np.int32)
  for i_zone, p_zone in enumerate(part_zones):
    node = PT.get_node_from_path(p_zone, node_path)
    n_elems[i_zone] = PT.get_child_from_name(node, 'PointList')[1].shape[1] if node else 0

  # Exchange
  shifted_part = par_utils.gather_and_shift(len(part_zones), comm, dtype=np.int32)
  size_per_part = np.empty(shifted_part[-1], dtype=np.int32)
  comm.Allgatherv(n_elems, [size_per_part, np.diff(shifted_part)]) 

  #Shift and deduce global numbering
  for i_zone, p_zone in enumerate(part_zones):
    node = PT.get_node_from_path(p_zone, node_path)
    if node:
      offset = shifted_part[comm.Get_rank()] + i_zone
      start = np.sum(size_per_part[:offset]) + 1
      distri_ud = MT.newGlobalNumbering(parent=node)
      PT.new_DataArray('Index', np.arange(start, start+size_per_part[offset], dtype=pdm_gnum_dtype), parent=distri_ud)

def create_part_pl_gnum(dist_zone: CGNSDistTree, 
                        part_zones: CGNSPartTree, 
                        node_path: CGNSTree,
                        comm: MPIComm) -> None:
  """
  Create a global numbering index for a given node, even if entity in
  this node appears in accross multiple partitions.
  Node must contain a partitioned pointList.
  """

  i_rank = comm.Get_rank()

  # Collect the part pointlist and move it back to global numbering
  ln_to_gn_list = list()
  for p_zone in part_zones:
    node = PT.get_node_from_path(p_zone, node_path)
    if node:
      if is_bcds := PT.get_label(node) == 'BCDataSet_t':
        bc_parent = PT.get_node_from_path(p_zone, PT.utils.path_head(node_path))
      location = PT.BCDataSet.GridLocation(node, bc_parent) if is_bcds else PT.Subset.GridLocation(node)
      if location == 'Vertex':
        ln_to_gn = PT.get_value(MT.getGlobalNumbering(p_zone, 'Vertex'))
      else:
        ln_to_gn = te_utils.create_all_elt_g_numbering(p_zone, PT.get_children_from_label(dist_zone, 'Elements_t'))
      part_pl = PT.get_child_from_name(node, 'PointList')[1][0]
      ln_to_gn_list.append(ln_to_gn[part_pl-1])

  blk_distri_f = par_utils.distribution_from_gnum(ln_to_gn_list, comm, full=True)

  #First count the element without multiplicity
  GI = EP.GlobalMultiIndexer(blk_distri_f, [gn-1 for gn in ln_to_gn_list], comm)
  mask = (GI.access_counts > 0)

  # Exchange size of filtered gnum and shift to create a create global numbering
  blk_n_elem = mask.sum()
  blk_offset = par_utils.gather_and_shift(blk_n_elem, comm, dtype=np.int32)
  group_gnum = np.arange(blk_n_elem, dtype=pdm_gnum_dtype)+blk_offset[i_rank]+1

  # Now send this back to partitions. Caution, We have to use a variable stride 
  # (1 if gnum is know; 0 elsewhere). With variable stride exchange2 seems simpler
  blk_stride = np.zeros(blk_distri_f[i_rank+1] - blk_distri_f[i_rank], dtype=int)
  blk_stride[mask] = 1

  data_out = GI.Take_v((blk_stride, group_gnum))
  part_lngn = [data[1] for data in data_out]

  #Add in partitioned zones
  i_zone = 0
  for p_zone in part_zones:
    node = PT.get_node_from_path(p_zone, node_path)
    if node:
      distri_ud = MT.newGlobalNumbering(parent=node)
      PT.new_DataArray('Index', part_lngn[i_zone], parent=distri_ud)
      i_zone += 1

def create_part_pr_gnum(dist_zone: CGNSDistTree, 
                        part_zones: CGNSPartTree,
                        node_path: CGNSTree, 
                        comm: MPIComm) -> None:
  """
  Create a global numbering index for a given node containing a partitioned point range
  """
  from maia.algo.part import point_cloud_utils

  ln_to_gn_list = list()
  idx_dim = PT.Zone.IndexDimension(dist_zone)
  for part_zone in part_zones:
    node = PT.get_node_from_path(part_zone, node_path)
    if node:
      loc = PT.Subset.GridLocation(node)
      if loc == 'FaceCenter':
        raise RuntimeError(f"Wrong location for node {node_path} (FaceCenter). Please use one of [IFaceCenter, JFaceCenter, KFaceCenter]")

      ln_to_gn_all = MT.getGlobalNumbering(part_zone, LOC_TO_GN[loc])[1]

      # Get entity local numbering as full list
      part_pr = PT.get_child_from_name(node, 'PointRange')[1]
      i_ar = np.arange(part_pr[0][0], part_pr[0][1]+1) #creation pointlist
      if idx_dim == 1:
        local_num = i_ar
      else:
        j_ar = np.arange(part_pr[1][0], part_pr[1][1]+1).reshape(-1,1)
        if idx_dim == 2:
          local_num = s_numbering.ij_to_index_from_loc(i_ar, j_ar, loc, PT.Zone.VertexSize(part_zone)).flatten()
        elif idx_dim == 3:
          k_ar = np.arange(part_pr[2][0], part_pr[2][1]+1).reshape(-1,1,1)
          local_num = s_numbering.ijk_to_index_from_loc(i_ar, j_ar, k_ar, loc, PT.Zone.VertexSize(part_zone)).flatten()

      ln_to_gn_list.append(ln_to_gn_all[local_num-1])

  index_gnum = point_cloud_utils.create_sub_numbering(ln_to_gn_list, comm)

  #Add in partitioned zones
  i_zone = 0
  for part_zone in part_zones:
    node = PT.get_node_from_path(part_zone, node_path)
    if node:
      MT.newGlobalNumbering({'Index': index_gnum[i_zone]}, parent=node)
      i_zone += 1

def part_pl_to_dist_pl(dist_zone: CGNSDistTree,
                       part_zones: CGNSPartTree,
                       node_path: CGNSTree, 
                       comm: MPIComm, 
                       allow_mult: bool = False) -> None:
  """
  Create a distributed point list for the node specified by its node_path
  from the partitioned point lists.
  Numbering :CGNS#GlobalNumbering/Index on node_path must have been created before.
  In addition, we assume that node_path exists in dist_tree and that its location is already set
  If allow_mult is True, leaf node of node_path is expanded search all partitioned leaf*. This can
  be usefull eg to merge splitted joins (match.0, match.1, ...)
  """
  ancestor, leaf = PTu.path_head(node_path), PTu.path_tail(node_path)
  dist_node = PT.get_node_from_path(dist_zone, node_path)

  if allow_mult:
    name_predicate = lambda n: MT.conv.get_split_prefix(PT.get_name(n)) == leaf
  else:
    name_predicate = lambda n: PT.get_name(n) == leaf
   
  if allow_mult:
    ln_to_gn_list = []
    for part_zone in part_zones:
      ancestor_n = part_zone if ancestor is None else PT.get_node_from_path(part_zone, ancestor)
      if ancestor_n is not None:
        ln_to_gn_list.extend([PT.get_value(MT.getGlobalNumbering(node, 'Index')) \
            for node in PT.get_children_from_predicate(ancestor_n, name_predicate)])
  else:
    gn_path = node_path + '/:CGNS#GlobalNumbering/Index'
    ln_to_gn_list = [PT.get_node_from_path(part_zone, gn_path)[1] for part_zone in part_zones \
        if PT.get_node_from_path(part_zone, gn_path) is not None]

  distri   = par_utils.distribution_from_gnum(ln_to_gn_list, comm)
  distri_f = par_utils.partial_to_full_distribution(distri, comm)
  GI = EP.GlobalMultiIndexer(distri_f, [gn-1 for gn in ln_to_gn_list], comm)

  idx_dim = 1 if PT.Zone.Type(dist_zone) == 'Unstructured' else dist_zone[1].shape[0]
  keys = ['pl_i', 'pl_j', 'pl_k'][:idx_dim]
  part_pl_list = {key: [] for key in keys}

  for part_zone in part_zones:
    ancestor_n = part_zone if ancestor is None else PT.get_node_from_path(part_zone, ancestor)
    if ancestor_n:
      for node in PT.iter_children_from_predicate(ancestor_n, name_predicate):
        part_pl = PT.get_child_from_name(node, 'PointList')[1]
        loc = PT.BCDataSet.GridLocation(node, ancestor_n) if PT.get_label(node) == 'BCDataSet_t' else PT.Subset.GridLocation(node)
        if PT.Zone.Type(part_zone) == 'Unstructured':
          if loc == 'Vertex':
            ln_to_gn = PT.get_value(MT.getGlobalNumbering(part_zone, 'Vertex'))
          else:
            ln_to_gn = te_utils.create_all_elt_g_numbering(part_zone, PT.get_children_from_label(dist_zone, 'Elements_t'))
          part_pl_list['pl_i'].append(ln_to_gn[part_pl[0]-1])
        else:
          ln_to_gn = MT.getGlobalNumbering(part_zone, LOC_TO_GN[loc])[1]
          ijk_glob = _part_triplet_to_dist_triplet(part_pl, loc, ln_to_gn, PT.Zone.VertexSize(part_zone), PT.Zone.VertexSize(dist_zone))
          for i, key in enumerate(keys):
            part_pl_list[key].append(ijk_glob[i])

  # Exchange and create dist pointlist
  dist_pl = []
  for key in keys: #This factorize U and S PL shapes
    dist_pl.append(GI.Put(part_pl_list[key]))
  dist_pl = np.asarray(dist_pl, order='F')
  pl = PT.new_IndexArray(value=dist_pl, parent=dist_node)
  assert pl[1].ndim == 2 and pl[1].shape[0] == idx_dim

  # Add distribution in dist_node
  MT.newDistribution({'Index' : distri}, parent=dist_node)


def _part_triplet_to_dist_triplet(
  ptriplet: List[int], 
  loc: str,
  ln_to_gn: Dict[int, int],
  pvtx_size: Tuple[int,int,int],
  dvtx_size: Tuple[int, int, int]) -> Tuple[Union[int,Any], Union[int, Any], Union[int, Any]]:
    
  """ Convert a structured partitioned (local) i,j,k triplet to the corresponding
  global triplet in the distributed block """
  pcell_size = tuple(k-1 for k in pvtx_size)
  dcell_size = tuple(k-1 for k in dvtx_size)
  idx_dim = ptriplet.size
  ptriplet = ptriplet.tolist() + [1] if idx_dim == 2 else ptriplet #Manage 2D
  if loc == 'Vertex':
    gnum = ln_to_gn[s_numbering.ijk_to_index(*ptriplet, pvtx_size)-1]
    dtriplet = s_numbering.index_to_ijk(gnum, dvtx_size)
  elif loc == 'CellCenter':
    gnum = ln_to_gn[s_numbering.ijk_to_index(*ptriplet, pcell_size)-1]
    dtriplet = s_numbering.index_to_ijk(gnum, dcell_size)
  elif loc == 'IFaceCenter':
    gnum = ln_to_gn[s_numbering.ijk_to_faceiIndex(*ptriplet, pcell_size, pvtx_size)-1]
    dtriplet = s_numbering.faceiIndex_to_ijk(gnum, dcell_size, dvtx_size)
  elif loc == 'JFaceCenter':
    gnum = ln_to_gn[s_numbering.ijk_to_facejIndex(*ptriplet, pcell_size, pvtx_size)-1]
    dtriplet = s_numbering.facejIndex_to_ijk(gnum, dcell_size, dvtx_size)
  elif loc == 'KFaceCenter':
    gnum = ln_to_gn[s_numbering.ijk_to_facekIndex(*ptriplet, pcell_size, pvtx_size)-1]
    dtriplet = s_numbering.facekIndex_to_ijk(gnum, dcell_size, dvtx_size)
  return dtriplet[:idx_dim] # Manage 2D

def part_pr_to_dist_pr(dist_zone, part_zones, node_path, comm, allow_mult=False):
  """
  Create a distributed point range for the node specified by its node_path
  from the partitioned point range. We assume that node_path exists in dist_tree. 
  If allow_mult is True, leaf node of node_path is expanded search all partitioned leaf*. This can
  be usefull eg to merge splitted joins (match.0, match.1, ...)
  """
  idx_dim = PT.Zone.IndexDimension(dist_zone)
  ancestor_n, leaf_n = PTu.path_head(node_path), PTu.path_tail(node_path)

  if allow_mult:
    name_predicate = lambda n: MT.conv.get_split_prefix(PT.get_name(n)) == leaf_n
  else:
    name_predicate = lambda n: PT.get_name(n) == leaf_n

  dist_node = PT.get_node_from_path(dist_zone, node_path)
  dist_vtx_size = PT.Zone.VertexSize(dist_zone)

  proc_bottom = list()
  proc_top = list()
  proc_permuted = np.zeros(idx_dim, bool)
  for part_zone in part_zones:
    part_vtx_size = PT.Zone.VertexSize(part_zone)

    ancestor_node = PT.get_node_from_path(part_zone, ancestor_n)
    part_nodes = PT.get_children_from_predicate(ancestor_node, name_predicate) if ancestor_node is not None else []
    
    for part_node in part_nodes:
      pr = PT.get_node_from_name(part_node, 'PointRange')[1].copy()
      loc = PT.Subset.GridLocation(part_node)

      # In order to get max/min properly, make PR increasing but register permuted dir
      permuted = pr[:,0] > pr[:,1]
      pr[permuted, 0], pr[permuted, 1] = pr[permuted, 1], pr[permuted, 0]
      proc_permuted = proc_permuted | permuted

      # Get the global triplet related to the min and max corners of the window
      ln_to_gn = MT.getGlobalNumbering(part_zone, LOC_TO_GN[loc])[1]
      proc_bottom.append(_part_triplet_to_dist_triplet(pr[:,0], loc, ln_to_gn, part_vtx_size, dist_vtx_size))
      proc_top.append(_part_triplet_to_dist_triplet(pr[:,1], loc, ln_to_gn, part_vtx_size, dist_vtx_size))

  all_bottom = comm.allgather(proc_bottom)
  all_top = comm.allgather(proc_top)
  all_bottom = [item for sublist in all_bottom for item in sublist]
  all_top = [item for sublist in all_top for item in sublist]

  # Select max and min corners to create PR
  dist_pr = np.empty((idx_dim,2), dtype=dist_zone[1].dtype, order='F')
  dist_pr[:,0] = min(all_bottom)
  dist_pr[:,1] = max(all_top)

  # Restore permuted dir
  glob_permuted = np.empty(idx_dim, dtype=bool)
  comm.Allreduce(proc_permuted, glob_permuted, op=MPI.LOR)
  dist_pr[glob_permuted, 0], dist_pr[glob_permuted, 1] = \
      dist_pr[glob_permuted, 1], dist_pr[glob_permuted, 0]

  # Compute distribution
  pr_size = (np.abs(dist_pr[:,1] - dist_pr[:,0]) + 1).prod()
  distri = par_utils.uniform_distribution(pr_size, comm)

  PT.new_IndexRange(value=dist_pr, parent=dist_node)
  MT.newDistribution({'Index' : distri}, parent=dist_node)

def part_elt_to_dist_elt(dist_zone, part_zones, elem_name, comm):
  """
  Create a distributed Elements_t node on the dist_zone from partitions.
  Partitions must have the global numbering informations.
  On the dist_zone, ElementRange of the created node are numbered per physical dimension
  and must be shifted afterward.
  """

  vtx_gnum_l  = te_utils.collect_cgns_g_numbering(part_zones, 'Vertex')
  elt_gnum_l  = te_utils.collect_cgns_g_numbering(part_zones, 'Element', elem_name)

  data_in_l = list()
  cst_stride = 0
  elt_id   = 0
  min_section_gn = np.iinfo(pdm_gnum_dtype).max
  max_section_gn = 0
  for ipart, part_zone in enumerate(part_zones):
    elt_n = PT.get_child_from_name(part_zone, elem_name)
    if elt_n is not None:
      elt_id = PT.Element.Type(elt_n)
      cst_stride = PT.Element.NVtx(elt_n)

      # Retrieve the ElementRange within the given dimension
      section_gnum = MT.getGlobalNumbering(elt_n, 'Sections')[1]
      min_section_gn = min(min_section_gn, np.min(section_gnum))
      max_section_gn = max(max_section_gn, np.max(section_gnum))

      # Move to global and add in part_data
      EC    = PT.get_child_from_name(elt_n, 'ElementConnectivity')[1]
      part_ec = vtx_gnum_l[ipart][EC-1]
      stride_in = cst_stride*np.ones(part_ec.size // cst_stride, int)
    else:
      part_ec = np.empty(0, pdm_gnum_dtype)
      stride_in = np.empty(0, int)

    data_in_l.append((stride_in, part_ec))

  #Get values for proc having no elt
  elt_id     = comm.allreduce(elt_id, MPI.MAX)
  min_section_gn = comm.allreduce(min_section_gn, MPI.MIN)
  max_section_gn = comm.allreduce(max_section_gn, MPI.MAX)

  # Exchange : for multiple elements (eg. BAR) we take the first received
  distri_elt   = par_utils.distribution_from_gnum(elt_gnum_l, comm)
  distri_elt_f = par_utils.partial_to_full_distribution(distri_elt, comm)

  GI = EP.GlobalMultiIndexer(distri_elt_f, [gn-1 for gn in elt_gnum_l], comm)

  # Faster than filtering, even if stride is constant
  _, dist_ec = GI.Put_v(data_in_l)

  # > Add in disttree
  elt_node = PT.new_Elements(elem_name, type=elt_id, erange=[min_section_gn, max_section_gn], econn=dist_ec, parent=dist_zone)

  MT.newDistribution({'Element' : distri_elt}, parent=elt_node)

def part_ngon_to_dist_ngon(dist_zone, part_zones, elem_name, comm):
  """
  Create a distributed Elements_t node for NGon on the dist_zone from partitions.
  Partitions must have the global numbering informations.
  On the dist_zone, ElementRange of the created NGon node will start at 1 and must
  be shifted afterward.
  """
  n_rank = comm.Get_size()
  i_rank = comm.Get_rank()
  # Prepare gnum lists
  vtx_gnum_l  = te_utils.collect_cgns_g_numbering(part_zones, 'Vertex')
  cell_gnum_l = te_utils.collect_cgns_g_numbering(part_zones, 'Cell')
  elt_gnum_l  = te_utils.collect_cgns_g_numbering(part_zones, 'Element', elem_name)

  # Init dicts
  p_data_pe = list()
  p_data_ec = list()
  p_strid_ec = list()
  p_strid_pe = list()

  has_pe = True

  # Collect partitioned data
  for ipart, part_zone in enumerate(part_zones):
    elem_n = PT.get_child_from_name(part_zone, elem_name)
    ER     = PT.get_child_from_name(elem_n, 'ElementRange')[1]
    EC     = PT.get_child_from_name(elem_n, 'ElementConnectivity')[1]
    ECIdx  = PT.get_child_from_name(elem_n, 'ElementStartOffset')[1]
    pe_n   = PT.get_child_from_name(elem_n, 'ParentElements')

    # Deal ElementConnectivity
    EC = vtx_gnum_l[ipart][EC-1]
    p_strid_ec.append(np.diff(ECIdx).astype(np.int32))
    p_data_ec.append(EC)

    # Deal PE if present
    if pe_n is not None:
      PE     = pe_n[1].ravel()

      internal_cells = np.where(PE != 0)[0]
      internal_cells_lids = PE[internal_cells]
      if ER[0] == 1:
        internal_cells_lids -= (PT.Element.Size(elem_n))
      PE[internal_cells] = cell_gnum_l[ipart][internal_cells_lids-1]

      p_strid_pe.append(2*np.ones(PE.shape[0]//2, dtype=np.int32))
      p_data_pe.append(PE)
    else:
      has_pe = False

  has_pe = comm.allreduce(has_pe, op=MPI.LAND)
  # Init PTB protocol
  # Note: If 3D ngon mesh without ParentElement, mesh must be coherent
  #       at partition interface (like preserve_orientation=True)
  #       Thats why we can merge face connectivity without problem
  PTB = EP.PartToBlock(None, elt_gnum_l, comm, keep_multiple=has_pe, legacy=True)
  PTBDistribution = PTB.getDistributionCopy()
  n_faceTot = PTBDistribution[n_rank]

  # Two echanges are needed, one for PE (with stride == 2), one for connectivity
  d_strid_ec, d_data_ec = PTB.exchange_field(p_data_ec, p_strid_ec)

  d_elt_n = d_strid_ec
  dist_ec = d_data_ec

  if has_pe:
    d_strid_pe, d_data_pe = PTB.exchange_field(p_data_pe, p_strid_pe)

    # Post treat : delete duplicated faces.
    dn_elt = d_strid_pe.shape[0]
    duplicated_idx = np.where(d_strid_pe != 2)[0]

    dist_pe = np.empty([dn_elt, 2], order='F', dtype=pdm_gnum_dtype)
    offset = 0
    for iFace in range(dn_elt):
      # Face was not shared with a second partition on this zone
      if d_strid_pe[iFace] == 2:
        dist_pe[iFace,:] = d_data_pe[offset:offset+2]
      # Face was a partition boundary -> we take the left cell of each received tuple
      elif d_strid_pe[iFace] == 4:
        if d_data_pe[offset] == 0: #Orientation was preserved and first cell was right
          dist_pe[iFace,0] = d_data_pe[offset+2]
          dist_pe[iFace,1] = d_data_pe[offset+1]
        else:
          if d_data_pe[offset+3] != 0: #Orientation was presered and first cell was left
            dist_pe[iFace,0] = d_data_pe[offset+0]
            dist_pe[iFace,1] = d_data_pe[offset+3]
          else: #Orientation was not preserved : take first coming
            dist_pe[iFace,0] = d_data_pe[offset+0]
            dist_pe[iFace,1] = d_data_pe[offset+2]
      else:
        raise RuntimeError("Something went wrong with face", iFace)
      offset += d_strid_pe[iFace]


    # Local elementStartOffset, but with duplicated face->vertex connectivity
    unfiltered_eso = np_utils.sizes_to_indices(d_strid_ec)
    # Array of bool (1d) indicating which indices of connectivity must be keeped
    # Then we just have to extract the good indices
    duplicated_ec = np.zeros(unfiltered_eso[dn_elt], dtype=bool)
    wrong_idx = np_utils.multi_arange(unfiltered_eso[duplicated_idx] + (d_elt_n[duplicated_idx] // 2),
                                      unfiltered_eso[duplicated_idx+1])
    duplicated_ec[wrong_idx] = 1
    dist_ec = d_data_ec[~duplicated_ec]

    d_elt_n[duplicated_idx] = d_elt_n[duplicated_idx] // 2

  #Now retrieve filtered ElementStartOffset using size and cumsum
  d_elt_eso = np_utils.sizes_to_indices(d_elt_n, pdm_gnum_dtype)

  #Local work is done, ElementStartOffset must now be shifted
  shift_eso = par_utils.gather_and_shift(d_elt_eso[-1], comm)
  d_elt_eso += shift_eso[i_rank]

  # > Add in disttree
  elt_range = np.array([1, n_faceTot], pdm_gnum_dtype)
  elt_node = PT.new_NGonElements(elem_name, erange=elt_range, eso=d_elt_eso, ec=dist_ec, parent=dist_zone)
  if has_pe:
    # Shift dist PE because we put NGon first
    np_utils.shift_nonzeros(dist_pe, n_faceTot)
    PT.new_DataArray('ParentElements', dist_pe, parent=elt_node)

  DistriFaceVtx = par_utils.gather_and_shift(dist_ec.shape[0], comm, pdm_gnum_dtype)
  distri_ud = MT.newDistribution(parent=elt_node)
  PT.new_DataArray('Element',           PTBDistribution[[i_rank, i_rank+1, n_rank]], parent=distri_ud)
  PT.new_DataArray('ElementConnectivity', DistriFaceVtx[[i_rank, i_rank+1, n_rank]], parent=distri_ud)

def part_nface_to_dist_nface(dist_zone, part_zones, elem_name, ngon_name, comm):
  """
  Create a distributed Elements_t node for NFace on the dist_zone from partitions.
  Partitions must have the global numbering informations.
  We assume that each NFace cell belongs to only one partition.
  On the dist_zone, ElementRange of the created NFace node will start at 1 and must
  be shifted afterward.
  """
  n_rank = comm.Get_size()
  i_rank = comm.Get_rank()
  # Prepare gnum lists
  cell_gnum_l = te_utils.collect_cgns_g_numbering(part_zones, 'Element', elem_name)
  ngon_gnum_l = te_utils.collect_cgns_g_numbering(part_zones, 'Element', ngon_name)

  # Init dicts
  part_data = list()

  # Collect partitioned data
  for ipart, part_zone in enumerate(part_zones):
    ngon_n  = PT.get_child_from_name(part_zone, ngon_name)
    ng_offset = PT.Element.Range(ngon_n)[0]
    nface_n = PT.get_child_from_name(part_zone, elem_name)
    EC     = PT.get_child_from_name(nface_n, 'ElementConnectivity')[1]
    ECIdx  = PT.get_child_from_name(nface_n, 'ElementStartOffset')[1]

    # Move to global and add in part_data
    EC_sign = np.sign(EC)
    part_data.append((np.diff(ECIdx),
                      EC_sign*ngon_gnum_l[ipart][np.abs(EC)-ng_offset]))

  # Exchange : we suppose that cell belong to only one part, so there is nothing to do
  distri_cell   = par_utils.distribution_from_gnum(cell_gnum_l, comm)
  distri_cell_f = par_utils.partial_to_full_distribution(distri_cell, comm)
  GI = EP.GlobalMultiIndexer(distri_cell_f, [gn-1 for gn in cell_gnum_l], comm)

  d_elt_n, dist_ec = GI.Put_v(part_data)

  # ElementStartOffset must be shifted
  dist_eso = np_utils.sizes_to_indices(d_elt_n, pdm_gnum_dtype)
  shift_eso = par_utils.gather_and_shift(dist_eso[-1], comm)
  dist_eso += shift_eso[comm.Get_rank()]

  # > Add in disttree
  n_cellTot = distri_cell[-1]
  elt_range = np.array([1, n_cellTot], dtype=pdm_gnum_dtype)
  elt_node = PT.new_NFaceElements(elem_name, erange=elt_range, eso=dist_eso, ec=dist_ec, parent=dist_zone)

  distri_cell_face = par_utils.dn_to_distribution(dist_ec.shape[0], comm)
  MT.newDistribution({'Element' : distri_cell, 'ElementConnectivity' : distri_cell_face}, parent=elt_node)