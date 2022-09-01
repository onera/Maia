from mpi4py import MPI
import numpy              as np
import Converter.Internal as I
import Pypdm.Pypdm        as PDM

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.utils     import np_utils, par_utils
from maia.transfer  import utils    as te_utils

def create_part_pl_gnum_unique(part_zones, node_path, comm):
  """
  Create a global numbering index for a given node, assuming that entity in
  this node are not duplicated over partitions.
  Node must contain a partitioned pointList.
  The global numbering is just a shift over the partitions.
  """
  # Collect the size of PL for each part
  n_elems = np.empty(len(part_zones), dtype=np.int32)
  for i_zone, p_zone in enumerate(part_zones):
    node = I.getNodeFromPath(p_zone, node_path)
    n_elems[i_zone] = PT.get_child_from_name(node, 'PointList')[1].shape[1] if node else 0

  # Exchange
  shifted_part = par_utils.gather_and_shift(len(part_zones), comm, dtype=np.int32)
  size_per_part = np.empty(shifted_part[-1], dtype=np.int32)
  comm.Allgatherv(n_elems, [size_per_part, np.diff(shifted_part)]) 

  #Shift and deduce global numbering
  for i_zone, p_zone in enumerate(part_zones):
    node = I.getNodeFromPath(p_zone, node_path)
    if node:
      offset = shifted_part[comm.Get_rank()] + i_zone
      start = np.sum(size_per_part[:offset]) + 1
      distri_ud = MT.newGlobalNumbering(parent=node)
      I.newDataArray('Index', np.arange(start, start+size_per_part[offset], dtype=pdm_gnum_dtype), distri_ud)

def create_part_pl_gnum(dist_zone, part_zones, node_path, comm):
  """
  Create a global numbering index for a given node, even if entity in
  this node appears in accross multiple partitions.
  Node must contain a partitioned pointList.
  """

  i_rank = comm.Get_rank()

  # Collect the part pointlist and move it back to global numbering
  ln_to_gn_list = list()
  for p_zone in part_zones:
    node = I.getNodeFromPath(p_zone, node_path)
    if node:
      if PT.Subset.GridLocation(node) == 'Vertex':
        ln_to_gn = I.getVal(MT.getGlobalNumbering(p_zone, 'Vertex'))
      else:
        ln_to_gn = te_utils.create_all_elt_g_numbering(p_zone, PT.get_children_from_label(dist_zone, 'Elements_t'))
      part_pl = PT.get_child_from_name(node, 'PointList')[1][0]
      ln_to_gn_list.append(ln_to_gn[part_pl-1])

  #Exchange is not needed. We use PTB just to count the element without multiplicity
  PTB = PDM.PartToBlock(comm, ln_to_gn_list, pWeight=None, partN=len(ln_to_gn_list),
                        t_distrib=0, t_post=1)

  # Exchange size of filtered gnum and shift to create a create global numbering
  blk_distri = PTB.getDistributionCopy()
  blk_gnum   = PTB.getBlockGnumCopy()
  blk_n_elem = len(blk_gnum)
  blk_offset = par_utils.gather_and_shift(blk_n_elem, comm, dtype=np.int32)
  group_gnum = np.arange(blk_n_elem, dtype=pdm_gnum_dtype)+blk_offset[i_rank]+1

  # Now send this back to partitions. Caution, We have to use a variable stride 
  # (1 if gnum is know; 0 elsewhere). With variable stride exchange2 seems simpler
  blk_stride = np.zeros(blk_distri[i_rank+1] - blk_distri[i_rank], dtype=np.int32)
  blk_stride[blk_gnum - blk_distri[i_rank] - 1] = 1

  BTP = PDM.BlockToPart(blk_distri, comm, ln_to_gn_list, len(ln_to_gn_list))
  _, part_lngn = BTP.exchange_field(group_gnum, blk_stride)

  #Add in partitioned zones
  i_zone = 0
  for p_zone in part_zones:
    node = I.getNodeFromPath(p_zone, node_path)
    if node:
      distri_ud = MT.newGlobalNumbering(parent=node)
      I.newDataArray('Index', part_lngn[i_zone], parent=distri_ud)
      i_zone += 1

def part_pl_to_dist_pl(dist_zone, part_zones, node_path, comm, allow_mult=False):
  """
  Create a distributed point list for the node specified by its node_path
  from the partitioned point lists.
  Numbering :CGNS#GlobalNumbering/Index on node_path must have been created before.
  In addition, we assume that node_path exists in dist_tree and that its location is already set
  If allow_mult is True, leaf node of node_path is expanded search all partitioned leaf*. This can
  be usefull eg to merge splitted joins (match.0, match.1, ...)
  """
  ancestor, leaf = PT.path_head(node_path), PT.path_tail(node_path)
  dist_node = I.getNodeFromPath(dist_zone, node_path)

  if allow_mult:
    ln_to_gn_list = []
    for part_zone in part_zones:
      ancestor_n = part_zone if ancestor is None else I.getNodeFromPath(part_zone, ancestor)
      ln_to_gn_list.extend([I.getVal(MT.getGlobalNumbering(node, 'Index')) \
          for node in PT.get_children_from_name(ancestor_n, leaf+'*')])
  else:
    gn_path = node_path + '/:CGNS#GlobalNumbering/Index'
    ln_to_gn_list = [I.getNodeFromPath(part_zone, gn_path)[1] for part_zone in part_zones \
        if I.getNodeFromPath(part_zone, gn_path) is not None]

  PTB = PDM.PartToBlock(comm, ln_to_gn_list, pWeight=None, partN=len(ln_to_gn_list),
                        t_distrib=0, t_post=1)

  part_pl_list = []
  for part_zone in part_zones:
    ancestor_n = part_zone if ancestor is None else I.getNodeFromPath(part_zone, ancestor)
    if ancestor_n:
      name = leaf + '*' if allow_mult else leaf
      for node in PT.iter_children_from_name(ancestor_n, name):
        if PT.Subset.GridLocation(node) == 'Vertex':
          ln_to_gn = I.getVal(MT.getGlobalNumbering(part_zone, 'Vertex'))
        else:
          ln_to_gn = te_utils.create_all_elt_g_numbering(part_zone, PT.get_children_from_label(dist_zone, 'Elements_t'))
        part_pl = PT.get_child_from_name(node, 'PointList')[1][0]
        part_pl_list.append(ln_to_gn[part_pl-1])

  _, dist_pl = PTB.exchange_field(part_pl_list)

  # Add distribution in dist_node
  i_rank, n_rank = comm.Get_rank(), comm.Get_size()
  distri_ud = MT.newDistribution(parent=dist_node)
  full_distri    = PTB.getDistributionCopy()
  I.newDataArray('Index', full_distri[[i_rank, i_rank+1, n_rank]], parent=distri_ud)

  # Create dist pointlist
  I.newPointList(value = dist_pl.reshape(1,-1), parent=dist_node)

def part_elt_to_dist_elt(dist_zone, part_zones, elem_name, comm):
  """
  Create a distributed Elements_t node on the dist_zone from partitions.
  Partitions must have the global numbering informations.
  On the dist_zone, ElementRange of the created node are numbered per physical dimension
  and must be shifted afterward.
  """
  n_rank = comm.Get_size()
  i_rank = comm.Get_rank()
  vtx_gnum_l  = te_utils.collect_cgns_g_numbering(part_zones, 'Vertex')
  elt_gnum_l  = te_utils.collect_cgns_g_numbering(part_zones, 'Element', elem_name)

  part_ec   = list()
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
      part_ec.append(vtx_gnum_l[ipart][EC-1])
    else:
      part_ec.append(np.empty(0, np.int32))

  #Get values for proc having no elt
  cst_stride = comm.allreduce(cst_stride, MPI.MAX)
  elt_id     = comm.allreduce(elt_id, MPI.MAX)
  min_section_gn = comm.allreduce(min_section_gn, MPI.MIN)
  max_section_gn = comm.allreduce(max_section_gn, MPI.MAX)

  # Exchange : for multiple elements (eg. BAR) we take the first received
  PTB = PDM.PartToBlock(comm, elt_gnum_l, None, len(elt_gnum_l),
                        t_distrib = 0, t_post = 1)
  PTBDistribution = PTB.getDistributionCopy()

  _, dist_ec = PTB.exchange_field(part_ec, cst_stride)

  # > Add in disttree
  elt_node = I.createUniqueChild(dist_zone, elem_name, 'Elements_t', np.array([elt_id, 0], np.int32))
  I.newPointRange('ElementRange',        [min_section_gn, max_section_gn], parent=elt_node)
  I.newDataArray ('ElementConnectivity', dist_ec,        parent=elt_node)

  distri_elt = par_utils.full_to_partial_distribution(PTBDistribution, comm)
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

  # Collect partitioned data
  for ipart, part_zone in enumerate(part_zones):
    elem_n = PT.get_child_from_name(part_zone, elem_name)
    ER     = PT.get_child_from_name(elem_n, 'ElementRange')[1]
    PE     = PT.get_child_from_name(elem_n, 'ParentElements')[1]
    EC     = PT.get_child_from_name(elem_n, 'ElementConnectivity')[1]
    ECIdx  = PT.get_child_from_name(elem_n, 'ElementStartOffset')[1]

    # Convert in global numbering and expected shape
    PE = PE.ravel()
    EC = vtx_gnum_l[ipart][EC-1]

    internal_cells = np.where(PE != 0)[0]
    internal_cells_lids = PE[internal_cells]
    if ER[0] == 1:
      internal_cells_lids -= (PT.Element.Size(elem_n))
    PE[internal_cells] = cell_gnum_l[ipart][internal_cells_lids-1]

    p_data_pe.append(PE)
    p_data_ec.append(EC)
    p_strid_pe.append(2*np.ones(PE.shape[0]//2, dtype=np.int32))
    p_strid_ec.append(np.diff(ECIdx).astype(np.int32))

  # Init PTB protocol
  PTB = PDM.PartToBlock(comm, elt_gnum_l, None, len(elt_gnum_l),
                        t_distrib = 0, t_post = 2)
  PTBDistribution = PTB.getDistributionCopy()

  # Two echanges are needed, one for PE (with stride == 2), one for connectivity
  d_data_ec = dict()
  d_strid_pe, d_data_pe = PTB.exchange_field(p_data_pe, p_strid_pe)
  d_strid_ec, d_data_ec = PTB.exchange_field(p_data_ec, p_strid_ec)

  # Post treat : delete duplicated faces. We chose to keep the first appearing
  dn_elt = d_strid_pe.shape[0]
  duplicated_idx = np.where(d_strid_pe != 2)[0]

  dist_pe = np.empty([dn_elt, 2], order='F', dtype=np.int32)
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

  d_elt_n = d_strid_ec
  # Local elementStartOffset, but with duplicated face->vertex connectivity
  unfiltered_eso = np_utils.sizes_to_indices(d_strid_ec)

  # Array of bool (1d) indicating which indices of connectivity must be keeped
  # Then we just have to extract the good indices
  duplicated_ec = np.zeros(unfiltered_eso[dn_elt], dtype=bool)
  wrong_idx = np_utils.multi_arange(unfiltered_eso[duplicated_idx] + (d_elt_n[duplicated_idx] // 2),
                                    unfiltered_eso[duplicated_idx+1])
  duplicated_ec[wrong_idx] = 1
  dist_ec = d_data_ec[~duplicated_ec]

  #Now retrieve filtered ElementStartOffset using size and cumsum
  d_elt_n[duplicated_idx] = d_elt_n[duplicated_idx] // 2
  d_elt_eso = np_utils.sizes_to_indices(d_elt_n)

  #Local work is done, ElementStartOffset must now be shifted
  shift_eso = par_utils.gather_and_shift(d_elt_eso[-1], comm)
  d_elt_eso += shift_eso[i_rank]

  n_faceTot = PTBDistribution[n_rank]
  # Shift dist PE because we put NGon first
  np_utils.shift_nonzeros(dist_pe, n_faceTot)
  # > Add in disttree
  elt_node = I.newElements(elem_name, 'NGON', parent=dist_zone)
  I.newPointRange('ElementRange',        [1, n_faceTot], parent=elt_node)
  I.newDataArray ('ParentElements',      dist_pe,        parent=elt_node)
  I.newDataArray ('ElementConnectivity', dist_ec,        parent=elt_node)
  I.newDataArray ('ElementStartOffset',  d_elt_eso,      parent=elt_node)

  DistriFaceVtx = par_utils.gather_and_shift(dist_ec.shape[0], comm, pdm_gnum_dtype)
  distri_ud = MT.newDistribution(parent=elt_node)
  I.newDataArray('Element',           PTBDistribution[[i_rank, i_rank+1, n_rank]], parent=distri_ud)
  I.newDataArray('ElementConnectivity', DistriFaceVtx[[i_rank, i_rank+1, n_rank]], parent=distri_ud)

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
  part_ec   = list()
  part_stride = list()

  # Collect partitioned data
  for ipart, part_zone in enumerate(part_zones):
    nface_n = PT.get_child_from_name(part_zone, elem_name)
    EC     = PT.get_child_from_name(nface_n, 'ElementConnectivity')[1]
    ECIdx  = PT.get_child_from_name(nface_n, 'ElementStartOffset')[1]

    # Move to global and add in part_data
    EC_sign = np.sign(EC)
    part_ec.append(EC_sign*ngon_gnum_l[ipart][np.abs(EC)-1])
    part_stride.append(np.diff(ECIdx).astype(np.int32))

  # Exchange : we suppose that cell belong to only one part, so there is nothing to do
  PTB = PDM.PartToBlock(comm, cell_gnum_l, None, len(cell_gnum_l),
                        t_distrib = 0, t_post = 0)
  PTBDistribution = PTB.getDistributionCopy()

  d_elt_n, dist_ec = PTB.exchange_field(part_ec, part_stride)

  # ElementStartOffset must be shifted
  dist_eso = np_utils.sizes_to_indices(d_elt_n)
  shift_eso = par_utils.gather_and_shift(dist_eso[-1], comm)
  dist_eso += shift_eso[comm.Get_rank()]

  # > Add in disttree
  elt_node = I.newElements(elem_name, 'NFACE', parent=dist_zone)
  n_cellTot = PTBDistribution[n_rank]
  I.newPointRange('ElementRange',        [1, n_cellTot], parent=elt_node)
  I.newDataArray ('ElementConnectivity', dist_ec,        parent=elt_node)
  I.newDataArray ('ElementStartOffset',  dist_eso,       parent=elt_node)

  distri_cell_face = par_utils.gather_and_shift(dist_ec.shape[0], comm, pdm_gnum_dtype)
  distri_ud = MT.newDistribution(parent=elt_node)
  I.newDataArray('Element',             PTBDistribution[[i_rank, i_rank+1, n_rank]], parent=distri_ud)
  I.newDataArray('ElementConnectivity',distri_cell_face[[i_rank, i_rank+1, n_rank]], parent=distri_ud)
