import numpy              as np
import Converter.Internal as I
import Pypdm.Pypdm        as PDM

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype

from maia.sids  import sids     as SIDS
from maia.utils import py_utils
from maia.utils.parallel import utils    as par_utils
from maia.tree_exchange  import utils    as te_utils

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
    n_elems[i_zone] = I.getNodeFromName1(node, 'PointList')[1].shape[1] if node else 0

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
      distri_ud = I.createUniqueChild(node, ':CGNS#GlobalNumbering', 'UserDefinedData_t')
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
      if SIDS.GridLocation(node) == 'Vertex':
        ln_to_gn = I.getNodeFromPath(p_zone, ':CGNS#GlobalNumbering/Vertex')[1]
      else:
        ln_to_gn = te_utils.create_all_elt_g_numbering(p_zone, I.getNodesFromType1(dist_zone, 'Elements_t'))
      part_pl = I.getNodeFromName1(node, 'PointList')[1][0]
      ln_to_gn_list.append(ln_to_gn[part_pl-1])

  #Exchange is not needed. We use PTB just to count the element without multiplicity
  PTB = PDM.PartToBlock(comm, ln_to_gn_list, pWeight=None, partN=len(ln_to_gn_list),
                        t_distrib=0, t_post=1, t_stride=0)

  # Exchange size of filtered gnum and shift to create a create global numbering
  blk_distri = PTB.getDistributionCopy()
  blk_gnum   = PTB.getBlockGnumCopy()
  blk_n_elem = len(blk_gnum)
  blk_offset = par_utils.gather_and_shift(blk_n_elem, comm, dtype=np.int32)
  group_gnum = np.arange(blk_n_elem, dtype=pdm_gnum_dtype)+blk_offset[i_rank]+1

  # Now send this back to partitions. Caution, We have to use a variable stride 
  # (1 if gnum is know; 0 elsewhere). With variable stride exchange2 seems simpler
  dist_data = {'lngn' : group_gnum}
  blk_stride = np.zeros(blk_distri[i_rank+1] - blk_distri[i_rank], dtype=np.int32)
  blk_stride[blk_gnum - blk_distri[i_rank] - 1] = 1
  part_data = dict()

  BTP = PDM.BlockToPart(blk_distri, comm, ln_to_gn_list, len(ln_to_gn_list))
  BTP.BlockToPart_Exchange2(dist_data, part_data, 1, blk_stride)

  #Add in partitioned zones
  i_zone = 0
  for p_zone in part_zones:
    node = I.getNodeFromPath(p_zone, node_path)
    if node:
      distri_ud = I.createUniqueChild(node, ':CGNS#GlobalNumbering', 'UserDefinedData_t')
      I.newDataArray('Index', part_data['lngn'][i_zone], parent=distri_ud)
      i_zone += 1

def part_pl_to_dist_pl(dist_zone, part_zones, node_path, comm):
  """
  Create a distributed point list for the node specified by its node_path
  from the partitioned point lists.
  Numbering :CGNS#GlobalNumbering/Index on node_path must have been created before.
  In addition, we assume that node_path exists in dist_tree and that its location is already set
  """
  dist_node = I.getNodeFromPath(dist_zone, node_path)

  ln_to_gn_list = te_utils.collect_cgns_g_numbering(part_zones, node_path + '/:CGNS#GlobalNumbering/Index')
  PTB = PDM.PartToBlock(comm, ln_to_gn_list, pWeight=None, partN=len(ln_to_gn_list),
                        t_distrib=0, t_post=1, t_stride=0)

  dist_data = dict()
  part_data = {'pl' : []}
  for part_zone in part_zones:
    node = I.getNodeFromPath(part_zone, node_path)
    if node:
      if SIDS.GridLocation(node) == 'Vertex':
        ln_to_gn = I.getNodeFromPath(part_zone, ':CGNS#GlobalNumbering/Vertex')[1]
      else:
        ln_to_gn = te_utils.create_all_elt_g_numbering(part_zone, I.getNodesFromType1(dist_zone, 'Elements_t'))
      part_pl = I.getNodeFromName1(node, 'PointList')[1][0]
      part_data['pl'].append(ln_to_gn[part_pl-1])
    else:
      part_data['pl'].append(np.empty(0, dtype=pdm_gnum_dtype))

  PTB.PartToBlock_Exchange(dist_data, part_data)

  # Add distribution in dist_node
  distri_ud = I.createUniqueChild(dist_node, ':CGNS#Distribution', 'UserDefinedData_t')
  full_distri    = PTB.getDistributionCopy()
  partial_distri = np.empty(3, dtype=pdm_gnum_dtype)
  partial_distri[:2] = full_distri[comm.Get_rank() : comm.Get_rank()+2]
  partial_distri[2]  = full_distri[-1]
  I.newDataArray('Index', partial_distri, parent=distri_ud)

  # Create dist pointlist
  I.newPointList(value = dist_data['pl'].reshape(1,-1), parent=dist_node)
  I.newIndexArray('PointList#Size', value=[1, partial_distri[2]], parent=dist_node)


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
  elt_gnum_path = elem_name + '/:CGNS#GlobalNumbering/Element'
  vtx_gnum_l  = te_utils.collect_cgns_g_numbering(part_zones, ':CGNS#GlobalNumbering/Vertex')
  cell_gnum_l = te_utils.collect_cgns_g_numbering(part_zones, ':CGNS#GlobalNumbering/Cell')
  elt_gnum_l  = te_utils.collect_cgns_g_numbering(part_zones, elt_gnum_path)

  # Init dicts
  p_data_pe = {'PE' : list()}
  p_data_ec = {'Connectivity' : list()}
  p_strid_ec = list()
  p_strid_pe = list()

  # Collect partitioned data
  for ipart, part_zone in enumerate(part_zones):
    elem_n = I.getNodeFromName1(part_zone, elem_name)
    PE     = I.getNodeFromName1(elem_n, 'ParentElements')[1]
    EC     = I.getNodeFromName1(elem_n, 'ElementConnectivity')[1]
    ECIdx  = I.getNodeFromName1(elem_n, 'ElementStartOffset')[1]

    # Convert in global numbering and expected shape
    PE = PE.ravel()
    EC = vtx_gnum_l[ipart][EC-1]

    internal_cells = np.where(PE != 0)[0]
    internal_cells_lids = PE[internal_cells]
    PE[internal_cells] = cell_gnum_l[ipart][internal_cells_lids-1]

    p_data_pe['PE'].append(PE)
    p_data_ec['Connectivity'].append(EC)
    p_strid_pe.append(2*np.ones(PE.shape[0], dtype=np.int32))
    p_strid_ec.append(np.diff(ECIdx).astype(np.int32))

  # Init PTB protocol
  PTB = PDM.PartToBlock(comm, elt_gnum_l, None, len(elt_gnum_l),
                        t_distrib = 0, t_post = 2, t_stride = 1)
  PTBDistribution = PTB.getDistributionCopy()

  # Two echanges are needed, one for PE (with stride == 2), one for connectivity
  d_data_ec = dict()
  d_data_pe  = dict()
  PTB.PartToBlock_Exchange(d_data_pe, p_data_pe, p_strid_pe)
  PTB.PartToBlock_Exchange(d_data_ec, p_data_ec, p_strid_ec)

  # Post treat : delete duplicated faces. We chose to keep the first appearing
  d_strid_pe = d_data_pe['PE#Stride']
  dn_elt = d_strid_pe.shape[0]
  duplicated_idx = np.where(d_strid_pe != 2)[0]

  dist_pe = np.empty([dn_elt, 2], order='F', dtype=np.int32)
  offset = 0
  for iFace in range(dn_elt):
    # Face was not shared with a second partition on this zone
    if d_strid_pe[iFace] == 2:
      dist_pe[iFace,:] = d_data_pe['PE'][offset:offset+2]
    # Face was a partition boundary -> we take the left cell of each received tuple
    elif d_strid_pe[iFace] == 4:
      dist_pe[iFace,:] = [d_data_pe['PE'][offset], d_data_pe['PE'][offset+2]]
    else:
      raise RuntimeError("Something went wrong with face", iFace)
    offset += d_strid_pe[iFace]

  d_elt_n = d_data_ec['Connectivity#Stride']
  # Local elementStartOffset, but with duplicated face->vertex connectivity
  unfiltered_eso = py_utils.sizes_to_indices(d_data_ec['Connectivity#Stride'])

  # Array of bool (1d) indicating which indices of connectivity must be keeped
  # Then we just have to extract the good indices
  duplicated_ec = np.zeros(unfiltered_eso[dn_elt], dtype=np.bool)
  wrong_idx = py_utils.multi_arange(unfiltered_eso[duplicated_idx] + (d_elt_n[duplicated_idx] // 2),
                                    unfiltered_eso[duplicated_idx+1])
  duplicated_ec[wrong_idx] = 1
  dist_ec = d_data_ec['Connectivity'][~duplicated_ec]

  #Now retrieve filtered ElementStartOffset using size and cumsum
  d_elt_n[duplicated_idx] = d_elt_n[duplicated_idx] // 2
  d_elt_eso = py_utils.sizes_to_indices(d_elt_n)

  #Local work is done, ElementStartOffset must now be shifted
  shift_eso = par_utils.gather_and_shift(d_elt_eso[-1], comm)
  d_elt_eso += shift_eso[i_rank]

  # > Add in disttree
  elt_node = I.newElements(elem_name, 'NGON', parent=dist_zone)
  n_faceTot = PTBDistribution[n_rank]
  I.newPointRange('ElementRange',        [1, n_faceTot], parent=elt_node)
  I.newDataArray ('ParentElements',      dist_pe,        parent=elt_node)
  I.newDataArray ('ElementConnectivity', dist_ec,        parent=elt_node)
  I.newDataArray ('ElementStartOffset',  d_elt_eso,      parent=elt_node)

  DistriFaceVtx = par_utils.gather_and_shift(dist_ec.shape[0], comm, pdm_gnum_dtype)
  distri_ud = I.createUniqueChild(elt_node, ':CGNS#Distribution', 'UserDefinedData_t')
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
  cell_gnum_l = te_utils.collect_cgns_g_numbering(part_zones, elem_name + '/:CGNS#GlobalNumbering/Element')
  ngon_gnum_l = te_utils.collect_cgns_g_numbering(part_zones, ngon_name + '/:CGNS#GlobalNumbering/Element')

  # Init dicts
  part_data   = {'Connectivity' : list()}
  part_stride = list()

  # Collect partitioned data
  for ipart, part_zone in enumerate(part_zones):
    nface_n = I.getNodeFromName1(part_zone, elem_name)
    EC     = I.getNodeFromName1(nface_n, 'ElementConnectivity')[1]
    ECIdx  = I.getNodeFromName1(nface_n, 'ElementStartOffset')[1]

    # Move to global and add in part_data
    part_data['Connectivity'].append(ngon_gnum_l[ipart][np.abs(EC)-1])
    part_stride.append(np.diff(ECIdx).astype(np.int32))

  # Exchange : we suppose that cell belong to only one part, so there is nothing to do
  PTB = PDM.PartToBlock(comm, cell_gnum_l, None, len(cell_gnum_l),
                        t_distrib = 0, t_post = 0, t_stride = 1)
  PTBDistribution = PTB.getDistributionCopy()

  dist_data = dict()
  PTB.PartToBlock_Exchange(dist_data, part_data, part_stride)

  dist_ec = dist_data['Connectivity']
  d_elt_n = dist_data['Connectivity#Stride']

  # ElementStartOffset must be shifted
  dist_eso = py_utils.sizes_to_indices(d_elt_n)
  shift_eso = par_utils.gather_and_shift(dist_eso[-1], comm)
  dist_eso += shift_eso[comm.Get_rank()]

  # > Add in disttree
  elt_node = I.newElements(elem_name, 'NFACE', parent=dist_zone)
  n_cellTot = PTBDistribution[n_rank]
  I.newPointRange('ElementRange',        [1, n_cellTot], parent=elt_node)
  I.newDataArray ('ElementConnectivity', dist_ec,        parent=elt_node)
  I.newDataArray ('ElementStartOffset',  dist_eso,       parent=elt_node)

  distri_cell_face = par_utils.gather_and_shift(dist_ec.shape[0], comm, pdm_gnum_dtype)
  distri_ud = I.createUniqueChild(elt_node, ':CGNS#Distribution', 'UserDefinedData_t')
  I.newDataArray('Element',             PTBDistribution[[i_rank, i_rank+1, n_rank]], parent=distri_ud)
  I.newDataArray('ElementConnectivity',distri_cell_face[[i_rank, i_rank+1, n_rank]], parent=distri_ud)
