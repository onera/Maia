import numpy              as np
import Converter.Internal as I
import Pypdm.Pypdm        as PDM

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype

from maia.sids import sids     as SIDS
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

