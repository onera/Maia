import numpy      as np

import Converter.Internal     as I
import maia.sids.Internal_ext as IE
import Pypdm.Pypdm as PDM

from maia.sids  import conventions as conv
import maia.tree_exchange.utils as te_utils
from maia.tree_exchange.part_to_dist import discover  as DIS
from maia.tree_exchange.part_to_dist import data_exchange  as PTB
from maia.tree_exchange.part_to_dist import index_exchange as IPTB

def match_jn_from_ordinals(dist_tree):
  """
  Retrieve for each original GridConnectivity_t node the donor zone and the opposite
  pointlist in the tree. This assume that index distribution was identical for two related
  gc nodes.
  For now, donorname does not include basename because of a solver failure.
  """
  ordinal_to_data = dict() #Will store zone name & PL for each GC
  gc_t_path = 'CGNSBase_t/Zone_t/ZoneGridConnectivity_t/GridConnectivity_t'

  for base,zone,zgc,gc in IE.getNodesWithParentsByMatching(dist_tree, gc_t_path):
    ordinal = I.getNodeFromName1(gc, 'Ordinal')[1][0]
    #ordinal_to_zname[ordinal] = I.getName(base) + '/' + I.getName(zone)
    ordinal_to_data[ordinal] = (I.getName(zone), I.getNodeFromName1(gc, 'PointList')[1])

  for base,zone,zgc,gc in IE.getNodesWithParentsByMatching(dist_tree, gc_t_path):
    ordinal_opp = I.getNodeFromName1(gc, 'OrdinalOpp')[1][0]
    donor_name, donor_pl = ordinal_to_data[ordinal_opp]
    I.setValue(gc, donor_name)
    I.newPointList('PointListDonor', donor_pl, parent=gc)

def disttree_from_parttree(part_tree, comm):
  """
  Regenerate a distributed tree from partitioned trees.
  Partitioned trees must include all GlobalNumbering data.
  For now only NGon/NFace trees are supported
  """
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  dist_tree = I.newCGNSTree()
  # > Discover partitioned zones to build dist_tree structure
  DIS.discover_nodes_from_matching(dist_tree, [part_tree], 'CGNSBase_t', comm, child_list=['Family_t'])
  DIS.discover_nodes_from_matching(dist_tree, [part_tree], 'CGNSBase_t/Zone_t', comm,\
      child_list = ['ZoneType_t'],
      merge_rule=lambda zpath : conv.get_part_prefix(zpath))

  for dist_base, dist_zone in IE.getNodesWithParentsByMatching(dist_tree, 'CGNSBase_t/Zone_t'):

    distri_ud = IE.newDistribution(parent=dist_zone)

    part_zones = te_utils.get_partitioned_zones(part_tree, I.getName(dist_base) + '/' + I.getName(dist_zone))

    # > Create vertex distribution and exchange vertex coordinates
    vtx_lngn_list = te_utils.collect_cgns_g_numbering(part_zones, 'Vertex')
    pdm_ptb = PDM.PartToBlock(comm, vtx_lngn_list, pWeight=None, partN=len(vtx_lngn_list),
                              t_distrib=0, t_post=1, t_stride=0)
    vtx_distri_pdm = pdm_ptb.getDistributionCopy()
    I.newDataArray('Vertex', vtx_distri_pdm[[i_rank, i_rank+1, n_rank]], parent=distri_ud)
    d_grid_co = I.newGridCoordinates('GridCoordinates', parent=dist_zone)
    for coord in ['CoordinateX', 'CoordinateY', 'CoordinateZ']:
      I.newDataArray(coord, parent=d_grid_co)
    PTB.part_coords_to_dist_coords(dist_zone, part_zones, comm)

    # > Create ngon and nface connectivities
    IPTB.part_ngon_to_dist_ngon(dist_zone, part_zones, 'NGonElements', comm)
    IPTB.part_nface_to_dist_nface(dist_zone, part_zones, 'NFaceElements', 'NGonElements', comm)

    # > Shift nface element_range and create all cell distri
    n_face_tot  = I.getNodeFromPath(dist_zone, 'NGonElements/ElementRange')[1][1]
    nface_range = I.getNodeFromPath(dist_zone, 'NFaceElements/ElementRange')[1]
    nface_range += n_face_tot

    cell_distri = I.getVal(IE.getDistribution(I.getNodeFromName(dist_zone, 'NFaceElements'), 'Element'))
    vtx_distri  = I.getVal(IE.getDistribution(dist_zone, 'Vertex'))
    I.newDataArray('Cell', cell_distri, parent=distri_ud)

    I.setValue(dist_zone, np.array([[vtx_distri[2], cell_distri[2], 0]], dtype=np.int32))

    # > BND and JNS
    bc_t_path = 'ZoneBC_t/BC_t'
    gc_t_path = ['ZoneGridConnectivity_t', lambda n: I.getType(n) == 'GridConnectivity_t' and not conv.is_intra_gc(I.getName(n))]

    # > Discover (skip GC created by partitioning)
    DIS.discover_nodes_from_matching(dist_zone, part_zones, bc_t_path, comm,
          child_list=['FamilyName_t', 'GridLocation_t'], get_value='all')
    # DIS.discover_nodes_from_matching(dist_zone, part_zones, gc_t_path, comm,
    #       child_list=['GridLocation_t', 'GridConnectivityProperty_t', 'Ordinal', 'OrdinalOpp'],
    #       merge_rule= lambda path: conv.get_split_prefix(path),
    #       skip_rule = lambda node: conv.is_intra_gc(I.getName(node)))
    DIS.discover_nodes_from_matching(dist_zone, part_zones, gc_t_path, comm,
          child_list=['GridLocation_t', 'GridConnectivityProperty_t', 'Ordinal', 'OrdinalOpp'],
          merge_rule= lambda path: conv.get_split_prefix(path))

    # > Index exchange
    for d_zbc, d_bc in IE.getNodesWithParentsByMatching(dist_zone, bc_t_path):
      IPTB.part_pl_to_dist_pl(dist_zone, part_zones, I.getName(d_zbc) + '/' + I.getName(d_bc), comm)
    for d_zgc, d_gc in IE.getNodesWithParentsByMatching(dist_zone, gc_t_path):
      IPTB.part_pl_to_dist_pl(dist_zone, part_zones, I.getName(d_zgc) + '/' + I.getName(d_gc), comm, True)

    # > Flow Solution and Discrete Data
    PTB.part_sol_to_dist_sol(dist_zone, part_zones, comm)

    # > Todo : BCDataSet

  match_jn_from_ordinals(dist_tree)

  return dist_tree

