import numpy      as np

import Converter.Internal as I
import Pypdm.Pypdm as PDM

from maia.utils import py_utils
import maia.tree_exchange.utils as te_utils
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

  for base,zone,zgc,gc in py_utils.getNodesWithParentsFromTypePath(dist_tree, gc_t_path):
    ordinal = I.getNodeFromName1(gc, 'Ordinal')[1][0]
    #ordinal_to_zname[ordinal] = I.getName(base) + '/' + I.getName(zone)
    ordinal_to_data[ordinal] = (I.getName(zone), I.getNodeFromName1(gc, 'PointList')[1])

  for base,zone,zgc,gc in py_utils.getNodesWithParentsFromTypePath(dist_tree, gc_t_path):
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

  # > Discover partitioned zones to build dist_tree structure
  dist_zone_pathes = PTB.discover_partitioned_zones(part_tree, comm)

  dist_tree = I.newCGNSTree()
  for dist_zone_path in dist_zone_pathes:
    base_name, zone_name = dist_zone_path.split('/')

    # > Create base and zone in dist tree
    dist_base = I.createUniqueChild(dist_tree, base_name, 'CGNSBase_t', [3,3])
    dist_zone = I.newZone(zone_name, zsize=[[0,0,0]], ztype='Unstructured', parent=dist_base)
    distri_ud = I.createUniqueChild(dist_zone, ':CGNS#Distribution', 'UserDefinedData_t')

    part_zones = te_utils.get_partitioned_zones(part_tree, dist_zone_path)

    # > Create vertex distribution and exchange vertex coordinates
    vtx_lngn_list = te_utils.collect_cgns_g_numbering(part_zones, ':CGNS#GlobalNumbering/Vertex')
    pdm_ptb = PDM.PartToBlock(comm, vtx_lngn_list, pWeight=None, partN=len(vtx_lngn_list),
                              t_distrib=0, t_post=1, t_stride=0)
    vtx_distri = pdm_ptb.getDistributionCopy()
    I.newDataArray('Vertex', vtx_distri[[i_rank, i_rank+1, n_rank]], parent=distri_ud)
    dist_zone[1][0][0] = vtx_distri[2]
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

    cell_distri = I.getNodeFromPath(dist_zone, 'NFaceElements/:CGNS#Distribution/Element')[1]
    I.newDataArray('Cell', cell_distri, parent=distri_ud)
    dist_zone[1][0][1] = cell_distri[2]

    # > BND
    part_bcs = dict()
    bc_t_path = 'ZoneBC_t/BC_t'
    for part_zone in part_zones:
      for p_zbc, p_bc in py_utils.getNodesWithParentsFromTypePath(part_zone, bc_t_path):
        childs  = [I.getNodeFromType1(p_bc, 'FamilyName_t'),
                   I.getNodeFromType1(p_bc, 'GridLocation_t')]
        part_bcs[I.getName(p_zbc) + '/' + I.getName(p_bc)] = (I.getValue(p_bc), childs)

    for rank_bc_names in comm.allgather(part_bcs):
      for bc_path, (kind, childs) in rank_bc_names.items():
        zbc_name, bc_name = bc_path.split('/')
        d_zbc = I.createUniqueChild(dist_zone, zbc_name, 'ZoneBC_t')
        if I.getNodeFromName1(d_zbc, bc_name) is None:
          d_bc = I.newBC(bc_name, btype=kind, parent=d_zbc)
          for child in childs:
            I._addChild(d_bc, child)

    # > Index exchange
    for d_zbc, d_bc in py_utils.getNodesWithParentsFromTypePath(dist_zone, bc_t_path):
      IPTB.part_pl_to_dist_pl(dist_zone, part_zones, I.getName(d_zbc) + '/' + I.getName(d_bc), comm)

    # > JNS
    # > Discover
    part_jns = dict()
    gc_t_path = 'ZoneGridConnectivity_t/GridConnectivity_t'
    for part_zone in part_zones:
      for p_zgc, p_gc in py_utils.getNodesWithParentsFromTypePath(part_zone, gc_t_path):
        #Treat only original joins
        if I.getNodeFromName1(p_gc, ':CGNS#GlobalNumbering') is not None:
          childs  = [I.getNodeFromName1(p_gc, 'Ordinal'), I.getNodeFromName(p_gc, 'OrdinalOpp'),
                     I.getNodeFromType1(p_gc, 'GridLocation_t'),
                     I.getNodeFromType1(p_gc, 'GridConnectivityProperty_t')]
          #Splited join must be added only once
          full_name = I.getName(p_zgc) + '/' + '.'.join(I.getName(p_gc).split('.')[:-1])
          part_jns[full_name] = childs

    for rank_jn_names in comm.allgather(part_jns):
      for jn_path, childs in rank_jn_names.items():
        d_zgc_name, d_gc_name = jn_path.split('/')
        d_zgc = I.createUniqueChild(dist_zone, d_zgc_name, 'ZoneGridConnectivity_t')
        if I.getNodeFromName1(d_zgc, d_gc_name) is None:
          d_gc = I.newGridConnectivity(d_gc_name, ctype='Abbuting1to1', parent=d_zgc)
          for child in childs:
            I._addChild(d_gc, child)

    # > Index exchange
    for d_zgc, d_gc in py_utils.getNodesWithParentsFromTypePath(dist_zone, gc_t_path):
      IPTB.part_pl_to_dist_pl(dist_zone, part_zones, I.getName(d_zgc) + '/' + I.getName(d_gc), comm, True)


    # > Flow Solution and Discrete Data
    PTB.part_sol_to_dist_sol(dist_zone, part_zones, comm)
    
    # > Todo : BCDataSet

  match_jn_from_ordinals(dist_tree)

  return dist_tree

