import numpy      as np

import Converter.Internal as I
import Pypdm.Pypdm as PDM

from maia.utils import py_utils
import maia.tree_exchange.utils as te_utils
from maia.tree_exchange.part_to_dist import data_exchange  as PTB
from maia.tree_exchange.part_to_dist import index_exchange as IPTB

def match_jn_from_ordinals(dist_tree):
  # > Resolve joins now. Because the distribution is homogeneous, all the
  #   data are already present in tree
  ordinal_to_pl    = dict()
  ordinal_to_zname = dict()
  gc_t_path = 'CGNSBase_t/Zone_t/ZoneGridConnectivity_t/GridConnectivity_t'

  for base,zone,zgc,gc in py_utils.getNodesWithParentsFromTypePath(dist_tree, gc_t_path):
    ordinal = gc[1][0]
    ordinal_to_pl[ordinal]    = I.getNodeFromName1(gc, 'PointList')[1]
    #ordinal_to_zname[ordinal] = I.getName(base) + '/' + I.getName(zone)
    ordinal_to_zname[ordinal] = I.getName(zone)

  for base,zone,zgc,gc in py_utils.getNodesWithParentsFromTypePath(dist_tree, gc_t_path):
    ordinal_opp = gc[1][1]
    PLOpp = I.newPointList('PointListDonor', ordinal_to_pl[ordinal_opp])
    I.addChild(gc, PLOpp)
    I.setValue(gc, ordinal_to_zname[ordinal_opp])

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
    part_bcs = []
    for part_zone in part_zones:
      part_bcs.extend([(I.getName(node), 'FaceCenter', I.getValue(node), I.getNodeFromType1(node, 'FamilyName_t'))\
          for node in I.getNodesFromType(part_zone, 'BC_t')])
    dist_bcs = []
    treated_bcs = []
    d_zbc = I.newZoneBC(parent=dist_zone)
    for rank_bc_names in comm.allgather(part_bcs):
      for bc_name, bc_loc, bc_type, bc_fam in rank_bc_names:
        if not bc_name in treated_bcs:
          treated_bcs.append(bc_name)
          d_bc = I.newBC(bc_name, btype=bc_type, parent=d_zbc)
          if bc_fam is not None:
            I._addChild(d_bc, bc_fam)
          I.newGridLocation(bc_loc, parent=d_bc)
          IPTB.part_pl_to_dist_pl(dist_zone, part_zones, 'ZoneBC/' + bc_name, comm)

    # > JNS
    part_jns = []
    dist_jns = []
    d_zgc = I.newZoneGridConnectivity(parent=dist_zone)
    for part_zone in part_zones:
      for JN in I.getNodesFromType(part_zone, 'GridConnectivity_t'):
        if I.getNodeFromName1(JN, ':CGNS#GlobalNumbering') is not None:
          ordinals = [I.getNodeFromName1(JN, 'Ordinal')[1][0], I.getNodeFromName1(JN, 'OrdinalOpp')[1][0]]
          gc_prop = I.getNodeFromType1(JN, 'GridConnectivityProperty_t')
          part_jns.append((I.getName(JN), 'FaceCenter', ordinals, gc_prop))

    treated_jns = []
    for rank_jn_names in comm.allgather(part_jns):
      for jn_name, jn_loc, ordinals, gc_prop in rank_jn_names:
        full_name = '.'.join(jn_name.split('.')[:-1])
        if not full_name in treated_jns:
          treated_jns.append(full_name)
          #tpm use of donorName to store ordinals
          d_gc = I.newGridConnectivity(full_name, donorName=ordinals, ctype='Abbuting1to1', parent=d_zgc)
          I._addChild(d_gc, gc_prop)
          I.newGridLocation(jn_loc, parent=d_gc)
          IPTB.part_pl_to_dist_pl(dist_zone, part_zones, 'ZoneGridConnectivity/' + full_name, comm, True)
    if not treated_jns:
      I._rmNode(dist_zone, d_zgc)


    # > Flow Solution and Discrete Data
    PTB.part_sol_to_dist_sol(dist_zone, part_zones, comm)
    
    # > Todo : BCDataSet


  match_jn_from_ordinals(dist_tree)

  return dist_tree

