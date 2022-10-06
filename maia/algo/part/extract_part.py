# =======================================================================================
# ---------------------------------------------------------------------------------------
import  numpy as np
# from    mpi4py import MPI

# MAIA
from    maia.utils                 import np_utils, layouts
from    maia.pytree.sids           import node_inspect       as sids
from    maia.pytree.maia           import conventions        as conv
import  maia.transfer.utils                                  as TEU
from    maia.transfer.part_to_dist import data_exchange      as PTD

# CASSIOPEE
import  Converter.Internal as I

# PARADIGM
import Pypdm.Pypdm as PDM
# ---------------------------------------------------------------------------------------
# =======================================================================================



# =======================================================================================
# ---------------------------------------------------------------------------------------
def extract_part_one_domain(part_zones, zsrpath, comm):
  """
  """
  n_part = len(part_zones)
  dim    = 3
  equilibrate = 0

  pdm_ep = PDM.ExtractPart(dim,
                           n_part,
                           1, # n_part_out
                           equilibrate,
                           PDM._PDM_SPLIT_DUAL_WITH_HILBERT,
                           comm)

  np_part1_cell_ln_to_gn = list()
  for i_part, part_zone in enumerate(part_zones):
    # Get NGon + NFac
    gridc_n    = I.getNodeFromName1(part_zone, 'GridCoordinates')
    cx         = I.getNodeFromName1(gridc_n, 'CoordinateX')[1]
    cy         = I.getNodeFromName1(gridc_n, 'CoordinateY')[1]
    cz         = I.getNodeFromName1(gridc_n, 'CoordinateZ')[1]
    vtx_coords = np_utils.interweave_arrays([cx,cy,cz])

    ngons  = [e for e in I.getNodesFromType1(part_zone, 'Elements_t') if sids.Element.CGNSName(e) == 'NGON_n']
    nfaces = [e for e in I.getNodesFromType1(part_zone, 'Elements_t') if sids.Element.CGNSName(e) == 'NFACE_n']
    assert len(nfaces) == len(ngons) == 1

    cell_face_idx = I.getNodeFromName1(nfaces[0], "ElementStartOffset")[1]
    cell_face     = I.getNodeFromName1(nfaces[0], "ElementConnectivity")[1]
    face_vtx_idx  = I.getNodeFromName1(ngons[0],  "ElementStartOffset")[1]
    face_vtx      = I.getNodeFromName1(ngons[0],  "ElementConnectivity")[1]

    vtx_ln_to_gn, face_ln_to_gn, cell_ln_to_gn = TEU.get_entities_numbering(part_zone)

    n_cell = cell_ln_to_gn.shape[0]
    n_face = face_ln_to_gn.shape[0]
    n_edge = 0
    n_vtx  = vtx_ln_to_gn .shape[0]

    pdm_ep.part_set(i_part,
                      n_cell,
                      n_face,
                      n_edge,
                      n_vtx,
                      cell_face_idx,
                      cell_face    ,
                      None,
                      None,
                      None,
                      face_vtx_idx ,
                      face_vtx     ,
                      cell_ln_to_gn,
                      face_ln_to_gn,
                      None,
                      vtx_ln_to_gn ,
                      vtx_coords)

    np_part1_cell_ln_to_gn.append(cell_ln_to_gn)

    zsr           = I.getNodeFromPath(part_zone, zsrpath)
    extract_l_num = I.getNodeFromName1(zsr, "PointList")

    pdm_ep.selected_lnum_set(i_part, extract_l_num[1])

  pdm_ep.compute()

  # > Reconstruction tu maillage d'iso
  n_extract_cell = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_CELL  )
  n_extract_face = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_FACE  )
  n_extract_edge = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_EDGE  )
  n_extract_vtx  = pdm_ep.n_entity_get(0, PDM._PDM_MESH_ENTITY_VERTEX)

  extract_vtx_coords = pdm_ep.vtx_coord_get(0)

  print("n_extract_cell : ", n_extract_cell)
  print("n_extract_face : ", n_extract_face)
  print("n_extract_edge : ", n_extract_edge)
  print("n_extract_vtx  : ", n_extract_vtx )

  extract_part_base = I.newCGNSBase('Base', cellDim=dim, physDim=3)

  if n_extract_cell == 0:
    extract_part_zone = I.newZone('zone', [[n_extract_vtx, n_extract_face, 0]],
                                  'Unstructured', parent=extract_part_base)

    ngon_n = I.newElements('NGonElements', 'NGON', erange = [1, n_extract_edge], parent=extract_part_zone)
    face_vtx_idx, face_vtx  = pdm_ep.connectivity_get(0, PDM._PDM_CONNECTIVITY_TYPE_EDGE_VTX)
    I.newDataArray('ElementConnectivity', face_vtx    , parent=ngon_n)
    I.newDataArray('ElementStartOffset' , face_vtx_idx, parent=ngon_n)

    nface_n = I.newElements('NFacElements', 'NFACE', erange = [n_extract_edge+1, n_extract_edge+n_extract_face], parent=extract_part_zone)
    cell_face_idx, cell_face = pdm_ep.connectivity_get(0, PDM._PDM_CONNECTIVITY_TYPE_FACE_VTX)
    I.newDataArray('ElementConnectivity', cell_face    , parent=nface_n)
    I.newDataArray('ElementStartOffset' , cell_face_idx, parent=nface_n)
  else:
    extract_part_zone = I.newZone('zone', [[n_extract_vtx, n_extract_cell, 0]],
                                  'Unstructured', parent=extract_part_base)

    ngon_n = I.newElements('NGonElements', 'NGON', erange = [1, n_extract_face], parent=extract_part_zone)
    face_vtx_idx, face_vtx  = pdm_ep.connectivity_get(0, PDM._PDM_CONNECTIVITY_TYPE_FACE_VTX)
    I.newDataArray('ElementConnectivity', face_vtx    , parent=ngon_n)
    I.newDataArray('ElementStartOffset' , face_vtx_idx, parent=ngon_n)

    nface_n = I.newElements('NFacElements', 'NFACE', erange = [n_extract_face+1, n_extract_face+n_extract_cell], parent=extract_part_zone)
    cell_face_idx, cell_face = pdm_ep.connectivity_get(0, PDM._PDM_CONNECTIVITY_TYPE_CELL_FACE)
    I.newDataArray('ElementConnectivity', cell_face    , parent=nface_n)
    I.newDataArray('ElementStartOffset' , cell_face_idx, parent=nface_n)

  # > Grid coordinates
  cx, cy, cz = layouts.interlaced_to_tuple_coords(extract_vtx_coords)
  extract_grid_coord = I.newGridCoordinates(parent=extract_part_zone)
  I.newDataArray('CoordinateX', cx, parent=extract_grid_coord)
  I.newDataArray('CoordinateY', cy, parent=extract_grid_coord)
  I.newDataArray('CoordinateZ', cz, parent=extract_grid_coord)

  np_part2_cell        = pdm_ep.ln_to_gn_get       (0, PDM._PDM_MESH_ENTITY_CELL  )
  np_part2_cell_parent = pdm_ep.parent_ln_to_gn_get(0, PDM._PDM_MESH_ENTITY_CELL  )

  print("np_part2_cell = ", np_part2_cell)
  print("np_part2_cell_parent = ", np_part2_cell_parent)

  part2_to_part1_idx = np.arange(0, np_part1_cell_ln_to_gn[0].shape[0], dtype=np.int32 )

  print("np_part1_cell_ln_to_gn ", np_part1_cell_ln_to_gn[0].shape[0])
  print("part2_to_part1_idx : ", part2_to_part1_idx.shape[0])

  # > On a la connectivity part2_to_part1 car part1 = les parents
  ptp = PDM.PartToPart(comm,
                       [np_part2_cell],
                       np_part1_cell_ln_to_gn,
                       [part2_to_part1_idx],
                       [np_part2_cell_parent])

  part2_stri = np.ones(np_part1_cell_ln_to_gn[0].shape[0], dtype=np.int32)
  part2_data = np_part1_cell_ln_to_gn

  # for data_array in I.getNodesFromType1(fs, "DataArray_t"):

  print(part2_stri)
  #￿> Stride cst
  # req_id = ptp.reverse_iexch(PDM._PDM_MPI_COMM_KIND_P2P,
  #                            PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART2,
  #                            np_part1_cell_ln_to_gn)
  #￿> Stride variable
  req_id = ptp.reverse_iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                             PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART2,
                             np_part1_cell_ln_to_gn,
                             part2_stride=[part2_stri])
  part1_strid, part1_data = ptp.reverse_wait(req_id)

  print("part1_strid :: ", part1_strid)
  print("part1_data  :: ", part1_data )

  return extract_part_base

def extract_part(part_tree, fspath, comm):
  """
  """
  dist_doms = I.newCGNSTree()
  PTD.discover_nodes_from_matching(dist_doms, [part_tree], 'CGNSBase_t/Zone_t', comm,
                                    merge_rule=lambda zpath : conv.get_part_prefix(zpath))

  part_tree_per_dom = list()
  # for base, zone in IE.getNodesWithParentsByMatching(dist_doms, ['CGNSBase_t', 'Zone_t']):
  for base in I.getNodesFromType(dist_doms,'CGNSBase_t'):
    for zone in I.getNodesFromType(dist_doms,'Zone_t'):
      part_tree_per_dom.append(TEU.get_partitioned_zones(part_tree, I.getName(base) + '/' + I.getName(zone)))

  extract_doms = I.newCGNSTree()
  for i_domain, part_zones in enumerate(part_tree_per_dom):
    extract_part = extract_part_one_domain(part_zones, fspath, comm)
    I._addChild(extract_doms, extract_part)


  return extract_doms
  
# ---------------------------------------------------------------------------------------
# =======================================================================================