import Converter.Internal as I
import maia.sids.sids as SIDS
import maia.sids.Internal_ext as IE
import numpy          as np
from maia.connectivity import connectivity_transform as CNT
from maia.sids  import elements_utils as EU
from maia.utils import py_utils
from maia       import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.distribution.distribution_function import create_distribution_node_from_distrib
from maia.utils.parallel                     import utils          as par_utils
from Pypdm.Pypdm import dconnectivity_to_extract_dconnectivity, compute_entity_distribution, part_dcoordinates_to_pcoordinates
from maia.tree_exchange.dist_to_part.index_exchange import collect_distributed_pl



def extract_zone_mesh_from_bcs(dist_zone, comm):
  """
  """
  distrib_vtx      = I.getVal(IE.getDistribution(dist_zone, 'Vertex'))
  distrib_cell     = I.getVal(IE.getDistribution(dist_zone, 'Cell'))

  # > Try to hook NGon
  found = False
  for elt in I.getNodesFromType1(dist_zone, 'Elements_t'):
    if SIDS.ElementType(elt) == 22:
      found    = True
      dface_vtx = I.getNodeFromName1(elt, 'ElementConnectivity')[1]
      ngon_pe   = I.getNodeFromName1(elt, 'ParentElements'     )[1]
      ngon_eso  = I.getNodeFromName1(elt, 'ElementStartOffset' )[1]

      distrib_face     = I.getVal(IE.getDistribution(elt, 'Element'))
      distrib_face_vtx = I.getVal(IE.getDistribution(elt, 'ElementConnectivity'))
  if not found :
    raise RuntimeError

  dn_vtx  = distrib_vtx [1] - distrib_vtx [0]

  if dn_vtx > 0:
    gridc_n    = I.getNodeFromName1(dist_zone, 'GridCoordinates')
    cx         = I.getNodeFromName1(gridc_n, 'CoordinateX')[1]
    cy         = I.getNodeFromName1(gridc_n, 'CoordinateY')[1]
    cz         = I.getNodeFromName1(gridc_n, 'CoordinateZ')[1]
    dvtx_coord = py_utils.interweave_arrays([cx,cy,cz])

  bc_point_lists = collect_distributed_pl(dist_zone, ['ZoneBC_t/BC_t'])
  delmt_bound_idx, delmt_bound = py_utils.concatenate_point_list(bc_point_lists, pdm_gnum_dtype)

  dn_face = distrib_face[1] - distrib_face[0]
  np_face_distrib = compute_entity_distribution(comm, dn_face)

  dface_vtx_idx = np.empty(  dn_face+1, dtype=np.int32     ) # Local index is int32bits
  CNT.compute_idx_local       (dface_vtx_idx, ngon_eso, distrib_face_vtx)

  np_extract_entity1_distrib, np_extract_entity2_distrib, np_dextract_entity1_entity2_idx, np_dextract_entity1_entity2, \
  np_dparent_entity1_g_num, np_dparent_entity2_g_num, np_entity1_old_to_new = dconnectivity_to_extract_dconnectivity(comm, delmt_bound, np_face_distrib, dface_vtx_idx, dface_vtx)


  print("np_extract_entity1_distrib      : ", np_extract_entity1_distrib      )
  print("np_extract_entity2_distrib      : ", np_extract_entity2_distrib      )
  print("np_dextract_entity1_entity2_idx : ", np_dextract_entity1_entity2_idx )
  print("np_dextract_entity1_entity2     : ", np_dextract_entity1_entity2     )
  print("np_dparent_entity1_g_num        : ", np_dparent_entity1_g_num        )
  print("np_dparent_entity2_g_num        : ", np_dparent_entity2_g_num        )
  print("np_entity1_old_to_new           : ", np_entity1_old_to_new           )

  np_vtx_distrib = compute_entity_distribution(comm, dn_vtx)

  l_pvtx_coord = part_dcoordinates_to_pcoordinates(comm, np_vtx_distrib, dvtx_coord, [np_dparent_entity2_g_num])

  print("l_pvtx_coord : ", l_pvtx_coord)

  n_rank = comm.Get_size()
  i_rank = comm.Get_rank()
  dn_extract_face = np_extract_entity1_distrib[i_rank+1] - np_extract_entity1_distrib[i_rank]
  dn_extract_vtx  = np_extract_entity2_distrib[i_rank+1] - np_extract_entity2_distrib[i_rank]

  # > Re-create zone
  dist_extract_zone = I.newZone(name  = I.getName(dist_zone),
                                zsize = [[np_extract_entity2_distrib[n_rank],np_extract_entity1_distrib[n_rank],0]],
                                ztype = 'Unstructured')


  distrib_facevtx = par_utils.gather_and_shift(np_dextract_entity1_entity2_idx[dn_extract_face], comm, pdm_gnum_dtype)

  # > Grid coordinates
  cx, cy, cz = CNT.interlaced_to_tuple_coords(l_pvtx_coord[0])
  grid_coord = I.newGridCoordinates(parent=dist_extract_zone)
  I.newDataArray('CoordinateX', cx, parent=grid_coord)
  I.newDataArray('CoordinateY', cy, parent=grid_coord)
  I.newDataArray('CoordinateZ', cz, parent=grid_coord)

  eso = distrib_facevtx[i_rank] + np_dextract_entity1_entity2_idx
  extract_ngon_n = I.newElements('NGonElements', 'NGON', erange = [1, np_extract_entity1_distrib[n_rank]], parent=dist_extract_zone)

  I.newDataArray('ElementConnectivity', np_dextract_entity1_entity2, parent=extract_ngon_n)
  I.newDataArray('ElementStartOffset' , eso                        , parent=extract_ngon_n)

  np_distrib_vtx     = np.array([np_extract_entity2_distrib[i_rank], np_extract_entity2_distrib[i_rank+1], np_extract_entity2_distrib[n_rank]]    )
  np_distrib_face    = np.array([np_extract_entity1_distrib[i_rank], np_extract_entity1_distrib[i_rank+1], np_extract_entity1_distrib[n_rank]]   )
  np_distrib_facevtx = np.array([distrib_facevtx           [i_rank], distrib_facevtx[i_rank+1]           , distrib_facevtx[n_rank]])

  create_distribution_node_from_distrib("Cell"               , dist_extract_zone, np_distrib_face   )
  create_distribution_node_from_distrib("Vertex"             , dist_extract_zone, np_distrib_vtx    )
  create_distribution_node_from_distrib("Element"            , extract_ngon_n   , np_distrib_face   )
  create_distribution_node_from_distrib("ElementConnectivity", extract_ngon_n   , np_distrib_facevtx)

  return dist_extract_zone

def extract_mesh_from_bcs(dist_tree, comm):
  """
  """
  all_zones = I.getZones(dist_tree)
  u_zones   = [zone for zone in all_zones if SIDS.Zone.Type(zone) == 'Unstructured']

  dist_extract_tree = I.newCGNSTree()
  dist_extract_base = I.newCGNSBase(parent=dist_extract_tree)
  for i_zone, zone in enumerate(u_zones):
    dist_extract_zone = extract_zone_mesh_from_bcs(zone, comm)
    I._addChild(dist_extract_base, dist_extract_zone)

  return dist_extract_tree
