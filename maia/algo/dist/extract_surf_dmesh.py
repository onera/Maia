import numpy          as np
from Pypdm.Pypdm import dconnectivity_to_extract_dconnectivity, part_dcoordinates_to_pcoordinates

import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia                      import npy_pdm_gnum_dtype     as pdm_gnum_dtype
from maia.utils                import np_utils, par_utils, layouts
from maia.transfer.dist_to_part.index_exchange import collect_distributed_pl


def _extract_faces(dist_zone, face_list, comm):
  """
  CGNS wrapping for Pypdm.dconnectivity_to_extract_dconnectivity : extract the faces
  specified in face_list from the distributed zone
  """

  # > Try to hook NGon
  ngon_node = PT.Zone.NGonNode(dist_zone)
  dface_vtx = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
  ngon_eso  = PT.get_child_from_name(ngon_node, 'ElementStartOffset' )[1]

  distrib_face     = I.getVal(MT.getDistribution(ngon_node, 'Element'))
  distrib_face_vtx = I.getVal(MT.getDistribution(ngon_node, 'ElementConnectivity'))

  dn_face = distrib_face[1] - distrib_face[0]
  np_face_distrib = par_utils.partial_to_full_distribution(distrib_face, comm)

  dface_vtx_idx = np.add(ngon_eso, -distrib_face_vtx[0], dtype=np.int32) #Local index is int32bits

  return dconnectivity_to_extract_dconnectivity(comm, face_list, np_face_distrib, dface_vtx_idx, dface_vtx)

def _extract_surf_zone(dist_zone, face_list, comm):
  """
  Extract the specified faces and create a CGNS 2d Zone containing only these faces.
  Extracted zone connectivity is described by a NGon Element node
  """
  n_rank = comm.Get_size()
  i_rank = comm.Get_rank()

  # > Extract  faces
  ex_face_distri, ex_vtx_distri, ex_face_vtx_idx, ex_face_vtx, ex_face_parent_gnum, ex_vtx_parent_gnum, \
    ex_face_old_to_new = _extract_faces(dist_zone, face_list, comm)

  # > Transfert extracted vertex coordinates
  distrib_vtx      = I.getVal(MT.getDistribution(dist_zone, 'Vertex'))
  dn_vtx  = distrib_vtx [1] - distrib_vtx [0]
  if dn_vtx > 0:
    cx, cy, cz = PT.Zone.coordinates(dist_zone)
    dvtx_coord = np_utils.interweave_arrays([cx,cy,cz])


  ini_vtx_distri = par_utils.partial_to_full_distribution(distrib_vtx, comm)
  ex_vtx_coords  = part_dcoordinates_to_pcoordinates(comm, ini_vtx_distri, dvtx_coord, [ex_vtx_parent_gnum])[0]

  # Create extracted zone
  dist_extract_zone = I.newZone(name  = I.getName(dist_zone) + '_surf',
                                zsize = [[ex_vtx_distri[n_rank],ex_face_distri[n_rank],0]],
                                ztype = 'Unstructured')

  ex_fvtx_distri = par_utils.gather_and_shift(ex_face_vtx_idx[-1], comm, pdm_gnum_dtype)

  # > Grid coordinates
  cx, cy, cz = layouts.interlaced_to_tuple_coords(ex_vtx_coords)
  grid_coord = I.newGridCoordinates(parent=dist_extract_zone)
  I.newDataArray('CoordinateX', cx, parent=grid_coord)
  I.newDataArray('CoordinateY', cy, parent=grid_coord)
  I.newDataArray('CoordinateZ', cz, parent=grid_coord)

  eso = ex_fvtx_distri[i_rank] + ex_face_vtx_idx.astype(pdm_gnum_dtype)
  extract_ngon_n = I.newElements('NGonElements', 'NGON', erange = [1, ex_face_distri[n_rank]], parent=dist_extract_zone)

  I.newDataArray('ElementConnectivity', ex_face_vtx, parent=extract_ngon_n)
  I.newDataArray('ElementStartOffset' , eso        , parent=extract_ngon_n)

  np_distrib_vtx     = ex_vtx_distri [[i_rank, i_rank+1, n_rank]]
  np_distrib_face    = ex_face_distri[[i_rank, i_rank+1, n_rank]]
  np_distrib_facevtx = ex_fvtx_distri[[i_rank, i_rank+1, n_rank]]

  MT.newDistribution({'Cell' : np_distrib_face, 'Vertex' : np_distrib_vtx}, dist_extract_zone)
  MT.newDistribution({'Element' : np_distrib_face, 'ElementConnectivity' : np_distrib_facevtx}, extract_ngon_n)

  return dist_extract_zone

def extract_surf_zone_from_queries(dist_zone, queries, comm):
  """
  Create a zone containing a surfacic mesh, extracted from all the faces found under the
  nodes matched by one of the queries
  """

  all_point_list = collect_distributed_pl(dist_zone, queries, filter_loc='FaceCenter')
  _, dface_list  = np_utils.concatenate_point_list(all_point_list, pdm_gnum_dtype)

  return _extract_surf_zone(dist_zone, dface_list, comm)

def extract_surf_tree_from_queries(dist_tree, queries, comm):
  """
  For each zone in the input dist_tree, find the faces under the nodes matched by one 
  of the queries and extract the surfacic zone. Return a surfacic dist_tree including
  all of this zones
  """

  surf_tree = I.newCGNSTree()
  for base, zone in PT.iter_children_from_predicates(dist_tree, ['CGNSBase_t', 'Zone_t'], ancestors=True):
    surf_base = I.createUniqueChild(surf_tree, I.getName(base), 'CGNSBase_t', [2,2])

    if PT.Zone.Type(zone) == 'Unstructured':
      surf_zone = extract_surf_zone_from_queries(zone, queries, comm)
      I._addChild(surf_base, surf_zone)
    else:
      print(f"Warning : skip structured zone {I.getName(zone)} in extract_surf_tree")

  return surf_tree

def extract_surf_tree_from_bc(dist_tree, comm):
  """
  Shortcut for extract_surf_tree_from_queries specialized for BC_t nodes
  """
  queries = [[lambda n: I.getType(n) == 'ZoneBC_t', lambda n: I.getType(n) == 'BC_t']]
  return extract_surf_tree_from_queries(dist_tree, queries, comm)

