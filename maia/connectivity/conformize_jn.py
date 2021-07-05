import numpy as np

import Converter.Internal as I

import maia.connectivity.vertex_list  as CVL
from maia import npy_pdm_gnum_dtype as pdm_dtype
from maia.tree_exchange.dist_to_part import data_exchange as MBTP
from maia.tree_exchange.part_to_dist import data_exchange as MPTB
from maia.sids import Internal_ext as IE

def get_vtx_coordinates(grid_coords_n, distri_vtx, requested_vtx_list, comm):
  """
  Get the coordinates of requested vertices ids (wraps BlockToPart) and
  return it as 3 numpy array
  """
  dist_data = dict()
  for data in I.getNodesFromType1(grid_coords_n, 'DataArray_t'):
    dist_data[I.getName(data)] = data[1]
    
  ln_to_gn_list = [np.asarray(requested_vtx, dtype=pdm_dtype) for requested_vtx in requested_vtx_list]

  return MBTP.dist_to_part(distri_vtx.astype(pdm_dtype), dist_data, ln_to_gn_list, comm)
  


def update_vtx_coordinates(grid_coords_n, part_data, distri_vtx, requested_vtx_list, comm):
  """
  Set the coordinates of requested vertices ids (wraps PartToBlock)
  """
  part_data["NodeId"]=[]
  for requested_vtx in requested_vtx_list:
    part_data["NodeId"].append(requested_vtx)
  

  ln_to_gn_list = [np.asarray(requested_vtx, dtype=pdm_dtype) for requested_vtx in requested_vtx_list]

  dist_data = MPTB.part_to_dist(distri_vtx.astype(pdm_dtype), part_data, ln_to_gn_list, comm)
  
  local_indices = dist_data['NodeId']-1-distri_vtx[0]
  
  for data in I.getNodesFromType1(grid_coords_n, 'DataArray_t'):
    data[1][local_indices] = dist_data[I.getName(data)]



def conformize_jn(dist_tree,JN_for_duplication_paths,comm):
  """
  conformization of join
  
  Etapes à réaliser :
  1. récupérér les path des deux 'GridConnectivity' qui composent le raccord
  2. pour chaque 'GridConnectivity', générer le PoinList et PointListDonor aux noeuds (Vertex)
  3. récupérer les coordonnées de correspondantes et en faire la moyenne
  4. mettre à jour les anciennes coordonnées avec celles moyennées
  """
  
  joinPath1, joinPath2 = JN_for_duplication_paths
  pl_vtx1, pl_vtx_opp1, distri_jn_vtx1 = CVL.generate_jn_vertex_list(dist_tree, joinPath1, comm)
  
  zone1Path = "/".join(joinPath1.split("/")[0:2])
  zone1 = I.getNodeFromPath(dist_tree,zone1Path)
  gridCoords1Node = I.getNodeFromType1(zone1,"GridCoordinates_t")
  distriVtxZone1  = I.getVal(IE.getDistribution(zone1, 'Vertex'))
  
  zone2Path = "/".join(joinPath2.split("/")[0:2])
  zone2 = I.getNodeFromPath(dist_tree,zone2Path)
  gridCoords2Node = I.getNodeFromType1(zone2,"GridCoordinates_t")
  distriVtxZone2  = I.getVal(IE.getDistribution(zone2, 'Vertex'))
  
  coord1 = get_vtx_coordinates(gridCoords1Node, distriVtxZone1, [pl_vtx1], comm)
  
  coord2 = get_vtx_coordinates(gridCoords2Node, distriVtxZone2, [pl_vtx_opp1], comm)
  
  part_data = dict()
  for data in I.getNodesFromType1(gridCoords1Node, 'DataArray_t'):
    dataName = I.getName(data)
    part_data[dataName] = [(coord1[dataName][0]+coord2[dataName][0])/2.]
    
  update_vtx_coordinates(gridCoords1Node, part_data, distriVtxZone1, [pl_vtx1], comm)
  
  update_vtx_coordinates(gridCoords2Node, part_data, distriVtxZone2, [pl_vtx_opp1], comm)
