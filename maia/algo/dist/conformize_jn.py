import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia.algo.dist.vertex_list    as VL
from maia.transfer import protocols as EP

def conformize_jn_pair(dist_tree, jn_paths, comm):
  """
  Ensure that the vertices belonging to the two sides of a 1to1 GridConnectivity
  have the same coordinates.

  Matching join with Vertex or FaceCenter location are admitted. Coordinates
  of vertices are made equal by computing the arithmetic mean of the two
  values.

  Input tree is modified inplace.

  Args:
    dist_tree  (CGNSTree): Input tree
    jn_pathes  (list of str): Pathes of the two matching ``GridConnectivity_t``
       nodes. Pathes must start from the root of the tree.
    comm       (`MPIComm`) : MPI communicator

  """
  coord_query = ['GridCoordinates_t', 'DataArray_t']
  
  # Get vtx ids and opposite vtx ids for this join
  location = PT.Subset.GridLocation(PT.get_node_from_path(dist_tree, jn_paths[0]))
  if location == 'Vertex':
    pl_vtx_list = [PT.get_node_from_path(dist_tree, jn_paths[0]+f'/PointList{d}')[1][0] for d in ['', 'Donor']]
  elif location == 'FaceCenter':
    pl_vtx_list = VL.generate_jn_vertex_list(dist_tree, jn_paths[0], comm)[:2]
  else:
    raise RuntimeError(f"Unsupported grid location for jn {jn_paths[0]}")

  # Collect data
  mean_coords = {}
  vtx_distris = []
  for i, path in enumerate(jn_paths):
    zone = PT.get_node_from_path(dist_tree, PT.path_head(path, 2))
    vtx_distri = PT.get_value(MT.getDistribution(zone, 'Vertex'))
    dist_coords = {}
    for grid_co_n, coord_n in PT.iter_nodes_from_predicates(zone, coord_query, ancestors=True):
      dist_coords[f"{PT.get_name(grid_co_n)}/{PT.get_name(coord_n)}"] = coord_n[1]
  
    part_coords = EP.block_to_part(dist_coords, vtx_distri, [pl_vtx_list[i]], comm)
  
    for path, value in part_coords.items():
      try:
        mean_coords[path][0] += 0.5*value[0]
      except KeyError:
        mean_coords[path] = [0.5*value[0]]
    vtx_distris.append(vtx_distri)
  
  # Send back the mean value to the two zones, and update tree
  for i, path in enumerate(jn_paths):
    zone = PT.get_node_from_path(dist_tree, PT.path_head(path, 2))
    mean_coords['NodeId'] = [pl_vtx_list[i]]
    dist_data = EP.part_to_block(mean_coords, vtx_distris[i], [pl_vtx_list[i]], comm)

    loc_indices = dist_data.pop('NodeId') - vtx_distri[0] - 1 

    # Update data
    for coord_path, value in dist_data.items():
      node = PT.get_node_from_path(zone, coord_path)
      node[1][loc_indices] = value
