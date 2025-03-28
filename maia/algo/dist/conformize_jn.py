import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.algo.dist import vertex_list as VL
from maia.transfer  import protocols   as EP
from maia.utils     import par_utils
from maia.typing    import CGNSDistTree, MPIComm, List

def conformize_jn_pair(dist_tree: CGNSDistTree,
                       jn_paths: List[str],
                       comm: MPIComm) -> None:
  """
  Ensure that the vertices belonging to the two sides of a 1to1 GridConnectivity
  have the same coordinates.

  Matching join with Vertex or FaceCenter location are admitted. Coordinates
  of vertices are made equal by computing the arithmetic mean of the two
  values.

  Input tree is modified inplace.

  Args:
    dist_tree  (CGNSDistTree): Input tree
    jn_pathes  (list of str) : Pathes of the two matching ``GridConnectivity_t``
       nodes. Pathes must start from the root of the tree.
    comm       (`MPIComm`)   : MPI communicator

  """
  
  # Get vtx ids and opposite vtx ids for this join
  location = PT.Subset.GridLocation(PT.get_node_from_path(dist_tree, jn_paths[0]))
  if location == 'Vertex':
    pl_vtx_list = [PT.get_node_from_path(dist_tree, jn_paths[0]+f'/PointList{d}')[1][0] for d in ['', 'Donor']]
  elif location == 'FaceCenter':
    pl_vtx_list = VL.generate_jn_vertex_list(dist_tree, jn_paths[0], comm)[:2]
  else:
    raise RuntimeError(f"Unsupported grid location for jn {jn_paths[0]}")

  zones = [PT.get_node_from_path(dist_tree, PT.utils.path_head(path, 2)) for path in jn_paths]
  dist_coords = [PT.Zone.coordinates(zone)             for zone in zones]
  vtx_distris = [MT.getDistribution(zone, 'Vertex')[1] for zone in zones]
  vtx_distris = [par_utils.partial_to_full_distribution(di, comm) for di in vtx_distris]

  indexer0 = EP.GlobalIndexer(vtx_distris[0], pl_vtx_list[0]-1, comm)
  indexer1 = EP.GlobalIndexer(vtx_distris[1], pl_vtx_list[1]-1, comm)

  for coord0, coord1 in zip(*dist_coords):
    # For each component X,Y,Z : extract vtx values on the two zones (Take),
    # compute the average, and then put back the average in the two zones (Put)
    mean_coords = 0.5*(indexer0.Take(coord0) + indexer1.Take(coord1))
    indexer0.Put(mean_coords, coord0)
    indexer1.Put(mean_coords, coord1)
  