import Converter.Internal as I

from maia import npy_pdm_gnum_dtype   as pdm_dtype
from maia.sids import sids
from maia.sids import Internal_ext    as IE
from maia.sids import pytree          as PT
import maia.connectivity.vertex_list  as VL
from maia.tree_exchange.dist_to_part import data_exchange as MBTP
from maia.tree_exchange.part_to_dist import data_exchange as MPTB

def conformize_jn(dist_tree, jn_paths, comm):
  """
  Ensure that the vertices belonging to the two sides of a 1to1 GridConnectivity
  have the same coordinates
  """
  coord_query = ['GridCoordinates_t', 'DataArray_t']
  
  # Get vtx ids and opposite vtx ids for this join
  location = sids.GridLocation(I.getNodeFromPath(dist_tree, jn_paths[0]))
  if location == 'Vertex':
    pl_vtx_list = [I.getNodeFromPath(dist_tree, jn_paths[0]+f'/PointList{d}')[1][0] for d in ['', 'Donor']]
  elif location == 'FaceCenter':
    pl_vtx_list = VL.generate_jn_vertex_list(dist_tree, jn_paths[0], comm)[:2]
  else:
    raise RuntimeError(f"Unsupported grid location for jn {jn_paths[0]}")

  # Collect data
  mean_coords = {}
  vtx_distris = []
  for i, path in enumerate(jn_paths):
    zone = I.getNodeFromPath(dist_tree, PT.path_head(path, 2))
    vtx_distri = I.getVal(IE.getDistribution(zone, 'Vertex')).astype(pdm_dtype, copy=False)
    dist_coords = {}
    for grid_co_n, coord_n in PT.iter_nodes_from_predicates(zone, coord_query, ancestors=True):
      dist_coords[f"{I.getName(grid_co_n)}/{I.getName(coord_n)}"] = coord_n[1]
  
    part_coords = MBTP.dist_to_part(vtx_distri, dist_coords, [pl_vtx_list[i]], comm)
  
    for path, value in part_coords.items():
      try:
        mean_coords[path][0] += 0.5*value[0]
      except KeyError:
        mean_coords[path] = [0.5*value[0]]
    vtx_distris.append(vtx_distri)
  
  # Send back the mean value to the two zones, and update tree
  for i, path in enumerate(jn_paths):
    zone = I.getNodeFromPath(dist_tree, PT.path_head(path, 2))
    mean_coords['NodeId'] = [pl_vtx_list[i]]
    dist_data = MPTB.part_to_dist(vtx_distris[i], mean_coords, [pl_vtx_list[i]], comm)

    loc_indices = dist_data.pop('NodeId') - vtx_distri[0] - 1 

    # Update data
    for coord_path, value in dist_data.items():
      node = I.getNodeFromPath(zone, coord_path)
      node[1][loc_indices] = value
