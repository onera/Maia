import maia

from maia.typing import *

def compute_wall_distance(part_tree: CGNSPartTree,
                          comm: MPIComm,
                          point_cloud: str = 'CellCenter',
                          out_fs_name: str = 'WallDistance',
                          **options: Any) -> None:
  """ Deprecated -- Use maia.algo.compute_wall_distance """
  maia.algo.compute_wall_distance(part_tree, comm, point_cloud, out_fs_name, **options)