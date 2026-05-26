import maia
import warnings

from maia.typing import *

def compute_wall_distance(part_tree: CGNSPartTree,
                          comm: MPIComm,
                          point_cloud: str = 'CellCenter',
                          out_fs_name: str = 'WallDistance',
                          **options: Any) -> None:
  """ Deprecated -- Use maia.algo.compute_wall_distance """
  # To remove when 1.10 is released
  warnings.warn("This API is deprecated. Use directly maia.algo.compute_wall_distance", DeprecationWarning, stacklevel=2)
  maia.algo.compute_wall_distance(part_tree, comm, point_cloud, out_fs_name, **options)