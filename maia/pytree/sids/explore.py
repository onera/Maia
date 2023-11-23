import numpy as np
import maia.pytree as PT

from maia.pytree.typing import *

from maia.pytree.compare import check_is_label

import warnings

def getZoneDonorPath(current_base:str, gc:CGNSTree) -> str:
  """ DEPRECATED : see ``PT.GridConnectivity.ZoneDonorPath`` """
  warnings.warn("This function is deprecated in favor of PT.GridConnectivity.ZoneDonorPath",
          DeprecationWarning, stacklevel=2)
  return PT.GridConnectivity.ZoneDonorPath(gc, current_base)


@check_is_label('ZoneSubRegion_t', 0)
@check_is_label('Zone_t', 1)
def getSubregionExtent(sub_region_node:CGNSTree, zone:CGNSTree) -> str:
  """ DEPRECATED : see ``PT.Subset.ZSRExtent`` """
  warnings.warn("This function is deprecated in favor of PT.Subset.ZSRExtent",
          DeprecationWarning, stacklevel=2)
  return PT.Subset.ZSRExtent(sub_region_node, zone)


def find_connected_zones(tree:CGNSTree) -> List[CGNSTree]:
  """ DEPRECATED : see ``PT.Tree.find_connected_zones`` """
  warnings.warn("This function is deprecated in favor of PT.Tree.find_connected_zones",
          DeprecationWarning, stacklevel=2)
  return PT.Tree.find_connected_zones(tree)

def find_periodic_jns(tree: CGNSTree, rtol=1e-5, atol=0.) -> Tuple[List[List[np.ndarray]], List[List[str]]]:
  """ DEPRECATED : see ``PT.Tree.find_periodic_jns`` """
  warnings.warn("This function is deprecated in favor of PT.Tree.find_periodic_jns",
          DeprecationWarning, stacklevel=2)
  return PT.Tree.find_periodic_jns(tree, rtol, atol)