import time
import warnings
import numpy as np
from mpi4py import MPI

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils import logging as mlog

from .closest_elt import find_closest_boundary, find_closest_boundary_propagation

from maia.typing import *

BC_WALLS = ['BCWall', 'BCWallViscous', 'BCWallViscousHeatFlux', 'BCWallViscousIsothermal']


def detect_wall_families(tree: CGNSTree, bcwalls: List[str] = BC_WALLS) -> List[str]:
  """
  Return the list of Families having a FamilyBC_t node whose value is in bcwalls list
  """
  IS_WALL_FAM = PT.pred.NodePredicate(lambda n : PT.get_value(PT.find_child_from_label(n, 'FamilyBC_t')) in bcwalls)
  fam_query = PT.pred.label_is('Family_t') & PT.pred.has_child_of_label('FamilyBC_t') & IS_WALL_FAM
  return [PT.get_name(family) for family in PT.iter_children_from_predicates(tree, ['CGNSBase_t', fam_query])]


def bcwall_pred(part_tree: CGNSTree, walls:List[str]=BC_WALLS) -> PT.pred.NodePredicate:
  wall_bc_families = detect_wall_families(part_tree, walls)
  is_wall_bc = PT.pred.value_in(walls) | PT.pred.any([PT.pred.belongs_to_family(family) for family in wall_bc_families])
  return is_wall_bc

def compute_wall_distance(part_tree: CGNSPartTree,
                          comm: MPIComm,
                          point_cloud: str = 'CellCenter',
                          out_fs_name: str = 'WallDistance',
                          **options: Any) -> None:
  """Compute wall distances and add it in tree.

  For each volumic point, compute the distance to the nearest face belonging to a BC of kind wall.
  BC are considered to be of kind wall if their BCType (or the one of their related family) is one of 
  ``'BCWall'``, ``'BCWallViscous'``, ``'BCWallViscousHeatFlux'`` or ``'BCWallViscousIsothermal'``.

  Note: 
    Propagation method requires ParaDiGMa access and is only available for unstructured cell centered
    NGon connectivities grids. In addition, partitions must have been created from a single initial domain
    with this method.

  Tree is modified inplace: computed distance are added in a DiscreteData container whose
  name can be specified with out_fs_name parameter.

  The following optional parameters can be used to control the underlying method:

    - ``method`` ({'cloud', 'propagation'}): Choice of the geometric method. Defaults to ``'cloud'``.
    - ``perio`` (bool): Take into account periodic connectivities. Defaults to ``True``.
      Only available when method=cloud.

  Args:
    part_tree (CGNSPartTree)   : Input partitioned tree
    comm       (MPIComm)       : MPI communicator
    point_cloud (str, optional): Points to project on the surface. Can either be one of
      "CellCenter" or "Vertex" (coordinates are retrieved from the mesh) or the name of a FlowSolution
      node in which coordinates are stored. Defaults to CellCenter.
    out_fs_name (str, optional): Name of the output DiscreteData_t node storing wall distance data.
    **options: Additional options related to geometric method (see above)

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #compute_wall_distance@start
        :end-before: #compute_wall_distance@end
        :dedent: 2
  """
  MT.check_cgns_part_tree(part_tree)

  method = options.get('method', 'cloud')
  assert method in ["cloud", "propagation"], "Unknow method, expected 'cloud' or 'propagation'"

  start = time.time()
  
  # Retrieve Wall Families (warning -- if we have a Family_t appearing under two bases 
  # with the same name, it can be wrongly selected)
  bnd_predicate = bcwall_pred(part_tree)

  if method == "cloud":
    find_closest_boundary(part_tree,
                          part_tree,
                          point_cloud,
                          comm,
                          bnd_predicate,
                          perio=options.get('perio', True))

  else:
    if options.get('perio', True):
      warnings.warn("WallDistance do not manage periodicities except for 'cloud' method", RuntimeWarning, stacklevel=2)
    find_closest_boundary_propagation(part_tree,
                                      point_cloud,
                                      comm,
                                      bnd_predicate)


  end = time.time()


  is_inf = False
  for zone in PT.iter_all_Zone_t(part_tree): #Rename Distance -> TurbulentDistance
    container = PT.find_child_from_name(zone, 'ClosestElement')

    PT.rm_children_from_name(container, 'TurbulenceDistance') # Cleanup
    dist = PT.find_child_from_name(container, "Distance")
    if len(val := PT.get_np_value(dist)) > 0:
      is_inf = val.item(0) == np.inf
    PT.set_name(dist, 'TurbulentDistance')

    PT.rm_children_from_name(zone, out_fs_name)
    PT.set_name(container, out_fs_name)

  if comm.allreduce(is_inf, MPI.LOR):
    mlog.warning(f"Wall distance computing skipped because no wall-like BC_t have been found in tree." \
                  " Default values used for output arrays.")
  else:
    mlog.info(f"Wall distance computed ({end-start:.2f} s)")

