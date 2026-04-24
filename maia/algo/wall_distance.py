import time
import warnings
import numpy as np
from mpi4py import MPI

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils import logging as mlog

from .part import closest_elt as pclosest_elt
from .dist import closest_elt as dclosest_elt

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

def compute_wall_distance(tree: CGNSTree,
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
    part_tree (CGNSPartTree)   : Input tree, distributed or partitioned
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

  method = options.get('method', 'cloud')
  assert method in ["cloud", "propagation"], "Unknow method, expected 'cloud' or 'propagation'"

  if MT.is_cgns_dist_tree(tree):
    impl = dclosest_elt
  elif MT.is_cgns_part_tree(tree):
    impl = pclosest_elt #type: ignore[misc] # Dispatch confuse mypy
  else:
    raise ValueError("Tree must be either distributed or partitioned")

  start = time.time()
  
  # Retrieve Wall Families (warning -- if we have a Family_t appearing under two bases 
  # with the same name, it can be wrongly selected)
  bnd_predicate = bcwall_pred(tree)


  if method == "cloud":
    impl.find_closest_boundary(tree, # type:ignore[arg-type] # Dispatch confuse mypy
                               tree, # type:ignore[arg-type] # Dispatch confuse mypy
                               point_cloud,
                               comm,
                               bnd_predicate,
                               perio=options.get('perio', True))

  else:
    if options.get('perio', True):
      warnings.warn("WallDistance do not manage periodicities except for 'cloud' method", RuntimeWarning, stacklevel=2)
    impl.find_closest_boundary_propagation(tree, # type:ignore[arg-type] # Dispatch confuse mypy
                                           comm,
                                           bnd_predicate)


  end = time.time()


  is_inf = False
  for zone in PT.iter_all_Zone_t(tree): #Rename Distance -> TurbulentDistance
    container = PT.find_child_from_name(zone, 'ClosestElement')

    PT.rm_children_from_name(container, 'TurbulenceDistance') # Cleanup
    dist = PT.find_child_from_name(container, "Distance")
    if len(val := PT.get_np_value(dist)) > 0:
      is_inf = val.item(0) == np.inf
    PT.set_name(dist, 'TurbulentDistance')

    if (dest := PT.get_child_from_name(zone, out_fs_name)) is not None:
      assert PT.Container.GridLocation(dest) == PT.Container.GridLocation(container)
      for array in PT.get_children_from_label(container, 'DataArray_t'):
        PT.update_child(dest, PT.get_name(array), 'DataArray_t', PT.get_value(array))
      PT.rm_child(zone, container)
    else:
      PT.set_name(container, out_fs_name)

  if comm.allreduce(is_inf, MPI.LOR):
    mlog.warning(f"Wall distance computing skipped because no wall-like BC_t have been found in tree." \
                  " Default values used for output arrays.")
  else:
    mlog.info(f"Wall distance computed ({end-start:.2f} s)")

