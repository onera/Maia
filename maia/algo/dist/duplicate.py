import numpy as np
import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia.algo.transform as TRF
import maia.algo.dist.conformize_jn as CCJ
import maia.algo.dist.matching_jns_tools as MJT
from   maia.typing import *
from   maia.pytree.typing import Predicates

from maia.utils import logging as mlog

def duplicate_from_periodic_jns(dist_tree: CGNSDistTree,
                                zone_paths: List[CGNSPath],
                                jn_paths_for_dupl: Tuple[List[CGNSPath], List[CGNSPath]],
                                dupl_nb: int,
                                comm: MPIComm,
                                conformize: bool = False,
                                apply_to_fields: bool = True) -> None:
  """Duplicate a mesh from a transformation defined in its periodic connectivities.

  Input tree is modified inplace.

  Args:
    dist_tree (CGNSDistTree): Input distributed tree
    zone_paths (list of str): List of pathes (BaseName/ZoneName) of the connected zones to duplicate
    jn_paths_for_dupl (pair of list of str): (listA, listB) where listA (resp. list B) stores all the
        pathes of the GridConnectivity nodes defining the first (resp. second) side of a periodic match.
    dupl_nb (int) : Number of duplications to perform
    comm       (MPIComm) : MPI communicator
    conformize (bool, optional): If true, ensure that the generated interface vertices have exactly same
        coordinates (see :func:`conformize_jn_pair`). Defaults to False.
    apply_to_fields (bool, optional): See :func:`maia.algo.transform_affine`. Defaults to ``True``.

  See also:
    For rotating periodicities, it is also possible to automatically recover a circular (360°) mesh
    with the function :func:`duplicate_from_rotation_jns_to_360`, which takes the same arguments,
    excepted ``dupl_nb``.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #duplicate_from_rotation_to_360@start
        :end-before: #duplicate_from_rotation_to_360@end
        :dedent: 2
  """
  
  #############
  # Example for a monozone duplication
  #
  # If jn = [[Path/To/MatchA],[Path/To/MatchB]]
  # Then first = MatchA and second = MatchB
  #
  #         ________                           ________________
  #         |      |                           |      ||      |       
  #         |      |                           |      ||      |       
  #         |      |                           |      ||      |       
  #         | Zone |           ===>>>          | Zone || Zone |       
  #         |      |                           |      || dup  |       
  #        /|      |\                         /|      ||      |\      
  #       / |______| \                       / |______||______| \     
  #      /            \                     /        /  \        \
  #   MatchA         MatchB              MatchA   MatchB \       MatchBDup
  #                                                     MatchADup 
  #
  # ------------------------- 
  # Example for a multizones duplication
  #
  # If jn = [[Path/To/MatchA1,Path/To/MatchA2],[Path/To/MatchB1,Path/To/MatchB2]]
  # Then first = MatchA1 and second = MatchB1
  #
  #                                                     MatchA1Dup 
  #   MatchA1         MatchB1             MatchA1  MatchB1 /       MatchB1Dup  
  #      \  _________  /                     \  ________\_/_______  /
  #       \ |       | /                       \ |       ||       | /        
  #        \| Zone1 |/                         \| Zone1 || Zone- |/         
  #         |       |                           |       || dup1  |       
  #         |-------|       ===>>>              |-------||-------|       
  #         |       |                           |       || Zone- |       
  #        /| Zone2 |\                         /| Zone2 || dup2  |\      
  #       / |_______| \                       / |_______||_______| \     
  #      /             \                     /         /  \         \
  #   MatchA2         MatchB2             MatchA2  MatchB2 \       MatchB2Dup
  #                                                     MatchA2Dup 
  #
  #############

  MT.check_cgns_dist_tree(dist_tree)
  if dupl_nb < 0:
    return

  jn_paths_a, jn_paths_b = jn_paths_for_dupl
  zones = [PT.find_node_from_path(dist_tree, path) for path in zone_paths]

  #Store initial values of joins
  jn_values_a = [PT.get_value(PT.find_node_from_path(dist_tree,jn_path_a)) for jn_path_a in jn_paths_a]
  jn_values_b = [PT.get_value(PT.find_node_from_path(dist_tree,jn_path_b)) for jn_path_b in jn_paths_b]

  # Prepare matching jns
  if conformize:
    jn_to_opp = {}
    for i, jn_path_a in enumerate(jn_paths_a):
      jn_path_b = MJT.get_jn_donor_path(dist_tree, jn_path_a)
      assert jn_path_b in jn_paths_b
      jn_to_opp[jn_path_a] = jn_path_b

  # Get first join in the first list of joins (A)
  first_join_in_matchs_a = PT.find_node_from_path(dist_tree, jn_paths_a[0])
  
  # Get transformation information
  rotation_center_a, rotation_angle_a, translation_a = PT.GridConnectivity.periodic_values(first_join_in_matchs_a)
  if rotation_angle_a.size == 2:
    rotation_angle_a = rotation_angle_a[0] if rotation_angle_a[0] != 0 else rotation_angle_a[1] # We dont know if angle is stored in array[0] or array[1]
  
  # Store initial periodicity information of joins of the second joins list (B)
  jn_b_properties = []
  for jn_path_b in jn_paths_b:
    jn_b_property = PT.find_node_from_path(dist_tree, f"{jn_path_b}/GridConnectivityProperty")
    jn_b_properties.append(PT.deep_copy(jn_b_property))

  # Get the name of all zones to duplicate in order to update the value of GridConnectivity
  # nodes not involved in the duplication (not in jn_paths_for_dupl)
  gc_values_to_update = zone_paths + [PT.get_name(zone) for zone in zones] #Manage both ways BaseName/ZoneName + ZoneName

  gc_predicate:Predicates = ["ZoneGridConnectivity_t",
                             lambda n : PT.get_label(n) in ["GridConnectivity_t", "GridConnectivity1to1_t"]]

  # Update the value of all GridConnectivity nodes not involved in the duplication from initial zones
  for zone_path, zone in zip(zone_paths, zones):
    for zgc, gc in PT.iter_children_from_predicates(zone, gc_predicate, ancestors=True):
      init_gc_path = f"{zone_path}/{PT.get_name(zgc)}/{PT.get_name(gc)}"
      if (init_gc_path not in jn_paths_a) and (init_gc_path not in jn_paths_b):
        gc_value = PT.get_value(gc)
        if gc_value in gc_values_to_update:
          PT.set_value(gc, f"{gc_value}.D0")
    PT.set_name(zone, f"{PT.get_name(zone)}.D0") #Update zone name
  
  # Duplicate 'dupl_nb' times the list of zones 'zones'
  for n in range(dupl_nb):
    for zone_path, zone in zip(zone_paths, zones):
      base_name, root_zone_name = zone_path.split('/')
      base = PT.find_child_from_name(dist_tree, base_name)
      duplicated_zone = PT.deep_copy(zone)
      PT.set_name(duplicated_zone, f"{root_zone_name}.D{n+1}")
      TRF.transform_affine(duplicated_zone,
                           rotation_center = rotation_center_a,
                           rotation_angle  = (n+1)*rotation_angle_a,
                           translation     = (n+1)*translation_a,
                           apply_to_fields = apply_to_fields)
  
      # Update the value of all GridConnectivity nodes not involved in the duplication from initial zones
      for zgc, gc in PT.iter_children_from_predicates(duplicated_zone, gc_predicate, ancestors=True):
        gc_path = f"{zone_path}/{PT.get_name(zgc)}/{PT.get_name(gc)}"
        if (gc_path not in jn_paths_a) and (gc_path not in jn_paths_b):
          gc_value = ".D0".join(PT.get_str_value(gc).split(".D0")[0:-1])
          if gc_value in gc_values_to_update:
            PT.set_value(gc, f"{gc_value}.D{n+1}")

      # Add duplicated zone to the suitable base
      PT.add_child(base, duplicated_zone)

    # Transform periodic joins of the second joins list (B) from previous set of zones
    # to non periodic joins
    for jb, jn_path_b in enumerate(jn_paths_b):
      jn_path_b_prev = PT.utils.update_path_elt(jn_path_b, 1, lambda zn : zn + f".D{n}")
      jn_b_prev_node = PT.find_node_from_path(dist_tree, jn_path_b_prev)
      PT.rm_children_from_label(jn_b_prev_node, "GridConnectivityProperty_t")
      PT.set_value(jn_b_prev_node, f"{jn_values_b[jb]}.D{n+1}")

    # Transform periodic joins of the fisrt joins list (A) from current set of zones
    # to non periodic joins
    for ja, jn_path_a in enumerate(jn_paths_a):
      jn_path_a_curr = PT.utils.update_path_elt(jn_path_a, 1, lambda zn : zn + f".D{n+1}")
      jn_a_curr_node = PT.find_node_from_path(dist_tree, jn_path_a_curr)
      PT.rm_children_from_label(jn_a_curr_node, "GridConnectivityProperty_t")
      PT.set_value(jn_a_curr_node, f"{jn_values_a[ja]}.D{n}")

    if conformize:
      for jn_path_a, jn_path_b in jn_to_opp.items():
        jn_path_a_curr = PT.utils.update_path_elt(jn_path_a, 1, lambda zn : zn + f".D{n+1}")
        jn_path_b_prev = PT.utils.update_path_elt(jn_path_b, 1, lambda zn : zn + f".D{n}")
        CCJ.conformize_jn_pair(dist_tree, (jn_path_a_curr, jn_path_b_prev), comm)

  # Update information for joins of the fisrt joins list (A) from initial set of zones
  for ja, jn_path_a in enumerate(jn_paths_a):
    jn_path_a_init = PT.utils.update_path_elt(jn_path_a, 1, lambda zn : zn + ".D0")
    jn_a_init_node = PT.find_node_from_path(dist_tree, jn_path_a_init)
    gcp_a_init = PT.find_child_from_label(jn_a_init_node, "GridConnectivityProperty_t")
    rotation_angle_a_node = PT.find_node_from_name(gcp_a_init, "RotationAngle", depth=2)
    translation_a_node    = PT.find_node_from_name(gcp_a_init, "Translation", depth=2)
    PT.set_value(rotation_angle_a_node, PT.get_np_value(rotation_angle_a_node) * (dupl_nb+1))
    PT.set_value(translation_a_node,    PT.get_np_value(translation_a_node)    * (dupl_nb+1))
    PT.set_value(jn_a_init_node, f"{jn_values_a[ja]}.D{dupl_nb}")

  # Update information for joins of the second joins list (B) from last set of duplicated zones
  for jb, jn_path_b in enumerate(jn_paths_b):
    jn_path_b_last = PT.utils.update_path_elt(jn_path_b, 1, lambda zn : zn + f".D{dupl_nb}")
    jn_b_last_node = PT.find_node_from_path(dist_tree, jn_path_b_last)
    PT.rm_children_from_label(jn_b_last_node, 'GridConnectivityProperty_t')
    PT.add_child(jn_b_last_node, jn_b_properties[jb])
    gcp_b_last = PT.find_child_from_label(jn_b_last_node, "GridConnectivityProperty_t")
    rotation_angle_b_node = PT.find_node_from_name(gcp_b_last, "RotationAngle", depth=2)
    translation_b_node    = PT.find_node_from_name(gcp_b_last, "Translation", depth=2)
    PT.set_value(rotation_angle_b_node, PT.get_np_value(rotation_angle_b_node) * (dupl_nb+1))
    PT.set_value(translation_b_node,    PT.get_np_value(translation_b_node)    * (dupl_nb+1))
    PT.set_value(jn_b_last_node, f"{jn_values_b[jb]}.D0")
  

def duplicate_from_rotation_jns_to_360(dist_tree: CGNSDistTree,
                                       zone_paths: List[CGNSPath],
                                       jn_paths_for_dupl: Tuple[List[CGNSPath], List[CGNSPath]],
                                       comm: MPIComm,
                                       conformize: bool = False,
                                       apply_to_fields: bool = True) -> None:
  """Reconstitute a circular mesh from an angular section of the geometry.

  Input tree is modified inplace.

  Args:
    dist_tree (CGNSDistTree): Input distributed tree
    zone_paths (list of str): List of pathes (BaseName/ZoneName) of the connected zones to duplicate
    jn_paths_for_dupl (pair of list of str): (listA, listB) where listA (resp. list B) stores all the
        pathes of the GridConnectivity nodes defining the first (resp. second) side of a periodic match.
    comm       (MPIComm) : MPI communicator
    conformize (bool, optional): If true, ensure that the generated interface vertices have exactly same
        coordinates (see :func:`conformize_jn_pair`). Defaults to False.
    apply_to_fields (bool, optional): See :func:`maia.algo.transform_affine`. Defaults to ``True``.

  See also:
    Instead of recovering the circular mesh, it is also possible to perfom a custom number 
    of duplications with the function :func:`duplicate_from_periodic_jns`. This function takes
    the additional (integer) argument ``dupl_nb`` before ``comm``, and also work with translation
    periodicities.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #duplicate_from_rotation_to_360@start
        :end-before: #duplicate_from_rotation_to_360@end
        :dedent: 2
  """
  MT.check_cgns_dist_tree(dist_tree)
  if conformize:
    jn_to_opp:Dict[CGNSPath, CGNSPath] = {}
    for i, jn_path_a in enumerate(jn_paths_for_dupl[0]):
      jn_path_b = MJT.get_jn_donor_path(dist_tree, jn_path_a)
      assert jn_path_b in jn_paths_for_dupl[1]
      jn_to_opp[jn_path_a] = jn_path_b
    _jn_paths_for_dupl:Tuple[List[CGNSPath], List[CGNSPath]] = ( [], [] )
    for path, path_opp in jn_to_opp.items():
      _jn_paths_for_dupl[0].append(path)
      _jn_paths_for_dupl[1].append(path_opp)
  else:
    _jn_paths_for_dupl = jn_paths_for_dupl

  # Get first join in the first list of joins (A)
  first_join_in_matchs_a = PT.find_node_from_path(dist_tree, _jn_paths_for_dupl[0][0])
  
  # Get transformation information
  rotation_center_a, rotation_angle_a, translation_a = PT.GridConnectivity.periodic_values(first_join_in_matchs_a)
  
  if (translation_a != 0).any():
    raise ValueError("The join is not periodic only by rotation !")

  # Find the number of duplication needed
  for i in range(len(rotation_angle_a)):
    if abs(rotation_angle_a[i]) < 5*np.finfo(np.float64).eps:
      rotation_angle_a[i] = 0.
  index = np.where(rotation_angle_a != 0)[0]
  if index.size == 1:
    sectors_number = abs(int(np.round(2*np.pi/rotation_angle_a[index])))
    rotation_angle_a[index] = np.sign(rotation_angle_a[index]) * 2*np.pi/sectors_number
  else:
    raise ValueError("Zone/Join not define a section of a row")

  if rotation_angle_a.size == 2:
    rotation_angle_a = rotation_angle_a[0] if rotation_angle_a[0] != 0. else rotation_angle_a[1]

  # Duplicate 'sectors_number - 1' times the list of zones 'zones'
  duplicate_from_periodic_jns(dist_tree, zone_paths, _jn_paths_for_dupl, sectors_number-1, 
      comm, conformize, apply_to_fields)

  # Transform periodic joins of the fisrt joins list (A) from initial set of zones
  # to non periodic joins
  for jn_path_a in _jn_paths_for_dupl[0]:
    jn_path_a_init = PT.utils.update_path_elt(jn_path_a, 1, lambda zn : zn + f".D{0}")
    jn_a_init_node = PT.find_node_from_path(dist_tree, jn_path_a_init)
    PT.rm_children_from_label(jn_a_init_node, "GridConnectivityProperty_t")

  # Transform periodic joins of the second joins list (B) from last set of duplicated zones
  # to non periodic joins
  for jn_path_b in _jn_paths_for_dupl[1]:
    jn_path_b_last = PT.utils.update_path_elt(jn_path_b, 1, lambda zn : zn + f".D{sectors_number-1}")
    jn_b_last_node = PT.find_node_from_path(dist_tree, jn_path_b_last)
    PT.rm_children_from_label(jn_b_last_node, "GridConnectivityProperty_t")

  if conformize:
    # Conformize last, other have been conformized in duplicate_from_periodic_jns
    for jn_path_a, jn_path_b in jn_to_opp.items():
      jn_path_a_init = PT.utils.update_path_elt(jn_path_a, 1, lambda zn : zn + f".D{0}")
      jn_path_b_last = PT.utils.update_path_elt(jn_path_b, 1, lambda zn : zn + f".D{sectors_number-1}")
      CCJ.conformize_jn_pair(dist_tree, (jn_path_a_init, jn_path_b_last), comm)


def _family_name_to_zones_and_jns_paths(dist_tree: CGNSDistTree,
                                        family_name: str) -> Tuple[List[CGNSPath], Tuple[List[CGNSPath], List[CGNSPath]]]:
  is_z_in_fam = lambda n : PT.get_label(n) == 'Zone_t' and PT.predicate.belongs_to_family(n, family_name)
  zone_paths = PT.predicates_to_paths(dist_tree, ['CGNSBase_t', is_z_in_fam])

  mask_tree = PT.shallow_copy(dist_tree)
  for mask_base in PT.get_all_CGNSBase_t(mask_tree):
      PT.keep_children_from_predicate(mask_base, is_z_in_fam)

  _, perio_jns = PT.Tree.find_periodic_jns(mask_tree)

  mlog.debug(f"The following zones have been detected for duplication:\n  {zone_paths}")
  mlog.debug(f"The following joins have been detected for duplication:\n  {perio_jns}")

  if len(perio_jns) < 2:
    raise RuntimeError("Not enought periodic transformation found in input tree")
  elif len(perio_jns) > 2:
    raise RuntimeError("Too many periodic transformation found in input tree")

  perio_jns_as_tuple = (perio_jns[0], perio_jns[1])

  return zone_paths, perio_jns_as_tuple

def duplicate_family_from_periodic_jns(dist_tree: CGNSDistTree,
                                       family_name: str,
                                       dupl_nb: int,
                                       comm: MPIComm,
                                       **kwargs) -> None:
  """Duplicate zones belonging to the specified family.

  This is a shortcut for :func:`duplicate_from_periodic_jns` with autodetection of:

  - Zones to duplicate (every zone belonging to the provided family)
  - Periodic connectivities to use for duplication. Note that **this function will fail** 
    if the number of periodic transformation found in the group of zones is not exactly one.

  Args:
    dist_tree (CGNSDistTree): Input distributed tree
    family_name (str): Name of family gathering the zones to duplicate
    dupl_nb (int) : Number of duplications to perform
    comm       (MPIComm) : MPI communicator
    kwargs: See :func:`duplicate_from_periodic_jns`

  See also:
    For rotating periodicities, it is also possible to automatically recover a circular (360°) mesh
    with the function :func:`duplicate_family_from_rotation_jns_to_360`, which takes the same arguments,
    excepted ``dupl_nb``.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #duplicate_family_from_periodic_jns@start
        :end-before: #duplicate_family_from_periodic_jns@end
        :dedent: 2
  """
  MT.check_cgns_dist_tree(dist_tree)
  zone_paths, perio_jns = _family_name_to_zones_and_jns_paths(dist_tree, family_name)
  duplicate_from_periodic_jns(dist_tree, zone_paths, perio_jns, dupl_nb, comm, **kwargs)

def duplicate_family_from_rotation_jns_to_360(dist_tree: CGNSDistTree,
                                              family_name: str,
                                              comm: MPIComm,
                                              **kwargs) -> None:
  """Reconstitute a circular mesh from an angular section of the geometry for zones
  belonging to the provided family"""
  MT.check_cgns_dist_tree(dist_tree)
  zone_paths, perio_jns = _family_name_to_zones_and_jns_paths(dist_tree, family_name)
  duplicate_from_rotation_jns_to_360(dist_tree, zone_paths, perio_jns, comm, **kwargs)