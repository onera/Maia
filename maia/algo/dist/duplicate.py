import numpy as np

import maia.pytree        as PT

from   maia.utils import py_utils
import maia.algo.transform as TRF
import maia.algo.dist.conformize_jn as CCJ
import maia.algo.dist.matching_jns_tools as MJT

def duplicate_from_periodic_jns(dist_tree, zone_paths, jn_paths_for_dupl, dupl_nb, comm,
      conformize=False, apply_to_fields = False):
  """
  Function to duplicate n times a set of connected zones
  > dist_tree : distributed tree from wich 'zones' come and in wich duplicated zones will be added
  > zone_paths : list of pathes (BaseName/ZoneName) of the connected zones to duplicate
  > jn_paths_for_dupl : list of 2 lists (listA,listB) where listA (resp listB) is the list 
                               that contains all GridConnectivity nodes defining the first (resp 
                               second) part of a periodic matching
  > dupl_nb : is the number of duplication apply to 'zones'
  > conformize : if True, compute the coordinates mean of each connected vertices and this mean replace
                 the previous coordinates for each vertices. In this case, the matching is perfect.
  > comm : MPI communicator
  > apply_to_fields : apply only the rotation to all vector fields in CGNS nodes of type : 
                      "FlowSolution_t", "DiscreteData_t", "ZoneSubRegion_t", "BCDataset_t"  
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

  if dupl_nb < 0:
    return

  jn_paths_a, jn_paths_b = jn_paths_for_dupl
  zones = [PT.get_node_from_path(dist_tree, path) for path in zone_paths]

  #Store initial values of joins
  jn_values_a = [PT.get_value(PT.get_node_from_path(dist_tree,jn_path_a)) for jn_path_a in jn_paths_a]
  jn_values_b = [PT.get_value(PT.get_node_from_path(dist_tree,jn_path_b)) for jn_path_b in jn_paths_b]

  # Prepare matching jns
  if conformize:
    jn_to_opp = {}
    for i, jn_path_a in enumerate(jn_paths_a):
      jn_path_b = MJT.get_jn_donor_path(dist_tree, jn_path_a)
      assert jn_path_b in jn_paths_b
      jn_to_opp[jn_path_a] = jn_path_b

  # Get first join in the first list of joins (A)
  first_join_in_matchs_a = PT.get_node_from_path(dist_tree, jn_paths_a[0])
  
  # Get transformation information
  rotation_center_a, rotation_angle_a, translation_a = PT.GridConnectivity.get_perio_values(first_join_in_matchs_a)
  
  # Store initial periodicity information of joins of the second joins list (B)
  jn_b_properties = []
  for jn_path_b in jn_paths_b:
    jn_b_init_node = PT.get_node_from_path(dist_tree, jn_path_b)
    jn_b_property  = PT.get_child_from_label(jn_b_init_node, "GridConnectivityProperty_t")
    jn_b_properties.append(PT.deep_copy(jn_b_property))

  # Get the name of all zones to duplicate in order to update the value of GridConnectivity
  # nodes not involved in the duplication (not in jn_paths_for_dupl)
  gc_values_to_update = zone_paths + [PT.get_name(zone) for zone in zones] #Manage both ways BaseName/ZoneName + ZoneName

  gc_predicate = ["ZoneGridConnectivity_t",
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
      base = PT.get_child_from_name(dist_tree, base_name)
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
          gc_value = ".D0".join(PT.get_value(gc).split(".D0")[0:-1])
          if gc_value in gc_values_to_update:
            PT.set_value(gc, f"{gc_value}.D{n+1}")

      # Add duplicated zone to the suitable base
      PT.add_child(base, duplicated_zone)

    # Transform periodic joins of the second joins list (B) from previous set of zones
    # to non periodic joins
    for jb, jn_path_b in enumerate(jn_paths_b):
      jn_path_b_prev = PT.update_path_elt(jn_path_b, 1, lambda zn : zn + f".D{n}")
      jn_b_prev_node = PT.get_node_from_path(dist_tree, jn_path_b_prev)
      PT.rm_children_from_label(jn_b_prev_node, "GridConnectivityProperty_t")
      PT.set_value(jn_b_prev_node, f"{jn_values_b[jb]}.D{n+1}")

    # Transform periodic joins of the fisrt joins list (A) from current set of zones
    # to non periodic joins
    for ja, jn_path_a in enumerate(jn_paths_a):
      jn_path_a_curr = PT.update_path_elt(jn_path_a, 1, lambda zn : zn + f".D{n+1}")
      jn_a_curr_node = PT.get_node_from_path(dist_tree, jn_path_a_curr)
      PT.rm_children_from_label(jn_a_curr_node, "GridConnectivityProperty_t")
      PT.set_value(jn_a_curr_node, f"{jn_values_a[ja]}.D{n}")

    if conformize:
      for jn_path_a, jn_path_b in jn_to_opp.items():
        jn_path_a_curr = PT.update_path_elt(jn_path_a, 1, lambda zn : zn + f".D{n+1}")
        jn_path_b_prev = PT.update_path_elt(jn_path_b, 1, lambda zn : zn + f".D{n}")
        CCJ.conformize_jn_pair(dist_tree, [jn_path_a_curr, jn_path_b_prev], comm)

  # Update information for joins of the fisrt joins list (A) from initial set of zones
  for ja, jn_path_a in enumerate(jn_paths_a):
    jn_path_a_init = PT.update_path_elt(jn_path_a, 1, lambda zn : zn + ".D0")
    jn_a_init_node = PT.get_node_from_path(dist_tree, jn_path_a_init)
    gcp_a_init = PT.get_child_from_label(jn_a_init_node, "GridConnectivityProperty_t")
    rotation_angle_a_node = PT.get_node_from_name(gcp_a_init, "RotationAngle", depth=2)
    translation_a_node    = PT.get_node_from_name(gcp_a_init, "Translation", depth=2)
    PT.set_value(rotation_angle_a_node, PT.get_value(rotation_angle_a_node) * (dupl_nb+1))
    PT.set_value(translation_a_node,    PT.get_value(translation_a_node)    * (dupl_nb+1))
    PT.set_value(jn_a_init_node, f"{jn_values_a[ja]}.D{dupl_nb}")

  # Update information for joins of the second joins list (B) from last set of duplicated zones
  for jb, jn_path_b in enumerate(jn_paths_b):
    jn_path_b_last = PT.update_path_elt(jn_path_b, 1, lambda zn : zn + f".D{dupl_nb}")
    jn_b_last_node = PT.get_node_from_path(dist_tree, jn_path_b_last)
    PT.rm_children_from_label(jn_b_last_node, 'GridConnectivityProperty_t')
    PT.add_child(jn_b_last_node, jn_b_properties[jb])
    gcp_b_last = PT.get_child_from_label(jn_b_last_node, "GridConnectivityProperty_t")
    rotation_angle_b_node = PT.get_node_from_name(gcp_b_last, "RotationAngle", depth=2)
    translation_b_node    = PT.get_node_from_name(gcp_b_last, "Translation", depth=2)
    PT.set_value(rotation_angle_b_node, PT.get_value(rotation_angle_b_node) * (dupl_nb+1))
    PT.set_value(translation_b_node,    PT.get_value(translation_b_node)    * (dupl_nb+1))
    PT.set_value(jn_b_last_node, f"{jn_values_b[jb]}.D0")
  

def duplicate_from_rotation_jns_to_360(dist_tree, zone_paths, jn_paths_for_dupl, comm,
      conformize=False, apply_to_fields=False):
  """Reconstitute a circular mesh from an angular section of the geometry.

  Input tree is modified inplace.

  Args:
    dist_tree (CGNSTree): Input distributed tree
    zone_paths (list of str): List of pathes (BaseName/ZoneName) of the connected zones to duplicate
    jn_paths_for_dupl (pair of list of str): (listA, listB) where listA (resp. list B) stores all the
        pathes of the GridConnectivity nodes defining the first (resp. second) side of a periodic match.
    comm       (MPIComm) : MPI communicator
    conformize (bool, optional): If true, ensure that the generated interface vertices have exactly same
        coordinates (see :func:`conformize_jn_pair`). Defaults to False.
    apply_to_fields (bool, optional): See :func:`maia.algo.transform_affine`. Defaults to False.

  """
  
  if conformize:
    jn_to_opp = {}
    for i, jn_path_a in enumerate(jn_paths_for_dupl[0]):
      jn_path_b = MJT.get_jn_donor_path(dist_tree, jn_path_a)
      assert jn_path_b in jn_paths_for_dupl[1]
      jn_to_opp[jn_path_a] = jn_path_b
    _jn_paths_for_dupl = [ [], [] ]
    for path, path_opp in jn_to_opp.items():
      _jn_paths_for_dupl[0].append(path)
      _jn_paths_for_dupl[1].append(path_opp)
  else:
    _jn_paths_for_dupl = jn_paths_for_dupl

  # Get first join in the first list of joins (A)
  first_join_in_matchs_a = PT.get_node_from_path(dist_tree, _jn_paths_for_dupl[0][0])
  
  # Get transformation information
  rotation_center_a, rotation_angle_a, translation_a = PT.GridConnectivity.get_perio_values(first_join_in_matchs_a)
  
  if (translation_a != np.array([0.,0.,0.])).any():
    raise ValueError("The join is not periodic only by rotation !")

  # Find the number of duplication needed
  index = np.where(rotation_angle_a != 0)[0]
  if index.size == 1:
    sectors_number = abs(int(np.round(2*np.pi/rotation_angle_a[index])))
    rotation_angle_a[index] = np.sign(rotation_angle_a[index]) * 2*np.pi/sectors_number
  else:
    raise ValueError("Zone/Join not define a section of a row")

  # Duplicate 'sectors_number - 1' times the list of zones 'zones'
  duplicate_from_periodic_jns(dist_tree, zone_paths, _jn_paths_for_dupl, sectors_number-1, 
      comm, conformize, apply_to_fields)

  # Transform periodic joins of the fisrt joins list (A) from initial set of zones
  # to non periodic joins
  for jn_path_a in _jn_paths_for_dupl[0]:
    jn_path_a_init = PT.update_path_elt(jn_path_a, 1, lambda zn : zn + f".D{0}")
    jn_a_init_node = PT.get_node_from_path(dist_tree, jn_path_a_init)
    PT.rm_children_from_label(jn_a_init_node, "GridConnectivityProperty_t")

  # Transform periodic joins of the second joins list (B) from last set of duplicated zones
  # to non periodic joins
  for jn_path_b in _jn_paths_for_dupl[1]:
    jn_path_b_last = PT.update_path_elt(jn_path_b, 1, lambda zn : zn + f".D{sectors_number-1}")
    jn_b_last_node = PT.get_node_from_path(dist_tree, jn_path_b_last)
    PT.rm_children_from_label(jn_b_last_node, "GridConnectivityProperty_t")

  if conformize:
    # Conformize last, other have been conformized in duplicate_from_periodic_jns
    for jn_path_a, jn_path_b in jn_to_opp.items():
      jn_path_a_init = PT.update_path_elt(jn_path_a, 1, lambda zn : zn + f".D{0}")
      jn_path_b_last = PT.update_path_elt(jn_path_b, 1, lambda zn : zn + f".D{sectors_number-1}")
      CCJ.conformize_jn_pair(dist_tree, [jn_path_a_init, jn_path_b_last], comm)
