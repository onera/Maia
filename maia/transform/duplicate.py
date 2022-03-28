import Converter.Internal as I
import numpy as np

from   maia.utils import py_utils
from   maia.sids  import pytree as PT
import maia.geometry.geometry          as GEO
import maia.connectivity.conformize_jn as CCJ

def _get_gc_root_name(gc_name):
  """ Remove the .D### suffix, if existing """
  idx = gc_name.rfind('.D') #Find last occurence
  return gc_name[:idx] if idx > -1 else gc_name

def duplicate_zone_with_transformation(zone, duplicated_zone_name,
                                       rotation_center = np.zeros(3),
                                       rotation_angle  = np.zeros(3),
                                       translation     = np.zeros(3),
                                       max_ordinal     = 0,
                                       apply_to_fields = False):
  """
  Function to create a new zone by duplication of an initial zone with prescribed transformation
  > zone : zone to be duplicated
  > duplicated_zone_name : name of the zone created by duplication
  > rotation_center : center coordinates of the rotation
  > rotation_angle : angles of the rotation 
  > translation : translation vector components
  > max_ordinal : if max_ordinal > 0, shift ordinal with max_ordinal value in duplicated zone
  > apply_to_fields : if True, apply only the rotation to all vector fields in CGNS nodes of type
                "FlowSolution_t", "DiscreteData_t", "ZoneSubRegion_t", "BCDataset_t"
  """
  
  # Zone duplication
  duplicated_zone = I.copyTree(zone)
  I.setName(duplicated_zone, duplicated_zone_name)

  GEO.transform_zone(duplicated_zone, rotation_center, rotation_angle, translation, apply_to_fields)
  
  gc_predicate = ["ZoneGridConnectivity_t",
                  lambda n : I.getType(n) in ["GridConnectivity_t", "GridConnectivity1to1_t"]]

  if max_ordinal > 0:
    for gc in PT.iter_children_from_predicates(duplicated_zone, gc_predicate):
      ordinal_n     = I.getNodeFromName(gc, 'Ordinal')
      ordinal_opp_n = I.getNodeFromName(gc, 'OrdinalOpp')
      I.setValue(ordinal_n,     I.getValue(ordinal_n)    +max_ordinal)
      I.setValue(ordinal_opp_n, I.getValue(ordinal_opp_n)+max_ordinal)
        
  return duplicated_zone

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
  zones = [I.getNodeFromPath(dist_tree, path) for path in zone_paths]

  # Prepare matching jns
  if conformize:
    jn_to_opp = {}
    for i, jn_path_a in enumerate(jn_paths_a):
      ord_opp = I.getNodeFromName1(I.getNodeFromPath(dist_tree, jn_path_a), 'OrdinalOpp')[1][0]
      jn_path_b = None
      for jn in py_utils.loop_from(jn_paths_b, i): # May stop earlier if jns are well ordered
        ord_cur = I.getNodeFromName1(I.getNodeFromPath(dist_tree, jn), 'Ordinal')[1][0]
        if ord_cur == ord_opp:
          jn_path_b = jn
          break
      assert jn_path_b is not None
      jn_to_opp[jn_path_a] = jn_path_b

  # Get first join in the first list of joins (A)
  first_join_in_matchs_a = I.getNodeFromPath(dist_tree, jn_paths_a[0])
  
  # Get transformation information
  gcp_a = I.getNodeFromType1(first_join_in_matchs_a, "GridConnectivityProperty_t")
  rotation_center_a = I.getVal(I.getNodeFromName2(gcp_a, "RotationCenter"))
  rotation_angle_a  = I.getVal(I.getNodeFromName2(gcp_a, "RotationAngle"))
  translation_a     = I.getVal(I.getNodeFromName2(gcp_a, "Translation"))
  
  # Store initial periodicity information of joins of the second joins list (B)
  jn_b_properties = []
  for jn_path_b in jn_paths_b:
    jn_b_init_node = I.getNodeFromPath(dist_tree, jn_path_b)
    jn_b_property  = I.getNodeFromType1(jn_b_init_node, "GridConnectivityProperty_t")
    jn_b_properties.append(I.copyTree(jn_b_property))

  # Get the name of all zones to duplicate in order to update the value of GridConnectivity
  # nodes not involved in the duplication (not in jn_paths_for_dupl)
  gc_values_to_update = zone_paths + [I.getName(zone) for zone in zones] #Manage both ways BaseName/ZoneName + ZoneName

  gc_predicate = ["ZoneGridConnectivity_t",
                  lambda n : I.getType(n) in ["GridConnectivity_t", "GridConnectivity1to1_t"]]

  # Update the value of all GridConnectivity nodes not involved in the duplication from initial zones
  for zone_path, zone in zip(zone_paths, zones):
    for zgc, gc in PT.iter_children_from_predicates(zone, gc_predicate, ancestors=True):
      init_gc_path = f"{zone_path}/{I.getName(zgc)}/{I.getName(gc)}"
      if (init_gc_path not in jn_paths_a) and (init_gc_path not in jn_paths_b):
        gc_value = I.getValue(gc)
        if gc_value in gc_values_to_update:
          I.setValue(gc, f"{gc_value}.D0")
    I.setName(zone, f"{I.getName(zone)}.D0") #Update zone name
  
  # Search the maximum of 'Ordinal' number in the whole dist_tree
  max_ordinal = 0
  for gc in PT.iter_children_from_predicates(dist_tree, ["CGNSBase_t", "Zone_t"] + gc_predicate):
    ordinal_n = I.getNodeFromName(gc, 'Ordinal')
    if ordinal_n is not None:
      max_ordinal = max(max_ordinal, I.getValue(ordinal_n))

  # Duplicate 'dupl_nb' times the list of zones 'zones'
  for n in range(dupl_nb):
    for zone_path, zone in zip(zone_paths, zones):
      base_name, root_zone_name = zone_path.split('/')
      base = I.getNodeFromName1(dist_tree, base_name)
      duplicated_zone = duplicate_zone_with_transformation(zone,
                                                           f"{root_zone_name}.D{n+1}",
                                                           rotation_center = rotation_center_a,
                                                           rotation_angle  = (n+1)*rotation_angle_a,
                                                           translation     = (n+1)*translation_a,
                                                           max_ordinal     = (n+1)*max_ordinal,
                                                           apply_to_fields = apply_to_fields)
  
      # Update the value of all GridConnectivity nodes not involved in the duplication from initial zones
      for zgc, gc in PT.iter_children_from_predicates(duplicated_zone, gc_predicate, ancestors=True):
        gc_path = f"{zone_path}/{I.getName(zgc)}/{I.getName(gc)}"
        init_gc_path = PT.update_path_elt(gc_path, 1, lambda zn: zn.split(".D")[0])
        if (init_gc_path not in jn_paths_a) and (init_gc_path not in jn_paths_b):
          gc_value = I.getValue(gc).split(".D0")[0]
          if gc_value in gc_values_to_update:
            I.setValue(gc, f"{gc_value}.D{n+1}")

      # Add duplicated zone to the suitable base
      I._addChild(base, duplicated_zone)

    # Transform periodic joins of the second joins list (B) from previous set of zones
    # to non periodic joins
    for jn_path_b in jn_paths_b:
      jn_path_b_prev = PT.update_path_elt(jn_path_b, 1, lambda zn : zn + f".D{n}")
      jn_b_prev_node = I.getNodeFromPath(dist_tree, jn_path_b_prev)
      I._rmNodesByType1(jn_b_prev_node, "GridConnectivityProperty_t")
      gc_value = I.getValue(jn_b_prev_node)
      I.setValue(jn_b_prev_node, f"{_get_gc_root_name(gc_value)}.D{n+1}")
      ordinal_opp_n = I.getNodeFromName(gc, 'OrdinalOpp')
      if ordinal_opp_n is not None:
        I.setValue(ordinal_opp_n, I.getValue(ordinal_opp_n) + max_ordinal)

    # Transform periodic joins of the fisrt joins list (A) from current set of zones
    # to non periodic joins
    for jn_path_a in jn_paths_a:
      jn_path_a_curr = PT.update_path_elt(jn_path_a, 1, lambda zn : zn + f".D{n+1}")
      jn_a_curr_node = I.getNodeFromPath(dist_tree, jn_path_a_curr)
      I._rmNodesByType1(jn_a_curr_node, "GridConnectivityProperty_t")
      I.setValue(jn_a_curr_node, f"{I.getValue(jn_a_curr_node)}.D{n}")
      ordinal_opp_n = I.getNodeFromName(gc, 'OrdinalOpp')
      if ordinal_opp_n is not None:
        I.setValue(ordinal_opp_n, I.getValue(ordinal_opp_n) - max_ordinal)

    if conformize:
      for jn_path_a, jn_path_b in jn_to_opp.items():
        jn_path_a_curr = PT.update_path_elt(jn_path_a, 1, lambda zn : zn + f".D{n+1}")
        jn_path_b_prev = PT.update_path_elt(jn_path_b, 1, lambda zn : zn + f".D{n}")
        CCJ.conformize_jn(dist_tree, [jn_path_a_curr, jn_path_b_prev], comm)
  
  # Update information for joins of the fisrt joins list (A) from initial set of zones
  for jn_path_a in jn_paths_a:
    jn_path_a_init = PT.update_path_elt(jn_path_a, 1, lambda zn : zn + ".D0")
    jn_a_init_node = I.getNodeFromPath(dist_tree, jn_path_a_init)
    gcp_a_init = I.getNodeFromType1(jn_a_init_node, "GridConnectivityProperty_t")
    rotation_angle_a_node = I.getNodeFromName2(gcp_a_init, "RotationAngle")
    translation_a_node    = I.getNodeFromName2(gcp_a_init, "Translation")
    I.setValue(rotation_angle_a_node, I.getValue(rotation_angle_a_node) * (dupl_nb+1))
    I.setValue(translation_a_node,    I.getValue(translation_a_node)    * (dupl_nb+1))
    I.setValue(jn_a_init_node, f"{I.getValue(jn_a_init_node)}.D{dupl_nb}")
    ordinal_opp_n = I.getNodeFromName(gc, 'OrdinalOpp')
    if ordinal_opp_n is not None:
      I.setValue(ordinal_opp_n, I.getValue(ordinal_opp_n) + dupl_nb*max_ordinal)

  # Update information for joins of the second joins list (B) from last set of duplicated zones
  for jn, jn_path_b in enumerate(jn_paths_b):
    jn_path_b_last = PT.update_path_elt(jn_path_b, 1, lambda zn : zn + f".D{dupl_nb}")
    jn_b_last_node = I.getNodeFromPath(dist_tree, jn_path_b_last)
    I._addChild(jn_b_last_node, jn_b_properties[jn])
    gcp_b_last = I.getNodeFromType1(jn_b_last_node, "GridConnectivityProperty_t")
    rotation_angle_b_node = I.getNodeFromName2(gcp_b_last, "RotationAngle")
    translation_b_node    = I.getNodeFromName2(gcp_b_last, "Translation")
    I.setValue(rotation_angle_b_node, I.getValue(rotation_angle_b_node) * (dupl_nb+1))
    I.setValue(translation_b_node,    I.getValue(translation_b_node)    * (dupl_nb+1))
    gc_value = I.getValue(jn_b_last_node)
    I.setValue(jn_b_last_node, f"{_get_gc_root_name(gc_value)}.D0")
    ordinal_opp_n = I.getNodeFromName(gc, 'OrdinalOpp')
    if ordinal_opp_n is not None:
      I.setValue(ordinal_opp_n, I.getValue(ordinal_opp_n) - dupl_nb*max_ordinal)
  

def duplicate_from_rotation_jns_to_360(dist_tree, zone_paths, jn_paths_for_dupl, comm,
      conformize=False, apply_to_fields=False):
  """
  Function to duplicate by rotation until we retrieve a 360° case
  > dist_tree : distributed tree from wich 'zones' come and in wich duplicated zones will be added
  > zone_paths : list of pathes (BaseName/ZoneName) of the connected zones to duplicate
  > jn_paths_for_dupl : list of 2 lists (listA,listB) where listA (resp listB) is the list 
                               that contains all GridConnectivity nodes defining the first (resp 
                               second) part of a periodic matching
  > conformize : if True, compute the coordinates mean of each connected vertices and this mean replace
                 the previous coordinates for each vertices. In this case, the matching is perfect.
  > comm : MPI communicator
  > apply_to_fields : apply only the rotation to all vector fields in CGNS nodes of type : 
                      "FlowSolution_t", "DiscreteData_t", "ZoneSubRegion_t", "BCDataset_t"  
  """
  
  if conformize:
    jn_to_opp = {}
    for i, jn_path_a in enumerate(jn_paths_for_dupl[0]):
      ord_opp = I.getNodeFromName1(I.getNodeFromPath(dist_tree, jn_path_a), 'OrdinalOpp')[1][0]
      jn_path_b = None
      for jn in py_utils.loop_from(jn_paths_for_dupl[1], i): # May stop earlier if jns are well ordered
        ord = I.getNodeFromName1(I.getNodeFromPath(dist_tree, jn), 'Ordinal')[1][0]
        if ord == ord_opp:
          jn_path_b = jn
          break
      assert jn_path_b is not None
      jn_to_opp[jn_path_a] = jn_path_b
    _jn_paths_for_dupl = [ [], [] ]
    for path, path_opp in jn_to_opp.items():
      _jn_paths_for_dupl[0].append(path)
      _jn_paths_for_dupl[1].append(path_opp)
  else:
    _jn_paths_for_dupl = jn_paths_for_dupl

  # Get first join in the first list of joins (A)
  first_join_in_matchs_a = I.getNodeFromPath(dist_tree, _jn_paths_for_dupl[0][0])
  
  # Get transformation information
  gcp_a = I.getNodeFromType1(first_join_in_matchs_a, "GridConnectivityProperty_t")
  rotation_center_a = I.getVal(I.getNodeFromName2(gcp_a, "RotationCenter"))
  rotation_angle_a  = I.getVal(I.getNodeFromName2(gcp_a, "RotationAngle"))
  translation_a     = I.getVal(I.getNodeFromName2(gcp_a, "Translation"))
  
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
    jn_a_init_node = I.getNodeFromPath(dist_tree, jn_path_a_init)
    I._rmNodesFromType1(jn_a_init_node, "GridConnectivityProperty_t")

  # Transform periodic joins of the second joins list (B) from last set of duplicated zones
  # to non periodic joins
  for jn_path_b in _jn_paths_for_dupl[1]:
    jn_path_b_last = PT.update_path_elt(jn_path_b, 1, lambda zn : zn + f".D{sectors_number-1}")
    jn_b_last_node = I.getNodeFromPath(dist_tree, jn_path_b_last)
    I._rmNodesFromType1(jn_b_last_node, "GridConnectivityProperty_t")

  if conformize:
    # Conformize last, other have been conformized in duplicate_from_periodic_jns
    for jn_path_a, jn_path_b in jn_to_opp.items():
      jn_path_a_init = PT.update_path_elt(jn_path_a, 1, lambda zn : zn + f".D{0}")
      jn_path_b_last = PT.update_path_elt(jn_path_b, 1, lambda zn : zn + f".D{sectors_number-1}")
      CCJ.conformize_jn(dist_tree, [jn_path_a_init, jn_path_b_last], comm)
  
