import Converter.Internal as I
import numpy as np
import copy

from   maia.utils import py_utils
from   maia.sids  import pytree as PT
import maia.geometry.geometry          as GEO
import maia.connectivity.conformize_jn as CCJ


def duplicate_zone_with_transformation(zone,duplicated_zone_name,
                                       rotation_center = np.array([0.,0.,0.]),
                                       rotation_angle  = np.array([0.,0.,0.]),
                                       translation     = np.array([0.,0.,0.]),
                                       max_ordinal     = 0,
                                       apply_to_fields = False):
  """
  Function to create a new zone by duplication of a intial zone with prescribed transformation
  > zone : zone will be duplicated
  > duplicated_zone_name : name of the zone created by duplication
  > rotation_center : center coordinates of the rotation
  > rotation_angle : angles of the rotation 
  > translation : translation vector components
  > max_ordinal : if max_ordinal > 0, shift ordinal with max_ordinal value in duplicated zone
  > apply_to_fields : apply only the rotation to all vector fields in CGNS nodes of type : 
                      "FlowSolution_t", "DiscreteData_t", "ZoneSubRegion_t", "BCDataset_t"
  """
  
  # Zone duplication
  duplicated_zone = copy.deepcopy(zone)
  I.setName(duplicated_zone,duplicated_zone_name)
  
  # Apply transformation
  duplicated_coords_n  = I.getNodeFromType1(duplicated_zone, "GridCoordinates_t")
  assert(duplicated_coords_n is not None)
  duplicated_coord_x_n  = I.getNodeFromName1(duplicated_coords_n, "CoordinateX")
  duplicated_coord_y_n  = I.getNodeFromName1(duplicated_coords_n, "CoordinateY")
  duplicated_coord_z_n  = I.getNodeFromName1(duplicated_coords_n, "CoordinateZ")
  duplicated_coords = [I.getVal(n) for n in [duplicated_coord_x_n, duplicated_coord_y_n, duplicated_coord_z_n]]
  
  modified_coord_x, modified_coord_y, modified_coord_z = GEO.transform_cart_vectors(
      *duplicated_coords, translation, rotation_center, rotation_angle)

  I.setValue(duplicated_coord_x_n,modified_coord_x)
  I.setValue(duplicated_coord_y_n,modified_coord_y)
  I.setValue(duplicated_coord_z_n,modified_coord_z)
  
  if max_ordinal>0:
    for zgc in I.getNodesFromType1(duplicated_zone, 'ZoneGridConnectivity_t'):
      gcs = I.getNodesFromType1(zgc, 'GridConnectivity_t') \
          + I.getNodesFromType1(zgc, 'GridConnectivity1to1_t')
      for gc in gcs:
        ordinal_n     = I.getNodeFromName(gc, 'Ordinal')
        ordinal_opp_n = I.getNodeFromName(gc, 'OrdinalOpp')
        I.setValue(ordinal_n,    I.getValue(ordinal_n)    +max_ordinal)
        I.setValue(ordinal_opp_n,I.getValue(ordinal_opp_n)+max_ordinal)
        
  if apply_to_fields:
    fields_nodes = []
    fields_nodes += I.getNodesFromType1(duplicated_zone, "FlowSolution_t")
    fields_nodes += I.getNodesFromType1(duplicated_zone, "DiscreteData_t")
    fields_nodes += I.getNodesFromType1(duplicated_zone, "ZoneSubRegion_t")
    zoneBC = I.getNodeFromType1(duplicated_zone, "ZoneBC_t")
    if zoneBC:
      for bc in I.getNodesFromType1(zoneBC, "BC_t"):
        fields_nodes += I.getNodesFromType1(bc, "BCDataSet_t")
    for fields_node in fields_nodes:
      data_names = []
      for data_array in I.getNodesFromType(fields_node, "DataArray_t"):
        data_names.append(I.getName(data_array))
      cartesian_vectors_basenames = py_utils.find_cartesian_vector_names(data_names)
      for basename in cartesian_vectors_basenames:
        vector_x_n = I.getNodeFromNameAndType(fields_node, basename+"X", "DataArray_t")
        vector_y_n = I.getNodeFromNameAndType(fields_node, basename+"Y", "DataArray_t")
        vector_z_n = I.getNodeFromNameAndType(fields_node, basename+"Z", "DataArray_t")
        vectors = [I.getVal(n) for n in [vector_x_n, vector_y_n, vector_z_n]]
        # Assume that vectors are position independant
        # Be careful, if coordinates vector needs to be transform, the translation is not apply !
        modified_vector_x, modified_vector_y, modified_vector_z = GEO.transform_cart_vectors(
            *vectors, rotation_center=rotation_center, rotation_angle=rotation_angle)
        I.setValue(vector_x_n,modified_vector_x)
        I.setValue(vector_y_n,modified_vector_y)
        I.setValue(vector_z_n,modified_vector_z)

  return duplicated_zone


def duplicate_n_zones_from_periodic_join(dist_tree,zones,jn_for_duplication_paths,
                                         duplication_number=1,
                                         conformize=False,comm=None,
                                         apply_to_fields = False):
  """
  Function to duplicate n times a set of connected zones
  > dist_tree : distributed tree from wich 'zones' come and in wich duplicated zones will be added
  > zones : list of connected zones to duplicate
  > jn_for_duplication_paths : list of 2 lists (listA,listB) where listA (resp listB) is the list 
                               that contains all GridConnectivity nodes defining the first (resp 
                               second) part of a periodic matching
  > duplication_number : is the number of duplication apply to 'zones'
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

  if duplication_number<0:
    return

  # Prepare matching jns
  if conformize:
    jn_to_opp = {}
    for i, jn_path_a in enumerate(jn_for_duplication_paths[0]):
      ord_opp = I.getNodeFromName1(I.getNodeFromPath(dist_tree, jn_path_a), 'OrdinalOpp')[1][0]
      jn_path_b = None
      for jn in py_utils.loop_from(jn_for_duplication_paths[1], i): # May stop earlier if jns are well ordered
        ord = I.getNodeFromName1(I.getNodeFromPath(dist_tree, jn), 'Ordinal')[1][0]
        if ord == ord_opp:
          jn_path_b = jn
          break
      assert jn_path_b is not None
      jn_to_opp[jn_path_a] = jn_path_b

  # Get first join in the first list of joins (A)
  first_join_in_matchs_a = I.getNodeFromPath(dist_tree,jn_for_duplication_paths[0][0])
  
  # Get transformation information
  gcp_a = I.getNodeFromType1(first_join_in_matchs_a, "GridConnectivityProperty_t")
  rotation_center_a = I.getVal(I.getNodeFromName2(gcp_a, "RotationCenter"))
  rotation_angle_a  = I.getVal(I.getNodeFromName2(gcp_a, "RotationAngle"))
  translation_a     = I.getVal(I.getNodeFromName2(gcp_a, "Translation"))
  
  # Store initial periodicity information of joins of the second joins list (B)
  jn_b_properties = [None]*len(jn_for_duplication_paths[1])
  for jn,jn_path_b in enumerate(jn_for_duplication_paths[1]):
    jn_b_init_node = I.getNodeFromPath(dist_tree, jn_path_b)
    gcp_b_init = copy.deepcopy(I.getNodeFromType1(jn_b_init_node, "GridConnectivityProperty_t"))
    jn_b_properties[jn] = gcp_b_init

  # Get the name of all zones to duplicate in order to update the value of GridConnectivity
  # nodes not involved in the duplication (not in jn_for_duplication_paths)
  # WARNING : a value of a GridConnectivity node could be described in 2 ways : 
  #           'ZoneName' or 'BaseName/ZoneName'
  zones_prefixes      = [I.getName(zone) for zone in zones]
  gc_values_to_update = []
  for zone in zones:
    gc_values_to_update.append(PT.path_tail(I.getPath(dist_tree, zone), 1))
    I.setName(zone, I.getName(zone)+".D0")
  gc_values_to_update += zones_prefixes

  # Update the value of all GridConnectivity nodes not involved in the duplication from initial zones
  for zone in zones:
    zgc  = I.getNodeFromType1(zone,"ZoneGridConnectivity_t")
    for gc in I.getNodesFromType1(zgc,"GridConnectivity_t") \
            + I.getNodesFromType1(zgc,"GridConnectivity1to1_t"):
      gc_path = I.getPath(dist_tree,gc,pyCGNSLike=True)[1:] #To replace
      init_gc_path = PT.update_path_elt(gc_path, 1, lambda zn: zn.split(".D")[0])
      if (init_gc_path not in jn_for_duplication_paths[0]) and (init_gc_path not in jn_for_duplication_paths[1]):
        gc_value = I.getValue(gc)
        if gc_value in gc_values_to_update:
          new_gc_value = gc_value+".D0"
          I.setValue(gc,new_gc_value)
  
  # Search the maximum of 'Ordinal' number in the whole dist_tree
  max_ordinal = 0
  for base in I.getBases(dist_tree):
      for zone in I.getZones(base):
        for zgc in I.getNodesFromType1(zone, 'ZoneGridConnectivity_t'):
          for gc in I.getNodesFromType1(zgc, 'GridConnectivity_t')+I.getNodesFromType1(zgc, 'GridConnectivity1to1_t'):
            ordinal_n = I.getNodeFromName(gc, 'Ordinal')
            if ordinal_n is not None:
              max_ordinal = max(max_ordinal,I.getValue(ordinal_n))

  # Duplicate 'duplication_number' times the list of zones 'zones'
  for n in range(duplication_number):
    for z,zone in enumerate(zones):
      duplicated_zone_name = zones_prefixes[z]+".D{0}".format(n+1)
      duplicated_zone = duplicate_zone_with_transformation(zone,duplicated_zone_name,
                                                   rotation_center = rotation_center_a,
                                                   rotation_angle  = (n+1)*rotation_angle_a,
                                                   translation     = (n+1)*translation_a,
                                                   max_ordinal     = (n+1)*max_ordinal,
                                                   apply_to_fields = apply_to_fields)
  
      # Update the value of all GridConnectivity nodes not involved in the duplication from initial zones
      zgc  = I.getNodeFromType1(duplicated_zone,"ZoneGridConnectivity_t")
      base_name = I.getPath(dist_tree,zone,pyCGNSLike=True)[1:].split("/")[0]
      for gc in I.getNodesFromType1(zgc,"GridConnectivity_t") \
              + I.getNodesFromType1(zgc,"GridConnectivity1to1_t"):
        gc_path = base_name+"/"+I.getPath(duplicated_zone,gc)
        init_gc_path = PT.update_path_elt(gc_path, 1, lambda zn: zn.split(".D")[0])
        if (init_gc_path not in jn_for_duplication_paths[0]) and (init_gc_path not in jn_for_duplication_paths[1]):
          gc_value = I.getValue(gc).split(".D0")[0]
          if gc_value in gc_values_to_update:
            new_gc_value = gc_value+".D{0}".format(n+1)
            I.setValue(gc,new_gc_value)

      # Add duplicated zone to the suitable base
      I._addChild(base,duplicated_zone)

    # Transform periodic joins of the second joins list (B) from previous set of zones
    # to non periodic joins
    for jn_path_b in jn_for_duplication_paths[1]:
      jn_path_b_prev = PT.update_path_elt(jn_path_b, 1, lambda zn : zn + f".D{n}")
      jn_b_prev_node = I.getNodeFromPath(dist_tree, jn_path_b_prev)
      gcp_b_prev = I.getNodeFromType1(jn_b_prev_node, "GridConnectivityProperty_t")
      I._rmNode(jn_b_prev_node,gcp_b_prev)
      gc_value = I.getValue(jn_b_prev_node)
      if len(gc_value.split('.D'))>1:
        new_gc_value = ".".join(gc_value.split('.D')[:-1])+".D{0}".format(n+1)
      else:
        new_gc_value = gc_value.split('.D')[0]+".D{0}".format(n+1)
      I.setValue(jn_b_prev_node,new_gc_value)
      ordinal_opp_n = I.getNodeFromName(gc, 'OrdinalOpp')
      if ordinal_opp_n is not None:
        I.setValue(ordinal_opp_n,I.getValue(ordinal_opp_n)+max_ordinal)

    # Transform periodic joins of the fisrt joins list (A) from current set of zones
    # to non periodic joins
    for jn_path_a in jn_for_duplication_paths[0]:
      jn_path_a_curr = PT.update_path_elt(jn_path_a, 1, lambda zn : zn + f".D{n+1}")
      jn_a_curr_node = I.getNodeFromPath(dist_tree, jn_path_a_curr)
      gcp_a_curr = I.getNodeFromType1(jn_a_curr_node, "GridConnectivityProperty_t")
      I._rmNode(jn_a_curr_node,gcp_a_curr)
      I.setValue(jn_a_curr_node,I.getValue(jn_a_curr_node)+".D{0}".format(n))
      ordinal_opp_n = I.getNodeFromName(gc, 'OrdinalOpp')
      if ordinal_opp_n is not None:
        I.setValue(ordinal_opp_n,I.getValue(ordinal_opp_n)-max_ordinal)

    if conformize:
      for jn_path_a, jn_path_b in jn_to_opp.items():
        jn_path_a_curr = PT.update_path_elt(jn_path_a, 1, lambda zn : zn + f".D{n+1}")
        jn_path_b_prev = PT.update_path_elt(jn_path_b, 1, lambda zn : zn + f".D{n}")
        CCJ.conformize_jn(dist_tree, [jn_path_a_curr, jn_path_b_prev], comm)
  
  # Update information for joins of the fisrt joins list (A) from initial set of zones
  for jn_path_a in jn_for_duplication_paths[0]:
    jn_path_a_init = PT.update_path_elt(jn_path_a, 1, lambda zn : zn + ".D0")
    jn_a_init_node = I.getNodeFromPath(dist_tree, jn_path_a_init)
    gcp_a_init = I.getNodeFromType1(jn_a_init_node, "GridConnectivityProperty_t")
    rotation_angle_a_node = I.getNodeFromName2(gcp_a_init, "RotationAngle")
    I.setValue(rotation_angle_a_node, I.getValue(rotation_angle_a_node)*(duplication_number+1))
    translation_a_node = I.getNodeFromName2(gcp_a_init, "Translation")
    I.setValue(translation_a_node, I.getValue(translation_a_node)*(duplication_number+1))
    I.setValue(jn_a_init_node,I.getValue(jn_a_init_node)+".D{0}".format(duplication_number))
    ordinal_opp_n = I.getNodeFromName(gc, 'OrdinalOpp')
    if ordinal_opp_n is not None:
      I.setValue(ordinal_opp_n,I.getValue(ordinal_opp_n)+duplication_number*max_ordinal)

  # Update information for joins of the second joins list (B) from last set of duplicated zones
  for jn,jn_path_b in enumerate(jn_for_duplication_paths[1]):
    jn_path_b_last = PT.update_path_elt(jn_path_b, 1, lambda zn : zn + f".D{duplication_number}")
    jn_b_last_node = I.getNodeFromPath(dist_tree, jn_path_b_last)
    I._addChild(jn_b_last_node,jn_b_properties[jn])
    gcp_b_last = I.getNodeFromType1(jn_b_last_node, "GridConnectivityProperty_t")
    rotation_angle_b_node = I.getNodeFromName2(gcp_b_last, "RotationAngle")
    I.setValue(rotation_angle_b_node, I.getValue(rotation_angle_b_node)*(duplication_number+1))
    translation_b_node = I.getNodeFromName2(gcp_b_last, "Translation")
    I.setValue(translation_b_node, I.getValue(translation_b_node)*(duplication_number+1))
    gc_value = I.getValue(jn_b_last_node)
    if len(gc_value.split('.D'))>1:
      new_gc_value = ".".join(gc_value.split('.D')[:-1])+".D0"
    else:
      new_gc_value = gc_value.split('.D')[0]+".D0"
    I.setValue(jn_b_last_node,new_gc_value)
    ordinal_opp_n = I.getNodeFromName(gc, 'OrdinalOpp')
    if ordinal_opp_n is not None:
      I.setValue(ordinal_opp_n,I.getValue(ordinal_opp_n)-duplication_number*max_ordinal)
  
  
def duplicate_zones_from_periodic_join_by_rotation_to_360(dist_tree,zones,jn_for_duplication_paths,
                                                          conformize=False,comm=None,
                                                          rotation_correction=True,
                                                          apply_to_fields=False):
  """
  Function to duplicate n times a set of connected zones
  > dist_tree : distributed tree from wich 'zones' come and in wich duplicated zones will be added
  > zones : list of connected zones to duplicate
  > jn_for_duplication_paths : list of 2 lists (listA,listB) where listA (resp listB) is the list 
                               that contains all GridConnectivity nodes defining the first (resp 
                               second) part of a periodic matching
  > conformize : if True, compute the coordinates mean of each connected vertices and this mean replace
                 the previous coordinates for each vertices. In this case, the matching is perfect.
  > comm : MPI communicator
  > rotation_correction : if True, recompute the rotation angle by searching an integer number of
                          angular sectors.
  > apply_to_fields : apply only the rotation to all vector fields in CGNS nodes of type : 
                      "FlowSolution_t", "DiscreteData_t", "ZoneSubRegion_t", "BCDataset_t"  
  """
  
  #############
  ##### TODO
  ##### > corriger les coordonnées des noeuds de la dernière zone pour assurer le match !
  #############

  if conformize:
    jn_to_opp = {}
    for i, jn_path_a in enumerate(jn_for_duplication_paths[0]):
      ord_opp = I.getNodeFromName1(I.getNodeFromPath(dist_tree, jn_path_a), 'OrdinalOpp')[1][0]
      jn_path_b = None
      for jn in py_utils.loop_from(jn_for_duplication_paths[1], i): # May stop earlier if jns are well ordered
        ord = I.getNodeFromName1(I.getNodeFromPath(dist_tree, jn), 'Ordinal')[1][0]
        if ord == ord_opp:
          jn_path_b = jn
          break
      assert jn_path_b is not None
      jn_to_opp[jn_path_a] = jn_path_b
    _jn_for_duplication_paths = [ [], [] ]
    for path, path_opp in jn_to_opp.items():
      _jn_for_duplication_paths[0].append(path)
      _jn_for_duplication_paths[1].append(path_opp)
  else:
    _jn_for_duplication_paths = jn_for_duplication_paths

  # Get first join in the first list of joins (A)
  first_join_in_matchs_a = I.getNodeFromPath(dist_tree, _jn_for_duplication_paths[0][0])
  
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
    if rotation_correction:
      rotation_angle_a[index] = np.sign(rotation_angle_a[index])*2*np.pi/sectors_number
  else:
    # TO DO : vérifier le type de l'erreur ??
    raise ValueError("Zone/Join not define a section of a row")
  
  # Duplicate 'sectors_number - 1' times the list of zones 'zones'
  duplicate_n_zones_from_periodic_join(dist_tree,zones, _jn_for_duplication_paths,
                                       duplication_number=sectors_number-1,
                                       conformize=conformize,comm=comm,
                                       apply_to_fields=apply_to_fields)

  # Transform periodic joins of the fisrt joins list (A) from initial set of zones
  # to non periodic joins
  for jn_path_a in _jn_for_duplication_paths[0]:
    jn_path_a_init = PT.update_path_elt(jn_path_a, 1, lambda zn : zn + f".D{0}")
    jn_a_init_node = I.getNodeFromPath(dist_tree, jn_path_a_init)
    gcp_a_init = I.getNodeFromType1(jn_a_init_node, "GridConnectivityProperty_t")
    I._rmNode(jn_a_init_node,gcp_a_init)

  # Transform periodic joins of the second joins list (B) from last set of duplicated zones
  # to non periodic joins
  for jn_path_b in _jn_for_duplication_paths[1]:
    jn_path_b_last = PT.update_path_elt(jn_path_b, 1, lambda zn : zn + f".D{sectors_number-1}")
    jn_b_last_node = I.getNodeFromPath(dist_tree, jn_path_b_last)
    gcp_b_last = I.getNodeFromType1(jn_b_last_node, "GridConnectivityProperty_t")
    I._rmNode(jn_b_last_node,gcp_b_last)

  
  if conformize:
    # Conformize last, other have been conformized in duplicate_n_zones_from_periodic_join
    for jn_path_a, jn_path_b in jn_to_opp.items():
      jn_path_a_init = PT.update_path_elt(jn_path_a, 1, lambda zn : zn + f".D{0}")
      jn_path_b_last = PT.update_path_elt(jn_path_b, 1, lambda zn : zn + f".D{sectors_number-1}")
      CCJ.conformize_jn(dist_tree, [jn_path_a_init, jn_path_b_last], comm)
  
