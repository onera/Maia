import Converter.Internal as I
import numpy as np
import copy

import maia.geometry.geometry          as GEO
import maia.connectivity.conformize_jn as CCJ


def _find_cartesian_vector_names_from_names(names):
  suffix_names_x = []
  suffix_names_y = []
  suffix_names_z = []
  for name in names:
    if name[-1] == "X":
      suffix_names_x.append(name[0:-1])
    elif name[-1] == "Y":
      suffix_names_y.append(name[0:-1])
    elif name[-1] == "Z":
      suffix_names_z.append(name[0:-1])

  return sorted(set(suffix_names_x)&set(suffix_names_y)&set(suffix_names_z))


def duplicate_zone_with_transformation(zone,duplicated_zone_name,
                                       rotation_center = np.array([0.,0.,0.]),
                                       rotation_angle  = np.array([0.,0.,0.]),
                                       translation     = np.array([0.,0.,0.]),
                                       max_ordinal     = 0,
                                       apply_to_fields = False):
  # Duplication de la zone
  duplicated_zone = copy.deepcopy(zone)
  I.setName(duplicated_zone,duplicated_zone_name)
  
  # Apply transformation
  duplicated_coords_n  = I.getNodeFromType1(duplicated_zone, "GridCoordinates_t")
  assert(duplicated_coords_n is not None)
  duplicated_coord_x_n  = I.getNodeFromName1(duplicated_coords_n, "CoordinateX")
  duplicated_coord_y_n  = I.getNodeFromName1(duplicated_coords_n, "CoordinateY")
  duplicated_coord_z_n  = I.getNodeFromName1(duplicated_coords_n, "CoordinateZ")
  
  modified_coord_x, modified_coord_y, modified_coord_z = \
                    GEO.apply_transformation_on_separated_components_of_cartesian_vectors(
                        rotation_center, rotation_angle, translation,
                        I.getVal(duplicated_coord_x_n),
                        I.getVal(duplicated_coord_y_n),
                        I.getVal(duplicated_coord_z_n))

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
      cartesian_vectors_basenames = _find_cartesian_vector_names_from_names(data_names)
      for basename in cartesian_vectors_basenames:
        vector_x_n = I.getNodeFromNameAndType(fields_node, basename+"X", "DataArray_t")
        vector_y_n = I.getNodeFromNameAndType(fields_node, basename+"Y", "DataArray_t")
        vector_z_n = I.getNodeFromNameAndType(fields_node, basename+"Z", "DataArray_t")
        # Assume that vectors are position independant
        # Be careful, if coordinates vector needs to be transform, the translation is not apply !
        modified_vector_x, modified_vector_y, modified_vector_z = \
                    GEO.apply_transformation_on_separated_components_of_cartesian_vectors(
                        rotation_center, rotation_angle, np.zeros(3),
                        I.getVal(vector_x_n),
                        I.getVal(vector_y_n),
                        I.getVal(vector_z_n))
        I.setValue(vector_x_n,modified_vector_x)
        I.setValue(vector_y_n,modified_vector_y)
        I.setValue(vector_z_n,modified_vector_z)

  return duplicated_zone


def duplicate_n_zones_from_periodic_join(dist_tree,zones,jn_for_duplication_paths,
                                         duplication_number=1,
                                         conformize=False,comm=None,
                                         apply_to_fields = False):
  
  #############
  # JN = (MatchA,MatchB) => first = MatchA et second = MatchB
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
  #############

  if duplication_number<0:
    return

  # Récupération de la base
  zone0_path = I.getPath(dist_tree,zones[0])
  base_path  = "/".join(zone0_path.split("/")[:-1])
  base      = I.getNodeFromPath(dist_tree,base_path)

  # Récupération du premier raccord de l'ensemble A
  first_join_in_matchs_a = I.getNodeFromPath(dist_tree,jn_for_duplication_paths[0][0])
  
  # Récupération des paramètres de transformation
  gcp_a = I.getNodeFromType1(first_join_in_matchs_a, "GridConnectivityProperty_t")
  rotation_center_a = I.getVal(I.getNodeFromName2(gcp_a, "RotationCenter"))
  rotation_angle_a  = I.getVal(I.getNodeFromName2(gcp_a, "RotationAngle"))
  translation_a     = I.getVal(I.getNodeFromName2(gcp_a, "Translation"))
  
  # Sauvegarde des informations de périodicité des raccords périodiques
  # "B" initiaux
  jn_b_properties = [None]*len(jn_for_duplication_paths[1])
  for jn,jn_path_b in enumerate(jn_for_duplication_paths[1]):
    jn_b_init_node = I.getNodeFromPath(dist_tree, jn_path_b)
    gcp_b_init = copy.deepcopy(I.getNodeFromType1(jn_b_init_node, "GridConnectivityProperty_t"))
    jn_b_properties[jn] = gcp_b_init

  # Recuperation des zones a dupliquer pour permettre de mettre à jour les raccords
  # entre ces zones qui ne sont pas dans jn_for_duplication_paths
  # Attention : les valeurs des raccords peuvent être sous deux formes :
  #             ZoneName ou BaseName/zoneName
  zones_prefixes      = [None]*len(zones)
  gc_values_to_update = [None]*len(zones)
  for z,zone in enumerate(zones):
    zones_prefixes[z]      = copy.deepcopy(I.getName(zone))
    gc_values_to_update[z] = "/".join(I.getPath(dist_tree,zone).split("/")[-2:])
    I.setName(zone,zones_prefixes[z]+".D0")
  gc_values_to_update += zones_prefixes

  # Mise à jour dees raccords entre ces zones qui ne sont pas dans jn_for_duplication_paths
  for zone in zones:
    zgc  = I.getNodeFromType1(zone,"ZoneGridConnectivity_t")
    for gc in I.getNodesFromType1(zgc,"GridConnectivity_t") \
            + I.getNodesFromType1(zgc,"GridConnectivity1to1_t"):
      gc_path = I.getPath(dist_tree,gc,pyCGNSLike=True)
      gc_path_splitted = gc_path.split("/")
      init_gc_path = "/".join([gc_path_splitted[1],gc_path_splitted[2].split(".D")[0]]+gc_path_splitted[3:])
      if (init_gc_path not in jn_for_duplication_paths[0]) and (init_gc_path not in jn_for_duplication_paths[1]):
        gc_value = I.getValue(gc)
        if gc_value in gc_values_to_update:
          new_gc_value = gc_value+".D0"
          I.setValue(gc,new_gc_value)

  max_ordinal = 0
  for base in I.getBases(dist_tree):
      for zone in I.getZones(base):
        for zgc in I.getNodesFromType1(zone, 'ZoneGridConnectivity_t'):
          for gc in I.getNodesFromType1(zgc, 'GridConnectivity_t')+I.getNodesFromType1(zgc, 'GridConnectivity1to1_t'):
            ordinal_n = I.getNodeFromName(gc, 'Ordinal')
            if ordinal_n is not None:
              max_ordinal = max(max_ordinal,I.getValue(ordinal_n))

  # Duplication
  for n in range(duplication_number):
    for z,zone in enumerate(zones):
      duplicated_zone_name = zones_prefixes[z]+".D{0}".format(n+1)
      duplicated_zone = duplicate_zone_with_transformation(zone,duplicated_zone_name,
                                                   rotation_center = rotation_center_a,
                                                   rotation_angle  = (n+1)*rotation_angle_a,
                                                   translation     = (n+1)*translation_a,
                                                   max_ordinal     = (n+1)*max_ordinal,
                                                   apply_to_fields = apply_to_fields)
  
      # Mise à jour des raccords qui ne sont pas dans jn_for_duplication_paths pour la zone dupliquée
      zgc  = I.getNodeFromType1(duplicated_zone,"ZoneGridConnectivity_t")
      base_name = I.getPath(dist_tree,zone,pyCGNSLike=True)[1:].split("/")[0]
      for gc in I.getNodesFromType1(zgc,"GridConnectivity_t") \
              + I.getNodesFromType1(zgc,"GridConnectivity1to1_t"):
        gc_path = base_name+"/"+I.getPath(duplicated_zone,gc)
        gc_path_splitted = gc_path.split("/")
        init_gc_path = "/".join(gc_path_splitted[0:1]+[gc_path_splitted[1].split(".D")[0]]+gc_path_splitted[2:])
        if (init_gc_path not in jn_for_duplication_paths[0]) and (init_gc_path not in jn_for_duplication_paths[1]):
          gc_value = I.getValue(gc).split(".D0")[0]
          if gc_value in gc_values_to_update:
            new_gc_value = gc_value+".D{0}".format(n+1)
            I.setValue(gc,new_gc_value)

      # Ajout de la zone dupliquée dans la base
      I._addChild(base,duplicated_zone)

    #> Transformation des raccords périodiques "B" de l'ensemble de zones
    #  précédent en raccords match
    for jn_path_b in jn_for_duplication_paths[1]:
      split_jn_path_b = jn_path_b.split("/")
      jn_path_b_prev = "/".join(split_jn_path_b[0:2]) + ".D{0}/".format(n) \
                     + "/".join(split_jn_path_b[2:])
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

    #> Transformation des raccords périodiques "A" de l'ensemble de zones
    #  courant en raccords match
    for jn_path_a in jn_for_duplication_paths[0]:
      split_jn_path_a = jn_path_a.split("/")
      jn_path_a_curr = "/".join(split_jn_path_a[0:2]) + ".D{0}/".format(n+1) \
                     + "/".join(split_jn_path_a[2:])
      jn_a_curr_node = I.getNodeFromPath(dist_tree, jn_path_a_curr)
      gcp_a_curr = I.getNodeFromType1(jn_a_curr_node, "GridConnectivityProperty_t")
      I._rmNode(jn_a_curr_node,gcp_a_curr)
      I.setValue(jn_a_curr_node,I.getValue(jn_a_curr_node)+".D{0}".format(n))
      ordinal_opp_n = I.getNodeFromName(gc, 'OrdinalOpp')
      if ordinal_opp_n is not None:
        I.setValue(ordinal_opp_n,I.getValue(ordinal_opp_n)-max_ordinal)

    if conformize:
      # TO DO : adapt for multi zones !!!
      if comm is None:
        raise ValueError("MPI communicator is mandatory for conformization !")
      # jn_for_duplication_paths = []
      # jn_for_duplication_paths.append(I.getPath(dist_tree,secondJoinPrevNode,pyCGNSLike=True)[1:])
      # jn_for_duplication_paths.append(I.getPath(dist_tree,firstJoinDupNode,pyCGNSLike=True)[1:])
      # CCJ.conformize_jn(dist_tree,jn_for_duplication_paths,comm)
  
  #> Mise à jour des raccords périodiques "A" de l'ensemble de zones initial
  for jn_path_a in jn_for_duplication_paths[0]:
    split_jn_path_a = jn_path_a.split("/")
    jn_path_a_init = "/".join(split_jn_path_a[0:2]) + ".D0/" \
                   + "/".join(split_jn_path_a[2:])
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

  #> Mise à jour des raccords périodiques "B" du dernier ensemble de zones dupliqué
  for jn,jn_path_b in enumerate(jn_for_duplication_paths[1]):
    split_jn_path_b = jn_path_b.split("/")
    jn_path_b_last = "/".join(split_jn_path_b[0:2]) + ".D{0}/".format(duplication_number) \
                   + "/".join(split_jn_path_b[2:])
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
  
  #############
  ##### TODO
  ##### > corriger les coordonnées des noeuds de la dernière zone pour assurer le match !
  #############
  
  # Récupération de la base
  zone0_path = I.getPath(dist_tree,zones[0])
  base_path  = "/".join(zone0_path.split("/")[:-1])
  base      = I.getNodeFromPath(dist_tree,base_path)

  # Récupération du premier raccord de l'ensemble A
  first_join_in_matchs_a = I.getNodeFromPath(dist_tree,jn_for_duplication_paths[0][0])
  
  # Récupération des paramètres de transformation
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
    # TO DO : vérifier le type de l'erreur
    raise ValueError("Zone/Join not define a section of a row")
  
  # Duplications
  duplicate_n_zones_from_periodic_join(dist_tree,zones,jn_for_duplication_paths,
                                       duplication_number=sectors_number-1,
                                       conformize=conformize,comm=comm,
                                       apply_to_fields=apply_to_fields)

  #> Transformation des raccords périodiques "A" de l'ensemble de zones initial
  #  en raccords match
  for jn_path_a in jn_for_duplication_paths[0]:
    split_jn_path_a = jn_path_a.split("/")
    jn_path_a_init = "/".join(split_jn_path_a[0:2]) + ".D0/" \
                   + "/".join(split_jn_path_a[2:])
    jn_a_init_node = I.getNodeFromPath(dist_tree, jn_path_a_init)
    gcp_a_init = I.getNodeFromType1(jn_a_init_node, "GridConnectivityProperty_t")
    I._rmNode(jn_a_init_node,gcp_a_init)

  #> Transformation des raccords périodiques "B" du dernier ensemble de zones dupliqué
  #  en raccords match non nécessaire car raccords déjà match par cionstruction
  for jn_path_b in jn_for_duplication_paths[1]:
    split_jn_path_b = jn_path_b.split("/")
    jn_path_b_last = "/".join(split_jn_path_b[0:2]) + ".D{0}/".format(sectors_number-1) \
                   + "/".join(split_jn_path_b[2:])
    jn_b_last_node = I.getNodeFromPath(dist_tree, jn_path_b_last)
    gcp_b_last = I.getNodeFromType1(jn_b_last_node, "GridConnectivityProperty_t")
    I._rmNode(jn_b_last_node,gcp_b_last)

  
  
  if conformize:
	# TO DO : adapt for multi zones !!!
	# Je pense que le conformize n'est pas utile ici car fait dans "_duplicate_n_zones_from_periodic_join()"
  # Faire uniquement le conformize pour le dernier raccord entre zones{0} et zones{N-1}
    if comm is None:
      raise ValueError("MPI communicator is mandatory for conformization !")
    #jn_for_duplication_paths = []
    #jn_for_duplication_paths.append(I.getPath(dist_tree,firstJoinNode,pyCGNSLike=True)[1:])
    #jn_for_duplication_paths.append(I.getPath(dist_tree,finalSecondJoinNode,pyCGNSLike=True)[1:])
    #CCJ.conformize_jn(dist_tree,jn_for_duplication_paths,comm)
  
