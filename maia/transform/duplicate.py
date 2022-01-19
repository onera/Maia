import Converter.Internal as I
import numpy as np
import copy

import maia.geometry.geometry          as GEO
import maia.connectivity.conformize_jn as CCJ


def _find_cartesian_vector_names_from_names(names):
  suffixX_names = []
  suffixY_names = []
  suffixZ_names = []
  for name in names:
    if name[-1] == "X":
      suffixX_names.append(name[0:-1])
    if name[-1] == "Y":
      suffixY_names.append(name[0:-1])
    if name[-1] == "Z":
      suffixZ_names.append(name[0:-1])
  basenames = []
  for name in suffixX_names:
    if (name in suffixY_names) and (name in suffixZ_names):
      basenames.append(name)
  return basenames


def duplicate_zone_with_transformation(zone,nameZoneDup,
                                       rotationCenter         =np.array([0.,0.,0.]),
                                       rotationAngle          =np.array([0.,0.,0.]),
                                       translation            =np.array([0.,0.,0.]),
                                       max_ordinal            =0,
                                       apply_to_flowsolutions = False):
  # Duplication de la zone
  zoneDup     = copy.deepcopy(zone)
  I.setName(zoneDup,nameZoneDup)
  
  # Apply transformation
  coordsDupNode  = I.getNodeFromType1(zoneDup, "GridCoordinates_t")
  assert(coordsDupNode is not None)
  coordXDupNode  = I.getNodeFromName1(coordsDupNode, "CoordinateX")
  coordYDupNode  = I.getNodeFromName1(coordsDupNode, "CoordinateY")
  coordZDupNode  = I.getNodeFromName1(coordsDupNode, "CoordinateZ")
  
  modCx, modCy, modCz = GEO.apply_transformation_on_separated_components_of_cartesian_vectors(rotationCenter, rotationAngle, translation,
                                                                                             I.getValue(coordXDupNode),
                                                                                             I.getValue(coordYDupNode),
                                                                                             I.getValue(coordZDupNode))

  I.setValue(coordXDupNode,modCx)
  I.setValue(coordYDupNode,modCy)
  I.setValue(coordZDupNode,modCz)
  
  if max_ordinal>0:
    for zgc in I.getNodesFromType1(zoneDup, 'ZoneGridConnectivity_t'):
      gcs = I.getNodesFromType1(zgc, 'GridConnectivity_t') + I.getNodesFromType1(zgc, 'GridConnectivity1to1_t')
      for gc in gcs:
        ordinal_n     = I.getNodeFromName(gc, 'Ordinal')
        ordinal_opp_n = I.getNodeFromName(gc, 'OrdinalOpp')
        I.setValue(ordinal_n,    I.getValue(ordinal_n)    +max_ordinal)
        I.setValue(ordinal_opp_n,I.getValue(ordinal_opp_n)+max_ordinal)
        
  if apply_to_flowsolutions:
    for fs in I.getNodesFromType(zoneDup, "FlowSolution_t"):
      data_names = []
      for data_array in I.getNodesFromType(fs, "DataArray_t"):
        data_names.append(I.getName(data_array))
      cartesian_vectors_basenames = _find_cartesian_vector_names_from_names(data_names)
      for basename in cartesian_vectors_basenames:
        vectorXNode = I.getNodeFromNameAndType(fs, basename+"X", "DataArray_t")
        vectorYNode = I.getNodeFromNameAndType(fs, basename+"Y", "DataArray_t")
        vectorZNode = I.getNodeFromNameAndType(fs, basename+"Z", "DataArray_t")
        modVx, modVy, modVz = GEO.apply_transformation_on_separated_components_of_cartesian_vectors(
                                              rotationCenter, rotationAngle, translation,
                                              I.getValue(vectorXNode),
                                              I.getValue(vectorYNode),
                                              I.getValue(vectorZNode))
        I.setValue(vectorXNode,modVx)
        I.setValue(vectorYNode,modVy)
        I.setValue(vectorZNode,modVz)

  return zoneDup


def _duplicate_zone_from_periodic_join(dist_tree,zone,JN_for_duplication_names,
                                       conformize=False,comm=None):
  #############
  ##### TODO
  ##### > gestion des autres raccords...
  ##### > autre chose ?
  #############
  
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

  # Récupération de la base
  pathZone    = I.getPath(dist_tree,zone)
  pathBase    = "/".join(pathZone.split("/")[:-1])
  base        = I.getNodeFromPath(dist_tree,pathBase)
  
  # Changement de nom de la zone dupliquée
  zoneNamePrefix = I.getName(zone)
  zoneName = zoneNamePrefix+".D0"
  I.setName(zone,zoneName)

  # Récupération des raccords
  ZGC               = I.getNodeFromType1(zone,"ZoneGridConnectivity_t")
  firstJoinNode     = I.getNodeFromName1(ZGC,JN_for_duplication_names[0])
  secondJoinNode    = I.getNodeFromName1(ZGC,JN_for_duplication_names[1])
  
  # Duplication de la zone
  #> récupération des paramètres de transformation
  GCP1 = I.getNodeFromType1(firstJoinNode, "GridConnectivityProperty_t")
  rotationCenter1Node = I.getNodeFromName2(GCP1, "RotationCenter")
  rotationAngle1Node  = I.getNodeFromName2(GCP1, "RotationAngle")
  translation1Node    = I.getNodeFromName2(GCP1, "Translation")
  rotationCenter1     = I.getValue(rotationCenter1Node)
  rotationAngle1      = I.getValue(rotationAngle1Node)
  translation1        = I.getValue(translation1Node)
  #> duplication
  zoneDupName = zoneNamePrefix+".D1"
  zoneDup = duplicate_zone_with_transformation(zone,zoneDupName,
                                            rotationCenter=rotationCenter1,
                                            rotationAngle=rotationAngle1,
                                            translation=translation1)
  
  # Mise à jour des raccords
  #> recuperation des raccords dupliqués
  ZGCDup            = I.getNodeFromType1(zoneDup,"ZoneGridConnectivity_t")
  firstJoinDupNode  = I.getNodeFromName1(ZGCDup,JN_for_duplication_names[0])
  secondJoinDupNode = I.getNodeFromName1(ZGCDup,JN_for_duplication_names[1])
  #> mise à jour des raccords initiaux
  #>>>> le raccord sur lequel s'appuie la duplication (first) reste un match péridique
  #     avec la zone dupliquée mais l'angle de rotation et la translation
  #     sont doublés
  I.setValue(translation1Node,  I.getValue(translation1Node)*2)
  I.setValue(rotationAngle1Node,I.getValue(rotationAngle1Node)*2)
  I.setValue(firstJoinNode,zoneDupName)
  #>>>> le raccord opposé à la duplication (second) devient un match
  #     non péridique avec la zone dupliquée mais les PointList/PointListDonor
  #     sont conservés
  GCP2 = I.getNodeFromType1(secondJoinNode, "GridConnectivityProperty_t")
  I._rmNode(secondJoinNode,GCP2)
  I.setValue(secondJoinNode,zoneDupName)
  #> mise à jour des raccords dupliqués
  #  c'est l'inverse
  #>>>>
  GCPDup2 = I.getNodeFromType1(secondJoinDupNode, "GridConnectivityProperty_t")
  translationDup2Node   = I.getNodeFromName2(GCPDup2, "Translation")
  rotationAngleDup2Node = I.getNodeFromName2(GCPDup2, "RotationAngle")
  I.setValue(translationDup2Node,  I.getValue(translationDup2Node)*2)
  I.setValue(rotationAngleDup2Node,I.getValue(rotationAngleDup2Node)*2)
  I.setValue(secondJoinDupNode,zoneName)
  #>>>>
  GCPDup1 = I.getNodeFromType1(firstJoinDupNode, "GridConnectivityProperty_t")
  I._rmNode(firstJoinDupNode,GCPDup1)
  I.setValue(firstJoinDupNode,zoneName)
  
  # Ajout de la zone dupliquée dans la base
  I._addChild(base,zoneDup)
  
  if conformize:
    if comm is None:
      raise ValueError("MPI communicator is mandatory for conformization !")
    JN_for_duplication_paths = []
    JN_for_duplication_paths.append(I.getPath(dist_tree,secondJoinNode,pyCGNSLike=True)[1:])
    JN_for_duplication_paths.append(I.getPath(dist_tree,firstJoinDupNode,pyCGNSLike=True)[1:])
    CCJ.conformize_jn(dist_tree,JN_for_duplication_paths,comm)


def _duplicate_n_zones_from_periodic_join(dist_tree,zones,JN_for_duplication_paths,N,
                                          conformize=False,comm=None):
  #############
  ##### TODO
  ##### > gestion des autres raccords...
  ##### > autre chose ?
  #############

  if N<0:
    return

  # Récupération de la base
  pathZone0 = I.getPath(dist_tree,zones[0])
  pathBase  = "/".join(pathZone0.split("/")[:-1])
  base      = I.getNodeFromPath(dist_tree,pathBase)

  # Récupération du premier raccord de l'ensemble A
  firstJoinInMatchingA = I.getNodeFromPath(dist_tree,JN_for_duplication_paths[0][0])
  
  # Récupération des paramètres de transformation
  GCPA = I.getNodeFromType1(firstJoinInMatchingA, "GridConnectivityProperty_t")
  rotationCenterANode = I.getNodeFromName2(GCPA, "RotationCenter")
  rotationAngleANode  = I.getNodeFromName2(GCPA, "RotationAngle")
  translationANode    = I.getNodeFromName2(GCPA, "Translation")
  rotationCenterA     = I.getValue(rotationCenterANode)
  rotationAngleA      = I.getValue(rotationAngleANode)
  translationA        = I.getValue(translationANode)
  
  # Sauvegarde des informations de périodicité des raccords périodiques
  # "B" initiaux
  jn_b_properties = [None]*len(JN_for_duplication_paths[1])
  for jn,jn_path_b in enumerate(JN_for_duplication_paths[1]):
    jn_b_init_node = I.getNodeFromPath(dist_tree, jn_path_b)
    #print(N,"b_init",jn_path_b,I.getValue(jn_b_init_node))
    gcp_b_init = copy.deepcopy(I.getNodeFromType1(jn_b_init_node, "GridConnectivityProperty_t"))
    jn_b_properties[jn] = gcp_b_init

  # Changement de nom de la zone dupliquée
  zonesNamesPrefixes = [None]*len(zones)
  for z,zone in enumerate(zones):
    zoneNamePrefix = copy.deepcopy(I.getName(zone))
    zonesNamesPrefixes[z] = zoneNamePrefix
    I.setName(zone,zoneNamePrefix+".D0")
    #> mise à jour des raccords matchs des zones à dupliquer
    zgc  = I.getNodeFromType1(zone,"ZoneGridConnectivity_t")
    for gc in I.getNodesFromType1(zgc,"GridConnectivity_t") \
            + I.getNodesFromType1(zgc,"GridConnectivity1to1_t"):
      gcp = I.getNodeFromType1(gc,"GridConnectivityProperty_t")
      # if 'rotor_Hm' in zonesNamesPrefixes[z]:
      #   if 'match2_0' in gc[0]:
      #     print("0",I.getValue(gc))
      if gcp is None:
        I.setValue(gc,I.getValue(gc)+".D0")
        # if 'rotor_Hm' in zonesNamesPrefixes[z]:
        #   if 'match2_0' in gc[0]:
        #     print("0",I.getValue(gc))

  max_ordinal = 0
  for base in I.getBases(dist_tree):
      for zone in I.getZones(base):
        for zgc in I.getNodesFromType1(zone, 'ZoneGridConnectivity_t'):
          for gc in I.getNodesFromType1(zgc, 'GridConnectivity_t')+I.getNodesFromType1(zgc, 'GridConnectivity1to1_t'):
            ordinal_n = I.getNodeFromName(gc, 'Ordinal')
            if ordinal_n is not None:
              max_ordinal = max(max_ordinal,I.getValue(ordinal_n))

  # Duplication
  for n in range(N):
    for z,zone in enumerate(zones):
      zoneDupName =  zonesNamesPrefixes[z]+".D{0}".format(n+1)
      zoneDup = duplicate_zone_with_transformation(zone,zoneDupName,
                                              rotationCenter=rotationCenterA,
                                              rotationAngle=(n+1)*rotationAngleA,
                                              translation=(n+1)*translationA,
                                              max_ordinal = (n+1)*max_ordinal)
  
      # Mise à jour des raccords matchs des zones dupliquées
      zgc  = I.getNodeFromType1(zoneDup,"ZoneGridConnectivity_t")
      for gc in I.getNodesFromType1(zgc,"GridConnectivity_t") \
              + I.getNodesFromType1(zgc,"GridConnectivity1to1_t"):
        gcp = I.getNodeFromType1(gc,"GridConnectivityProperty_t")
        if gcp is None:
          # I.setValue(gc,zonesNamesPrefixes[z]+".D{0}".format(n+1))
          gc_value = I.getValue(gc)
          new_gc_value = ".".join(gc_value.split('.D')[:-1])+".D{0}".format(n+1)
          I.setValue(gc,new_gc_value)
          # if 'rotor_Hm' in zonesNamesPrefixes[z]:
          #   if 'match2_0' in gc[0]:
          #     print(n,gc_value,new_gc_value,I.getValue(gc))
    
      # Ajout de la zone dupliquée dans la base
      I._addChild(base,zoneDup)

    #> Transformation des raccords périodiques "B" de l'ensemble de zones
    #  précédent en raccords match
    for jn_path_b in JN_for_duplication_paths[1]:
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
      # I.setValue(jn_b_prev_node,I.getValue(jn_b_prev_node)[:-3]+".D{0}".format(n+1))
      # if 'match2_0' in jn_b_prev_node[0]:
      #   print(n,"b_prev",jn_path_b,jn_path_b_prev,I.getValue(jn_b_prev_node))

    #> Transformation des raccords périodiques "A" de l'ensemble de zones
    #  courant en raccords match
    for jn_path_a in JN_for_duplication_paths[0]:
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
      # print(n,"a_curr",jn_path_a_curr,I.getValue(jn_a_curr_node))

    if conformize:
      # TO DO : adapt for multi zones !!!
      if comm is None:
        raise ValueError("MPI communicator is mandatory for conformization !")
      # JN_for_duplication_paths = []
      # JN_for_duplication_paths.append(I.getPath(dist_tree,secondJoinPrevNode,pyCGNSLike=True)[1:])
      # JN_for_duplication_paths.append(I.getPath(dist_tree,firstJoinDupNode,pyCGNSLike=True)[1:])
      # CCJ.conformize_jn(dist_tree,JN_for_duplication_paths,comm)
  
  #> Mise à jour des raccords périodiques "A" de l'ensemble de zones initial
  for jn_path_a in JN_for_duplication_paths[0]:
    split_jn_path_a = jn_path_a.split("/")
    jn_path_a_init = "/".join(split_jn_path_a[0:2]) + ".D0/" \
                   + "/".join(split_jn_path_a[2:])
    jn_a_init_node = I.getNodeFromPath(dist_tree, jn_path_a_init)
    gcp_a_init = I.getNodeFromType1(jn_a_init_node, "GridConnectivityProperty_t")
    rotation_angle_a_node = I.getNodeFromName2(gcp_a_init, "RotationAngle")
    I.setValue(rotation_angle_a_node, I.getValue(rotation_angle_a_node)*(N+1))
    translation_a_node = I.getNodeFromName2(gcp_a_init, "Translation")
    I.setValue(translation_a_node, I.getValue(translation_a_node)*(N+1))
    I.setValue(jn_a_init_node,I.getValue(jn_a_init_node)+".D{0}".format(N))
    #print(N,"a_init",jn_path_a_init,I.getValue(jn_a_init_node))
    ordinal_opp_n = I.getNodeFromName(gc, 'OrdinalOpp')
    if ordinal_opp_n is not None:
      I.setValue(ordinal_opp_n,I.getValue(ordinal_opp_n)+N*max_ordinal)

  #> Mise à jour des raccords périodiques "B" du dernier ensemble de zones dupliqué
  for jn,jn_path_b in enumerate(JN_for_duplication_paths[1]):
    split_jn_path_b = jn_path_b.split("/")
    #print(N,"b_last",split_jn_path_b)
    jn_path_b_last = "/".join(split_jn_path_b[0:2]) + ".D{0}/".format(N) \
                   + "/".join(split_jn_path_b[2:])
    jn_b_last_node = I.getNodeFromPath(dist_tree, jn_path_b_last)
    I._addChild(jn_b_last_node,jn_b_properties[jn])
    gcp_b_last = I.getNodeFromType1(jn_b_last_node, "GridConnectivityProperty_t")
    rotation_angle_b_node = I.getNodeFromName2(gcp_b_last, "RotationAngle")
    I.setValue(rotation_angle_b_node, I.getValue(rotation_angle_b_node)*(N+1))
    translation_b_node = I.getNodeFromName2(gcp_b_last, "Translation")
    I.setValue(translation_b_node, I.getValue(translation_b_node)*(N+1))
    gc_value = I.getValue(jn_b_last_node)
    if len(gc_value.split('.D'))>1:
      new_gc_value = ".".join(gc_value.split('.D')[:-1])+".D0"
    else:
      new_gc_value = gc_value.split('.D')[0]+".D0"
    I.setValue(jn_b_last_node,new_gc_value)
    # I.setValue(jn_b_last_node,I.getValue(jn_b_last_node)+".D0")
    ordinal_opp_n = I.getNodeFromName(gc, 'OrdinalOpp')
    if ordinal_opp_n is not None:
      I.setValue(ordinal_opp_n,I.getValue(ordinal_opp_n)-N*max_ordinal)
  
  
def _duplicate_zones_from_periodic_join_by_rotation_to_360(dist_tree,zones,JN_for_duplication_paths,
                                                           conformize=False,comm=None,
                                                           rotation_correction=True):
  
  #############
  ##### TODO
  ##### > astuce pour que l'angle de la rotation soit le plus proche possible de la bonne valeur
  #####     a. définir N le nombre de secteurs
  #####     b. recalculer l'angle de rotation exact
  #####     c. se servir de ce nouvel angle
  #####     => ceci doit permettre de répartir les erreurs d'arrondi sur tous les secteurs
  ##### > mettre un warning si translation != [0.,0.,0.] => DONE
  ##### > definir N puis appliquer '_duplicateNZonesFromJoin' => DONE
  ##### > transformer le raccord péridique entre 0 et N en match => DONE
  ##### > corriger les coordonnées des noeuds de la dernière zone pour assurer le match !
  #############
  
  # Récupération de la base
  pathZone0 = I.getPath(dist_tree,zones[0])
  pathBase  = "/".join(pathZone0.split("/")[:-1])
  base      = I.getNodeFromPath(dist_tree,pathBase)

  # Récupération du premier raccord de l'ensemble A
  firstJoinInMatchingA = I.getNodeFromPath(dist_tree,JN_for_duplication_paths[0][0])
  
  # Récupération des paramètres de transformation
  GCPA = I.getNodeFromType1(firstJoinInMatchingA, "GridConnectivityProperty_t")
  rotationCenterANode = I.getNodeFromName2(GCPA, "RotationCenter")
  rotationAngleANode  = I.getNodeFromName2(GCPA, "RotationAngle")
  translationANode    = I.getNodeFromName2(GCPA, "Translation")
  rotationCenterA     = I.getValue(rotationCenterANode)
  rotationAngleA      = I.getValue(rotationAngleANode)
  translationA        = I.getValue(translationANode)
  
  if (translationA != np.array([0.,0.,0.])).any():
    raise ValueError("The join is not periodic only by rotation !")
  
  # Find the number of duplication needed
  index = np.where(rotationAngleA != 0)[0]
  if index.size == 1:
    N = abs(int(np.round(2*np.pi/rotationAngleA[index])))
    if rotation_correction:
      rotationAngleA[index] = np.sign(rotationAngleA[index])*2*np.pi/N
  else:
    # TO DO : vérifier le type de l'erreur
    raise ValueError("Zone/Join not define a section of a row")
  
  # Duplications
  _duplicate_n_zones_from_periodic_join(dist_tree,zones,JN_for_duplication_paths,N-1,
                                        conformize=conformize,comm=comm)

  #> Transformation des raccords périodiques "A" de l'ensemble de zones initial
  #  en raccords match
  for jn_path_a in JN_for_duplication_paths[0]:
    split_jn_path_a = jn_path_a.split("/")
    jn_path_a_init = "/".join(split_jn_path_a[0:2]) + ".D0/" \
                   + "/".join(split_jn_path_a[2:])
    jn_a_init_node = I.getNodeFromPath(dist_tree, jn_path_a_init)
    gcp_a_init = I.getNodeFromType1(jn_a_init_node, "GridConnectivityProperty_t")
    I._rmNode(jn_a_init_node,gcp_a_init)

  #> Transformation des raccords périodiques "B" du dernier ensemble de zones dupliqué
  #  en raccords match non nécessaire car raccords déjà match par cionstruction
  for jn_path_b in JN_for_duplication_paths[1]:
    split_jn_path_b = jn_path_b.split("/")
    jn_path_b_last = "/".join(split_jn_path_b[0:2]) + ".D{0}/".format(N-1) \
                   + "/".join(split_jn_path_b[2:])
    jn_b_last_node = I.getNodeFromPath(dist_tree, jn_path_b_last)
    gcp_b_last = I.getNodeFromType1(jn_b_last_node, "GridConnectivityProperty_t")
    I._rmNode(jn_b_last_node,gcp_b_last)

  
  
  if conformize:
	# TO DO : adapt for multi zones !!!
	# Je pense que le conformize n'est pas utile ici car fait dans "_duplicate_n_zones_from_periodic_join()"
    if comm is None:
      raise ValueError("MPI communicator is mandatory for conformization !")
    #JN_for_duplication_paths = []
    #JN_for_duplication_paths.append(I.getPath(dist_tree,firstJoinNode,pyCGNSLike=True)[1:])
    #JN_for_duplication_paths.append(I.getPath(dist_tree,finalSecondJoinNode,pyCGNSLike=True)[1:])
    #CCJ.conformize_jn(dist_tree,JN_for_duplication_paths,comm)
  
