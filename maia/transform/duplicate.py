import Converter.Internal as I
import numpy as np
import copy


def duplicateZoneWithTransformation(zone,nameZoneDup,
                                    rotationCenter=np.array([0.,0.,0.]),
                                    rotationAngle=np.array([0.,0.,0.]),
                                    translation=np.array([0.,0.,0.])):
  # Duplication de la zone
  zoneDup     = copy.deepcopy(zone)
  I.setName(zoneDup,nameZoneDup)
  
  # Apply transformation
  coordsDupNode  = I.getNodeFromType1(zoneDup, "GridCoordinates_t")
  coordXDupNode  = I.getNodeFromName1(coordsDupNode, "CoordinateX")
  coordYDupNode  = I.getNodeFromName1(coordsDupNode, "CoordinateY")
  coordZDupNode  = I.getNodeFromName1(coordsDupNode, "CoordinateZ")
  
  modCx, modCy, modCz = applyTransformation(rotationCenter, rotationAngle, translation, 
                                            I.getValue(coordXDupNode),
                                            I.getValue(coordYDupNode),
                                            I.getValue(coordZDupNode))

  I.setValue(coordXDupNode,modCx)
  I.setValue(coordYDupNode,modCy)
  I.setValue(coordZDupNode,modCz)
  
  return zoneDup


def _duplicateZoneFromPeriodicJoin(tree,zone,JN_for_duplication_Name):
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
  pathZone    = I.getPath(tree,zone)
  pathBase    = "/".join(pathZone.split("/")[:-1])
  base        = I.getNodeFromPath(tree,pathBase)
  
  # Changement de nom de la zone dupliquée
  zoneNamePrefix = I.getName(zone)
  zoneName = zoneNamePrefix+".D0"
  I.setName(zone,zoneName)

  # Récupération des raccords
  ZGC               = I.getNodeFromType1(zone,"ZoneGridConnectivity_t")
  firstJoinNode     = I.getNodeFromName1(ZGC,JN_for_duplication_Name[0])
  secondJoinNode    = I.getNodeFromName1(ZGC,JN_for_duplication_Name[1])
  
  # Duplication de la zone
  #> récupération des paramètres de transformation
  GCP1 = I.getNodeFromType1(firstJoinNode, "GridConnectivityProperty_t")
  rotationCenter1Node = I.getNodeFromName(GCP1, "RotationCenter")
  rotationAngle1Node  = I.getNodeFromName(GCP1, "RotationAngle")
  translation1Node    = I.getNodeFromName(GCP1, "Translation")
  rotationCenter1     = I.getValue(rotationCenter1Node)
  rotationAngle1      = I.getValue(rotationAngle1Node)
  translation1        = I.getValue(translation1Node)
  #> duplication
  zoneDupName = zoneNamePrefix+".D1"
  zoneDup = duplicateZoneWithTransformation(zone,zoneDupName,
                                            rotationCenter=rotationCenter1,
                                            rotationAngle=rotationAngle1,
                                            translation=translation1)
  
  # Mise à jour des raccords
  #> recuperation des raccords dupliqués
  ZGCDup            = I.getNodeFromType1(zoneDup,"ZoneGridConnectivity_t")
  firstJoinDupNode  = I.getNodeFromName1(ZGCDup,JN_for_duplication_Name[0])
  secondJoinDupNode = I.getNodeFromName1(ZGCDup,JN_for_duplication_Name[1])
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
  translationDup2Node   = I.getNodeFromName(GCPDup2, "Translation")
  rotationAngleDup2Node = I.getNodeFromName(GCPDup2, "RotationAngle")
  I.setValue(translationDup2Node,  I.getValue(translationDup2Node)*2)
  I.setValue(rotationAngleDup2Node,I.getValue(rotationAngleDup2Node)*2)
  I.setValue(secondJoinDupNode,zoneName)
  #>>>>
  GCPDup1 = I.getNodeFromType1(firstJoinDupNode, "GridConnectivityProperty_t")
  I._rmNode(firstJoinDupNode,GCPDup1)
  I.setValue(firstJoinDupNode,zoneName)
  
  # Ajout de la zone dupliquée dans la base
  I._addChild(base,zoneDup)


def _duplicateNZonesFromPeriodicJoin(tree,zone,JN_for_duplication_Name,N):
  #############
  ##### TODO
  ##### > gestion des autres raccords...
  ##### > autre chose ?
  #############

  # Récupération de la base
  pathZone    = I.getPath(tree,zone)
  pathBase    = "/".join(pathZone.split("/")[:-1])
  base        = I.getNodeFromPath(tree,pathBase)
  
  # Changement de nom de la zone dupliquée
  zoneNamePrefix = I.getName(zone)
  zoneName = zoneNamePrefix+".D0"
  I.setName(zone,zoneName)

  # Récupération des raccords
  ZGC            = I.getNodeFromType1(zone,"ZoneGridConnectivity_t")
  firstJoinNode  = I.getNodeFromName1(ZGC,JN_for_duplication_Name[0])
  secondJoinNode = I.getNodeFromName1(ZGC,JN_for_duplication_Name[1])
  
  # Récupération des paramètres de transformation
  GCP1 = I.getNodeFromType1(firstJoinNode, "GridConnectivityProperty_t")
  rotationCenter1Node = I.getNodeFromName(GCP1, "RotationCenter")
  rotationAngle1Node  = I.getNodeFromName(GCP1, "RotationAngle")
  translation1Node    = I.getNodeFromName(GCP1, "Translation")
  rotationCenter1     = I.getValue(rotationCenter1Node)
  rotationAngle1      = I.getValue(rotationAngle1Node)
  translation1        = I.getValue(translation1Node)
  
  # Duplication
  for n in range(N):
    zoneDupName = zoneNamePrefix+".D{0}".format(n+1)
    zoneDup = duplicateZoneWithTransformation(zone,zoneDupName,
                                              rotationCenter=rotationCenter1,
                                              rotationAngle=(n+1)*rotationAngle1,
                                              translation=(n+1)*translation1)
  
  
  
    # Mise à jour des raccords
    #> recuperation des raccords de la zone précédente
    zonePrevName       = zoneNamePrefix+".D{0}".format(n)
    zonePrev           = I.getNodeFromName1(base,zonePrevName)
    ZGCPrev            = I.getNodeFromType1(zonePrev,"ZoneGridConnectivity_t")
    secondJoinPrevNode = I.getNodeFromName1(ZGCPrev,JN_for_duplication_Name[1])
    #> recuperation des raccords dupliqués
    ZGCDup           = I.getNodeFromType1(zoneDup,"ZoneGridConnectivity_t")
    firstJoinDupNode = I.getNodeFromName1(ZGCDup,JN_for_duplication_Name[0])
    #> récupération des 'GridConnectivityProperty_t'
    GCPPrev2 = I.getNodeFromType1(secondJoinPrevNode, "GridConnectivityProperty_t")
    GCPDup1  = I.getNodeFromType1(firstJoinDupNode, "GridConnectivityProperty_t")
    #> mise à jour du nouveau raccord entre prev et dup :
    #   le raccord opposé à la duplication (second) devient un match
    #   non péridique avec la zone dupliquée mais les PointList/PointListDonor
    #   sont conservés
    #>>>>
    I._rmNode(secondJoinPrevNode,GCPPrev2)
    I.setValue(secondJoinPrevNode,zoneDupName)
    #>>>>
    I._rmNode(firstJoinDupNode,GCPDup1)
    I.setValue(firstJoinDupNode,zonePrevName)
    
    # Ajout de la zone dupliquée dans la base
    I._addChild(base,zoneDup)
    
    
  # Mise a jour du raccord périodique :
  #   le raccord sur lequel s'appuie la duplication (first) reste un match péridique
  #   avec la dernière zone dupliquée mais l'angle de rotation et la translation
  #   sont multipliés par N+1
  #>>>>
  I.setValue(translation1Node,  I.getValue(translation1Node)*(N+1))
  I.setValue(rotationAngle1Node,I.getValue(rotationAngle1Node)*(N+1))
  I.setValue(firstJoinNode,zoneDupName)
  #>>>>
  secondJoinDupNode = I.getNodeFromName1(ZGCDup,JN_for_duplication_Name[1])
  GCPDup2 = copy.deepcopy(GCP1)
  translationDup2Node   = I.getNodeFromName(GCPDup2, "Translation")
  rotationAngleDup2Node = I.getNodeFromName(GCPDup2, "RotationAngle")
  I.setValue(translationDup2Node,  I.getValue(translationDup2Node)*(-1))
  I.setValue(rotationAngleDup2Node,I.getValue(rotationAngleDup2Node)*(-1))
  I._addChild(secondJoinDupNode,GCPDup2)
  I.setValue(secondJoinDupNode,zoneNamePrefix+".D0")
  
  
def _duplicateZonesFromPeriodicJoinByRotationTo360(tree,zone,JN_for_duplication_Name):
  
  #############
  ##### TODO
  ##### > mettre un warning si translation != [0.,0.,0.]
  ##### > definir N puis appliquer '_duplicateNZonesFromJoin'
  ##### > transformer le raccord péridique entre 0 et N en match
  ##### > corriger les coordonnées des noeuds de la dernière zone pour assurer le match !
  #############
  
  # Informations générales
  pathZone       = I.getPath(tree,zone)
  pathBase       = "/".join(pathZone.split("/")[:-1])
  base           = I.getNodeFromPath(tree,pathBase)
  zoneNamePrefix = I.getName(zone)
  
  # Récupération des raccords
  ZGC            = I.getNodeFromType1(zone,"ZoneGridConnectivity_t")
  firstJoinNode  = I.getNodeFromName1(ZGC,JN_for_duplication_Name[0])
  
  # Récupération des paramètres de transformation
  GCP1 = I.getNodeFromType1(firstJoinNode, "GridConnectivityProperty_t")
  rotationAngle1Node  = I.getNodeFromName(GCP1, "RotationAngle")
  rotationAngle1      = I.getValue(rotationAngle1Node)
  rotationAngle1[1] = np.pi
  
  # Find the number of duplication needed
  index = np.where(rotationAngle1 != 0)[0]
  if index.size == 1:
    N = int(np.round(2*np.pi/rotationAngle1[index]))
    # N = 3
    # rotationAngle1[1] = 1.
    print(N)
  else:
    # TO DO : vérifier le type de l'erreur
    raise ValueError("Zone/Join not define a section of a row")
  
  # Duplication
  if N > 1: # Sinon c'est que l'on a déjà la roue entière
    _duplicateNZonesFromPeriodicJoin(tree,zone,JN_for_duplication_Name,N-1)
  
  # Transform periodic match join between zone.D0 and zone.D(N-1) to match join
  #> on zone.D0
  I._rmNode(firstJoinNode,GCP1)
  #> on zone.D(N-1)
  finalZoneName       = zoneNamePrefix+".D{0}".format(N-1)
  finalZone           = I.getNodeFromName1(base,finalZoneName)
  finalZGC            = I.getNodeFromType1(finalZone,"ZoneGridConnectivity_t")
  finalSecondJoinNode = I.getNodeFromName1(finalZGC,JN_for_duplication_Name[1])
  # finalGCP2           = I.getNodeFromType1(finalSecondJoinNode, "GridConnectivityProperty_t")
  # I._rmNode(finalSecondJoinNode,finalGCP2)
  I._rmNodesByType(finalSecondJoinNode,"GridConnectivityProperty_t")
  
