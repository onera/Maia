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
  # TODO : à remplacer par la fonction codée par JulienCoulet
  coordsDupNode = I.getNodeFromType1(zoneDup, "GridCoordinates_t")
  coordXDupNode = I.getNodeFromName1(coordsDupNode, "CoordinateX")
  coordYDupNode = I.getNodeFromName1(coordsDupNode, "CoordinateY")
  coordZDupNode = I.getNodeFromName1(coordsDupNode, "CoordinateZ")
  coordXDup     = I.getValue(coordXDupNode) + translation[0]
  coordYDup     = I.getValue(coordYDupNode) + translation[1]
  coordZDup     = I.getValue(coordZDupNode) + translation[2]
  I.setValue(coordXDupNode,coordXDup)
  I.setValue(coordYDupNode,coordYDup)
  I.setValue(coordZDupNode,coordZDup)
  
  return zoneDup


def _duplicateZoneFromJoin(tree,zone,JN_for_duplication_Name):
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
  zoneName = I.getName(zone)
  I.setName(zone,zoneName+".D0")

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
  zoneDupName = zoneName+".D1"
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
  #>>>> le raccord sur lequel s'appuie la duplication (first) devient un match
  #     non péridique avec la zone dupliquée mais les PointList/PointListDonor
  #     sont conservés
  I._rmNode(firstJoinNode,GCP1)
  I.setValue(firstJoinNode,zoneDupName)
  #>>>> le raccord opposé à la duplication (second) reste un match péridique
  #     avec la zone dupliquée mais l'angle de rotation et la translation
  #     sont doublés
  GCP2 = I.getNodeFromType1(secondJoinNode, "GridConnectivityProperty_t")
  translation2Node   = I.getNodeFromName(GCP2, "Translation")
  rotationAngle2Node = I.getNodeFromName(GCP2, "RotationAngle")
  I.setValue(translation2Node,  I.getValue(translation2Node)*2)
  I.setValue(rotationAngle2Node,I.getValue(rotationAngle2Node)*2)
  I.setValue(secondJoinNode,zoneDupName)
  #> mise à jour des raccords dupliqués
  #  c'est l'inverse
  #>>>>
  GCPDup2 = I.getNodeFromType1(secondJoinDupNode, "GridConnectivityProperty_t")
  I._rmNode(secondJoinDupNode,GCPDup2)
  #>>>>
  GCPDup1 = I.getNodeFromType1(firstJoinDupNode, "GridConnectivityProperty_t")
  translationDup1Node   = I.getNodeFromName(GCPDup1, "Translation")
  rotationAngleDup1Node = I.getNodeFromName(GCPDup1, "RotationAngle")
  I.setValue(translationDup1Node,  I.getValue(translationDup1Node)*2)
  I.setValue(rotationAngleDup1Node,I.getValue(rotationAngleDup1Node)*2)
  
  # Ajout de la zone dupliquée dans la base
  I._addChild(base,zoneDup)


def _duplicateNZonesFromJoin(tree,zone,JN_for_duplication_Name,N):
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
  zoneName = I.getName(zone)
  I.setName(zone,zoneName+".D0")

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
  
  #TO DO
  # Est-ce que en jouant sur les références on peut juste récupérer les raccords
  # et le reste se met à jour ?
  
  # Duplication
  for n in range(N):
    zoneDupName = zoneName+".D{0}".format(n+1)
    zoneDup = duplicateZoneWithTransformation(zone,zoneDupName,
                                              rotationCenter=rotationCenter1,
                                              rotationAngle=(n+1)*rotationAngle1,
                                              translation=(n+1)*translation1)
  
  
  
    # Mise à jour des raccords
    #> recuperation des raccords de la zone précédente
    zonePrevName       = zoneName+".D{0}".format(n)
    zonePrev           = I.getNodeFromName1(base,zonePrevName)
    ZGCPrev            = I.getNodeFromType1(zonePrev,"ZoneGridConnectivity_t")
    firstJoinPrevNode  = I.getNodeFromName1(ZGCPrev,JN_for_duplication_Name[0])
    #> recuperation des raccords dupliqués
    ZGCDup            = I.getNodeFromType1(zoneDup,"ZoneGridConnectivity_t")
    secondJoinDupNode = I.getNodeFromName1(ZGCDup,JN_for_duplication_Name[1])
    #> récupération des 'GridConnectivityProperty_t'
    GCPPrev1 = I.getNodeFromType1(firstJoinPrevNode, "GridConnectivityProperty_t")
    GCPDup2  = I.getNodeFromType1(secondJoinDupNode, "GridConnectivityProperty_t")
    #> mise à jour des raccords entre prev et dup :
    #   le raccord sur lequel s'appuie la duplication (first) devient un match
    #   non péridique avec la zone dupliquée mais les PointList/PointListDonor
    #   sont conservés
    #>>>>
    I._rmNode(firstJoinPrevNode,GCPPrev1)
    I.setValue(firstJoinPrevNode,zoneDupName)
    #>>>>
    I._rmNode(secondJoinDupNode,GCPDup2)
    I.setValue(secondJoinDupNode,zonePrevName)
    
    # Ajout de la zone dupliquée dans la base
    I._addChild(base,zoneDup)
    
    
  # Mise a jour du raccord périodique :
  #   le raccord opposé à la duplication (second) reste un match péridique
  #   avec la zone dupliquée mais l'angle de rotation et la translation
  #   sont multiploés par N+1
  #>>>>
  GCP2 = I.getNodeFromType1(secondJoinNode, "GridConnectivityProperty_t")
  translation2Node   = I.getNodeFromName(GCP2, "Translation")
  rotationAngle2Node = I.getNodeFromName(GCP2, "RotationAngle")
  I.setValue(translation2Node,  I.getValue(translation2Node)*(N+1))
  I.setValue(rotationAngle2Node,I.getValue(rotationAngle2Node)*(N+1))
  I.setValue(secondJoinNode,zoneDupName)
  #>>>>
  I.printTree(ZGC)
  firstJoinDupNode  = I.getNodeFromName1(ZGCDup,JN_for_duplication_Name[0])
  GCPDup1 = I.getNodeFromType1(firstJoinDupNode, "GridConnectivityProperty_t")
  translationDup1Node   = I.getNodeFromName(GCPDup1, "Translation")
  rotationAngleDup1Node = I.getNodeFromName(GCPDup1, "RotationAngle")
  I.setValue(translationDup1Node,  I.getValue(translationDup1Node)*(N+1))
  I.setValue(rotationAngleDup1Node,I.getValue(rotationAngleDup1Node)*(N+1))
  I.setValue(secondJoinNode,zoneName+".D0")
  
  
def _duplicateZonesFromJoinTo360(tree,zone,JN_for_duplication_Name):
  
  #############
  ##### TODO
  ##### > definir N puis appliquer '_duplicateNZonesFromJoin'
  ##### > transformer le raccord péridique entre 0 et N en match
  ##### > corriger les coordonnées des noeuds de la dernière zone pour assurer le match !
  #############
  
  # Récupération des raccords
  ZGC            = I.getNodeFromType1(zone,"ZoneGridConnectivity_t")
  firstJoinNode  = I.getNodeFromName1(ZGC,JN_for_duplication_Name[0])
  
  # Récupération des paramètres de transformation
  GCP1 = I.getNodeFromType1(firstJoinNode, "GridConnectivityProperty_t")
  rotationAngle1Node  = I.getNodeFromName(GCP1, "RotationAngle")
  rotationAngle1      = I.getValue(rotationAngle1Node)
  rotationAngle1[1] = 1.
  
  # Find the number of duplication needed
  index = np.where(rotationAngle1 != 0)[0]
  if index.size == 1:
    N = np.round(2*np.pi/rotationAngle1[index])
    N = 3
    rotationAngle1[1] = 1.
    print(N)
  else:
    # TO DO : vérifier le type de l'erreur
    raise ValueError("Zone/Join not define a section of a row")
   
  if N > 1: # Sinon c'est que l'on a déjà la roue entière
    _duplicateNZonesFromJoin(tree,zone,JN_for_duplication_Name,N-1)
  
  pass
