def _duplicateZone(tree,zone,JN_for_duplication_Name):
  #############
  ##### TODO
  ##### > gestion des autres raccords...
  ##### > autre chose ?
  #############
  # Duplication de la zone
  zoneDup     = copy.deepcopy(zone)
  nameZoneDup = I.getName(zone)+"Duplicated"
  I.setName(zoneDup,nameZoneDup)
  pathZone    = I.getPath(tree,zone)
  pathBase    = "/".join(pathZone.split("/")[:-1])
  base        = I.getNodeFromPath(tree,pathBase)
  I._addChild(base,zoneDup)
  zoneDup     = I.getNodeFromNameAndType(base,nameZoneDup,"Zone_t")
  
  # Récupération des raccords
  ZGC               = I.getNodeFromType1(zone,"ZoneGridConnectivity_t")
  firstJoinNode     = I.getNodeFromName1(ZGC,JN_for_duplication_Name[0])
  secondJoinNode    = I.getNodeFromName1(ZGC,JN_for_duplication_Name[1])
  ZGCDup            = I.getNodeFromType1(zoneDup,"ZoneGridConnectivity_t")
  firstJoinDupNode  = I.getNodeFromName1(ZGCDup,JN_for_duplication_Name[0])
  secondJoinDupNode = I.getNodeFromName1(ZGCDup,JN_for_duplication_Name[1])
  
  # Apply transformation
  # TODO : à remplacer par la fonction codée par JulienCoulet
  #> recupération du noeud 'GridConnectivityProperty_t'
  GCP1 = I.getNodeFromType1(firstJoinNode, "GridConnectivityProperty_t")
  translation1Node   = I.getNodeFromName(GCP1, "Translation")
  rotationAngle1Node = I.getNodeFromName(GCP1, "RotationAngle")
  translation1       = I.getValue(translation1Node)
  #> mise à jour des coordonnées dupliquées
  coordsDupNode = I.getNodeFromType1(zoneDup, "GridCoordinates_t")
  coordXDupNode = I.getNodeFromName1(coordsDupNode, "CoordinateX")
  coordYDupNode = I.getNodeFromName1(coordsDupNode, "CoordinateY")
  coordZDupNode = I.getNodeFromName1(coordsDupNode, "CoordinateZ")
  coordXDup     = I.getValue(coordXDupNode) + translation1[0]
  coordYDup     = I.getValue(coordYDupNode) + translation1[1]
  coordZDup     = I.getValue(coordZDupNode) + translation1[2]
  I.setValue(coordXDupNode,coordXDup)
  I.setValue(coordYDupNode,coordYDup)
  I.setValue(coordZDupNode,coordZDup)
  
  # Mise à jour des raccords
  #> mise à jour des raccords initiaux
  #>>>> le raccord sur lequel s'appuie la duplication (first) devient un match
  #     non péridique avec la zone dupliquée mais les PointList/PointListDonor
  #     sont conservés
  I._rmNode(firstJoinNode,GCP1)
  I.setValue(firstJoinNode,nameZoneDup)
  #>>>> le raccord opposé à la duplication (second) reste un match péridique
  #     avec la zone dupliquée mais l'angle de rotation et la translation
  #     sont doublés
  GCP2 = I.getNodeFromType1(secondJoinNode, "GridConnectivityProperty_t")
  translation2Node   = I.getNodeFromName(GCP2, "Translation")
  rotationAngle2Node = I.getNodeFromName(GCP2, "RotationAngle")
  I.setValue(translation2Node,  I.getValue(translation2Node)*2)
  I.setValue(rotationAngle2Node,I.getValue(rotationAngle2Node)*2)
  I.setValue(secondJoinNode,nameZoneDup)
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
