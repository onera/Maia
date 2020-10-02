#  - But :
#
#  1) Composant pyCGNS : ensemble cohérent de service en // en composant des algorithmes provenant de ParaDiGM essentiellement.
#
#
#  - Algo
#
#   --> hdf_filter
#
#   --> lecture / ecriture : DistTree ET PartTree  (hdf_filter + save)
#   --> Load balancing
#
#   --> partitionnement : pypart  (Service complet / avec sans ghost cell / )
#
#   --> Service de transfert DistTree / PartTree
#
#   --> fetch topologique // 3D/2D/1D + maia  + [ distribué / partitioné ]
#   --> dconnectivity_transform / pcoonectivty_trsfrom
#   --> Déstructuration // ( convertArray2NGon )
#   --> Transfert distribué cell <-> noeuds (underconstruction)
#
#
#   --> Génération de maillage // --> Cube / Plan
#
#   --> cgns_registery
#
#
#   --> Dist2Wall : ok
#   --> connectMatch / connectMatchPeriodic + [ distribué / partitioné ]
#
#   --> ExtractMesh :(MeshLocation + Interpolation + distant_neightbor )
#   --> Calcul des Gradients : Une description
#
#
#   --> Overlay (intersection surfacique de maillage) : Une description
#   --> MeshLocation : Location d'un nuage de point dans un maillage
#
#
#   --> Cloud
#
#
#  - Doc
#
#  - Test (gros boulot)
#
#   --> Dev de pytest + pytest // (généralisation de doctest MPI )
#
#   --> Doc sphinx - OK





