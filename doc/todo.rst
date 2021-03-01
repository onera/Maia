.. _todo:

TODO
====

- spack
- doc : génération sphinx + pages


Fonctionnalités :

# Déjà fait
   hdf_filter
   lecture / ecriture : DistTree ET PartTree  (hdf_filter + save)
   Load balancing
   partitionnement : pypart  (Service complet / avec sans ghost cell / )

   Service de transfert DistTree / PartTree

   fetch topologique // 3D/2D/1D + maia  + [ distribué / partitioné ]
   dconnectivity_transform / pcoonectivty_trsfrom
   Déstructuration // ( convertArray2NGon )
   Transfert distribué cell <-> noeuds (underconstruction)
   cgns_registry
   Génération de maillage // --> Cube / Plan
   connectMatch / connectMatchPeriodic + [ distribué / partitioné ]


# À faire
   Partitionnement avec ghost cell
   Transfert distribué cell <-> noeuds (underconstruction)
   Dist2Wall : ok
   ExtractMesh : (Interpolation d'un maillage à l'autre)
   Calcul des Gradients : Une description
   Overlay (intersection surfacique de maillage) : Une description
   MeshLocation : Location d'un nuage de point dans un maillage
   Cloud
   Face->Vertex (Adjoint)




