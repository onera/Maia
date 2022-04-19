Conversion Structuré - Non structuré distribuée
===============================================

Cette note reprend le descriptif du developpement `merge_request_3
<http://gitlab-elsa-test.onecert.fr/clef/maia/merge_requests/3>`_
en attendant une meilleure intégration dans la documentation.

Dev permettant de convertir un arbre distribué structuré en arbre distribué non structuré.

Étant donnée que le nouveau chargement du dist_tree S produit des tableaux "à plat" (grâce aux slabs),
il n'y a rien à transformer pour les données (vertex, flowsolution, bcdata, etc.).
Il faut en revanche générer, de manière consistante avec les slabs de données,  

- la connectivité NGON; 
- les pointLists à partir des pointRanges.

Les deux points s'appuient sur la création d'une numérotation non structuré du maillage cartésien en i,j,k croissant,
et on retrouve dans le code deux briques correspondant à ces deux fonctionnalités : `compute_all_ngon_connectivity`
et `compute_pointList_from_pointRanges`. La première est utilisée pour les NGon tandis que la seconde est utilisée
pour les BC et GC (dans ce dernier cas elle permet d'obtenir le PL et PLDonor, l'idée étant que l'un des
deux joins impose son découpage en slab à l'autre pour conserver le match).

Quelques notes:

- La construction des ngons pourrait être isolée de manière à générer directement un disttree U à
  partir de la taille de la zone, à la manière de cartNGon().
- Les zones subregion ne sont pas encore traitées, mais à priori :

  + s'il n'y a que de la donnée (et pas de pointRange spécifique, *ie* la SZR est liée à une BC/GC) :
    il faut juste copier la data
  + s'il y a un PR spécifique, on devrait pouvoir appeler `compute_pointList_from_pointRanges`.
    Néanmoins cette function n'a pas été testée dans le cas d'un PointRange représentant un volume.

- Même chose pour les BCDataSet, mais cette fois sans la réserve sur le PR volumique. 
- L'utilisateur à la possibilité de choisir la gridLocation attendue en sortie pour les BC/GC.
  Mais je m’aperçois à l'instant qu'il faudrait empêcher cela si un BCDataSet est présent, puisque
  on ne serait dans ce cas plus consistant avec la donnée. 
- Effets de bords: 

  + modification de `point_range_size` qui permet de calculer la taille d'un PR pour gérer les PR inversé :
    en effet il est permis d'avoir un PR du style [[1,3], [5,5], [**4,1**]], ce qui arrive sur les joins.
  + correction dans `cgns_io/correct_tree.py` : les GridConnectivityProperty n'étaient pas chargés sur les
    raccords *1to1*.
  + ajout d'un correction sur les PR dans `cgns_io/correct_tree.py` : on s'est rendu compte que cassiopée
    refuse le cas précédent d'un PR inversé, et les remet donc dans l'ordre croissant, ce qui casse le lien
    avec le *transform*. On remet donc tout d'équerre lors de la lecture de l'arbre si un incohérence avec le
    *transform* est détectée. 

Reste à faire :

- il faudrait ajouter, en plus des TU, un test "fonctionnel" qui illustrerait l'usage de la fonction top-level.
  A priori on attend d'avoir plus de recul sur l'organisation de ces tests pour mettre ça en place.
- BCDS et ZoneSubRegion (urgent ?)

La perfo à l'air assez bonne, j'ai mesuré sur un cube S 193x193x193 avec 6 BCs : 18 secondes en sequentiel
et environ 4 secondes sur 4 coeurs pour la conversion !


