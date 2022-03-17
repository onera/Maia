import pytest
from pytest_mpi_check._decorator import mark_mpi_test

import numpy as np
import os

from maia.utils       import parse_yaml_cgns
from maia.utils       import parse_cgns_yaml
from maia.transform   import duplicate
from maia.sids.pytree import compare
from maia.generate    import dcube_generator  as DCG
from maia.cgns_io     import cgns_io_tree     as IOT

import Converter.Internal as I


###############################################################################
class Test_find_cartesian_vector_names_from_names():
  names = ["Tata","TotoY","TotoZ","Titi","totoX"]
  # --------------------------------------------------------------------------- #
  def test_find_cartesian_vector_names_from_names1(self):
    assert(duplicate._find_cartesian_vector_names_from_names(self.names) == [])
  # --------------------------------------------------------------------------- #
  def test_find_cartesian_vector_names_from_names2(self):
    self.names.append("TotoX")
    assert(duplicate._find_cartesian_vector_names_from_names(self.names) == ["Toto"])
###############################################################################

###############################################################################
def test_duplicate_zone_with_transformation1():
  yz = """
       Zone Zone_t I4 [[18,4,0]]:
         ZoneType ZoneType_t "Unstructured":
         GridCoordinates GridCoordinates_t:
           CoordinateX DataArray_t:
             R4 : [ 0,1,2,
                    0,1,2,
                    0,1,2,
                    0,1,2,
                    0,1,2,
                    0,1,2 ]
           CoordinateY DataArray_t:
             R4 : [ 0,0,0,
                    1,1,1,
                    2,2,2,
                    0,0,0,
                    1,1,1,
                    2,2,2 ]
           CoordinateZ DataArray_t:
             R4 : [ 0,0,0,
                    0,0,0,
                    0,0,0,
                    1,1,1,
                    1,1,1,
                    1,1,1 ]
       """
  expected_yz = """
                DuplicatedZone Zone_t I4 [[18,4,0]]:
                  ZoneType ZoneType_t "Unstructured":
                  GridCoordinates GridCoordinates_t:
                    CoordinateX DataArray_t:
                      R4 : [ 0,1,2,
                             0,1,2,
                             0,1,2,
                             0,1,2,
                             0,1,2,
                             0,1,2 ]
                    CoordinateY DataArray_t:
                      R4 : [ 0,0,0,
                             1,1,1,
                             2,2,2,
                             0,0,0,
                             1,1,1,
                             2,2,2 ]
                    CoordinateZ DataArray_t:
                      R4 : [ 0,0,0,
                             0,0,0,
                             0,0,0,
                             1,1,1,
                             1,1,1,
                             1,1,1 ]
                """
  zone            = parse_yaml_cgns.to_node(yz)
  expected_zone   = parse_yaml_cgns.to_node(expected_yz)
  duplicated_zone = duplicate.duplicate_zone_with_transformation(zone,"DuplicatedZone")
  assert(duplicated_zone[0]=="DuplicatedZone")
  assert(compare.is_same_tree(duplicated_zone, expected_zone))
###############################################################################

###############################################################################
def test_duplicate_zone_with_transformation2():
  yaml_dir  = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'share', 'meshes')
  yaml_path = os.path.join(yaml_dir, 'quarter_crown_square_8.yaml')
  yaml_file = open(yaml_path,'r')
  tree = parse_yaml_cgns.to_cgns_tree(yaml_file.read())
  zone = I.getZones(tree)[0]
  rotationCenter  = np.array([0.,0.,0.])
  rotationAngle   = np.array([0.,0.,np.pi])
  translation     = np.array([0.,0.,0.])
  max_ordinal     = 4
  duplicated_zone = duplicate.duplicate_zone_with_transformation(zone,"DuplicatedZone",
                                                                 rotationCenter  = rotationCenter,
                                                                 rotationAngle   = rotationAngle,
                                                                 translation     = translation,
                                                                 max_ordinal     = max_ordinal,
                                                                 apply_to_fields = True)
  
  assert(duplicated_zone[0]=="DuplicatedZone")
  
  coord_x            = I.getVal(I.getNodeFromName(zone,           "CoordinateX"))
  coord_y            = I.getVal(I.getNodeFromName(zone,           "CoordinateY"))
  coord_z            = I.getVal(I.getNodeFromName(zone,           "CoordinateZ"))
  duplicated_coord_x = I.getVal(I.getNodeFromName(duplicated_zone,"CoordinateX"))
  duplicated_coord_y = I.getVal(I.getNodeFromName(duplicated_zone,"CoordinateY"))
  duplicated_coord_z = I.getVal(I.getNodeFromName(duplicated_zone,"CoordinateZ"))
  assert(np.allclose(coord_x,-duplicated_coord_x))
  assert(np.allclose(coord_y,-duplicated_coord_y))
  assert(np.allclose(coord_z, duplicated_coord_z))
  
  zgc            = I.getNodeFromType1(zone,           'ZoneGridConnectivity_t')
  duplicated_zgc = I.getNodeFromType1(duplicated_zone,'ZoneGridConnectivity_t')
  gcs = I.getNodesFromType1(zgc, 'GridConnectivity_t')
  for gc in gcs:
    duplicated_gc = I.getNodeFromNameAndType(duplicated_zgc,I.getName(gc),'GridConnectivity_t')
    ordinal_n                = I.getNodeFromName(gc,           'Ordinal')
    ordinal_opp_n            = I.getNodeFromName(gc,           'OrdinalOpp')
    duplicated_ordinal_n     = I.getNodeFromName(duplicated_gc,'Ordinal')
    duplicated_ordinal_opp_n = I.getNodeFromName(duplicated_gc,'OrdinalOpp')
    assert(I.getValue(ordinal_n)+4     == I.getValue(duplicated_ordinal_n))
    assert(I.getValue(ordinal_opp_n)+4 == I.getValue(duplicated_ordinal_opp_n))
    
  flow_solution            = I.getNodeFromType1(zone,           'FlowSolution_t')
  duplicated_flow_solution = I.getNodeFromType1(duplicated_zone,'FlowSolution_t')
  data_x             = I.getVal(I.getNodeFromName(flow_solution,           "DataX"))
  data_y             = I.getVal(I.getNodeFromName(flow_solution,           "DataY"))
  data_z             = I.getVal(I.getNodeFromName(flow_solution,           "DataZ"))
  density            = I.getVal(I.getNodeFromName(flow_solution,           "Density"))
  duplicated_data_x  = I.getVal(I.getNodeFromName(duplicated_flow_solution,"DataX"))
  duplicated_data_y  = I.getVal(I.getNodeFromName(duplicated_flow_solution,"DataY"))
  duplicated_data_z  = I.getVal(I.getNodeFromName(duplicated_flow_solution,"DataZ"))
  duplicated_density = I.getVal(I.getNodeFromName(duplicated_flow_solution,"Density"))
  assert(np.allclose(data_x,-duplicated_data_x))
  assert(np.allclose(data_y,-duplicated_data_y))
  assert(np.allclose(data_z, duplicated_data_z))
  assert((density == duplicated_density).all())
    
  zone_subregion            = I.getNodeFromType1(zone,           'ZoneSubRegion_t')
  duplicated_zone_subregion = I.getNodeFromType1(duplicated_zone,'ZoneSubRegion_t')
  toto_x            = I.getVal(I.getNodeFromName(zone_subregion,           "TotoX"))
  toto_y            = I.getVal(I.getNodeFromName(zone_subregion,           "TotoY"))
  toto_z            = I.getVal(I.getNodeFromName(zone_subregion,           "TotoZ"))
  tata              = I.getVal(I.getNodeFromName(zone_subregion,           "Tata"))
  duplicated_toto_x = I.getVal(I.getNodeFromName(duplicated_zone_subregion,"TotoX"))
  duplicated_toto_y = I.getVal(I.getNodeFromName(duplicated_zone_subregion,"TotoY"))
  duplicated_toto_z = I.getVal(I.getNodeFromName(duplicated_zone_subregion,"TotoZ"))
  duplicated_tata   = I.getVal(I.getNodeFromName(duplicated_zone_subregion,"Tata"))
  assert(np.allclose(toto_x,-duplicated_toto_x))
  assert(np.allclose(toto_y,-duplicated_toto_y))
  assert(np.allclose(toto_z, duplicated_toto_z))
  assert((tata == duplicated_tata).all())
    
  discrete_data            = I.getNodeFromType1(zone,           'DiscreteData_t')
  duplicated_discrete_data = I.getNodeFromType1(duplicated_zone,'DiscreteData_t')
  titi_x            = I.getVal(I.getNodeFromName(discrete_data,           "TitiX"))
  titi_y            = I.getVal(I.getNodeFromName(discrete_data,           "TitiY"))
  titi_z            = I.getVal(I.getNodeFromName(discrete_data,           "TitiZ"))
  tyty              = I.getVal(I.getNodeFromName(discrete_data,           "Tyty"))
  duplicated_titi_x = I.getVal(I.getNodeFromName(duplicated_discrete_data,"TitiX"))
  duplicated_titi_y = I.getVal(I.getNodeFromName(duplicated_discrete_data,"TitiY"))
  duplicated_titi_z = I.getVal(I.getNodeFromName(duplicated_discrete_data,"TitiZ"))
  duplicated_tyty   = I.getVal(I.getNodeFromName(duplicated_discrete_data,"Tyty"))
  assert(np.allclose(titi_x,-duplicated_titi_x))
  assert(np.allclose(titi_y,-duplicated_titi_y))
  assert(np.allclose(titi_z, duplicated_titi_z))
  assert((tyty == duplicated_tyty).all())
  
  bc_data            = I.getNodeFromPath(zone,           'ZoneBC/Bnd2/BCDataSet/DirichletData')
  duplicated_bc_data = I.getNodeFromPath(duplicated_zone,'ZoneBC/Bnd2/BCDataSet/DirichletData')
  tutu_x            = I.getVal(I.getNodeFromName(bc_data,           "TutuX"))
  tutu_y            = I.getVal(I.getNodeFromName(bc_data,           "TutuY"))
  tutu_z            = I.getVal(I.getNodeFromName(bc_data,           "TutuZ"))
  tete              = I.getVal(I.getNodeFromName(bc_data,           "Tete"))
  duplicated_tutu_x = I.getVal(I.getNodeFromName(duplicated_bc_data,"TutuX"))
  duplicated_tutu_y = I.getVal(I.getNodeFromName(duplicated_bc_data,"TutuY"))
  duplicated_tutu_z = I.getVal(I.getNodeFromName(duplicated_bc_data,"TutuZ"))
  duplicated_tete   = I.getVal(I.getNodeFromName(duplicated_bc_data,"Tete"))
  assert(np.allclose(tutu_x,-duplicated_tutu_x))
  assert(np.allclose(tutu_y,-duplicated_tutu_y))
  assert(np.allclose(tutu_z, duplicated_tutu_z))
  assert((tete == duplicated_tete).all())
###############################################################################

###############################################################################
# @mark_mpi_test([1,2,3])
@mark_mpi_test([1,])
def test_duplicate_n_zones_from_periodic_join(sub_comm):
  yaml_dir  = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'share', 'meshes')
  yaml_path = os.path.join(yaml_dir, 'quarter_crown_square_8.yaml')
  tree = IOT.file_to_dist_tree(yaml_path,sub_comm)
  
  match_perio_by_trans_a = 'Base/Zone/ZoneGridConnectivity/match1_0.0'
  match_perio_by_trans_b = 'Base/Zone/ZoneGridConnectivity/match1_1.0'
  JN_for_duplication_paths = [[match_perio_by_trans_a],[match_perio_by_trans_b]]
  
  duplicate._duplicate_n_zones_from_periodic_join(tree,I.getZones(tree),
                                                  JN_for_duplication_paths)
  
  assert (len(I.getZones(tree)) == 2)
  
  # TODO
  # verification des coordonnees
  # verification des ordinaux
  # vérification des raccords
  # >> doublement de la translation
  # >> mise a jour des valeurs
  # >> mise à jour des autres raccords ?
  

###############################################################################

###############################################################################
# @mark_mpi_test([1,2,3])
@mark_mpi_test([1,])
def test_duplicate_zones_from_periodic_join_by_rotation_to_360(sub_comm):
  yaml_dir  = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'share', 'meshes')
  yaml_path = os.path.join(yaml_dir, 'quarter_crown_square_8.yaml')
  tree = IOT.file_to_dist_tree(yaml_path,sub_comm)
  
  match_perio_by_rot_a = 'Base/Zone/ZoneGridConnectivity/match1_0'
  match_perio_by_rot_b = 'Base/Zone/ZoneGridConnectivity/match1_1'
  JN_for_duplication_paths = [[match_perio_by_rot_a],[match_perio_by_rot_b]]
  
  duplicate._duplicate_zones_from_periodic_join_by_rotation_to_360(tree,I.getZones(tree),
                                                                   JN_for_duplication_paths)
  
  assert (len(I.getZones(tree)) == 4)
  
  # TODO
  # verification des coordonnees
  # verification des ordinaux
  # vérification des raccords
  # >> doublement de la translation
  # >> mise a jour des valeurs
  # >> mise à jour des autres raccords ?
