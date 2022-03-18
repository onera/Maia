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
  
  zone_basename = I.getName(I.getZones(tree)[0])
  
  duplicate.duplicate_n_zones_from_periodic_join(tree,I.getZones(tree),
                                                  JN_for_duplication_paths)
  
  assert (len(I.getZones(tree)) == 2)
  
  zone0 = I.getZones(tree)[0]
  zone1 = I.getZones(tree)[1]
  assert((I.getName(zone0) == zone_basename+".D0") and (I.getName(zone1) == zone_basename+".D1"))

  coord0_x = I.getVal(I.getNodeFromName(zone0,"CoordinateX"))
  coord0_y = I.getVal(I.getNodeFromName(zone0,"CoordinateY"))
  coord0_z = I.getVal(I.getNodeFromName(zone0,"CoordinateZ"))
  coord1_x = I.getVal(I.getNodeFromName(zone1,"CoordinateX"))
  coord1_y = I.getVal(I.getNodeFromName(zone1,"CoordinateY"))
  coord1_z = I.getVal(I.getNodeFromName(zone1,"CoordinateZ"))
  assert(np.allclose(coord0_x,coord1_x   ))
  assert(np.allclose(coord0_y,coord1_y   ))
  assert(np.allclose(coord0_z,coord1_z+2.))
  
  zgc0 = I.getNodeFromType1(zone0,"ZoneGridConnectivity_t")
  zgc1 = I.getNodeFromType1(zone1,"ZoneGridConnectivity_t")
  for gc0 in I.getNodesFromType1(zgc0,"GridConnectivity_t"):
    gc0_name = I.getName(gc0)
    gc1 = I.getNodeFromNameAndType(zgc1,gc0_name,"GridConnectivity_t")
    ordinal0 = I.getVal(I.getNodeFromNameAndType(gc0,"Ordinal","UserDefinedData_t"))
    ordinal1 = I.getVal(I.getNodeFromNameAndType(gc1,"Ordinal","UserDefinedData_t"))
    assert(ordinal0 == ordinal1-4)
    if gc0_name in ["match1_0","match1_1"]: #Joins by rotation
      # TODO : need developpement to correct value of gcs for gcs not concerned by duplication
      # assert(I.getValue(gc0)==zone_basename+".D0")
      # assert(I.getValue(gc1)==zone_basename+".D1")
      gcp0 = I.getNodeFromType1(gc0,"GridConnectivityProperty_t")
      gcp1 = I.getNodeFromType1(gc1,"GridConnectivityProperty_t")
      perio0 = I.getNodeFromType1(gcp0,"Periodic_t")
      perio1 = I.getNodeFromType1(gcp1,"Periodic_t")
      rotation_center0 = I.getVal(I.getNodeFromName1(perio0,"RotationCenter"))
      rotation_center1 = I.getVal(I.getNodeFromName1(perio1,"RotationCenter"))
      rotation_angle0  = I.getVal(I.getNodeFromName1(perio0,"RotationAngle"))
      rotation_angle1  = I.getVal(I.getNodeFromName1(perio1,"RotationAngle"))
      translation0     = I.getVal(I.getNodeFromName1(perio0,"Translation"))
      translation1     = I.getVal(I.getNodeFromName1(perio1,"Translation"))
      assert((rotation_center0 == rotation_center1).all())
      assert((rotation_angle0  == rotation_angle1).all())
      assert((translation0     == translation1).all())
    elif gc0_name in ["match1_0.0","match1_1.0"]: #Joins by translation
      assert(I.getValue(gc0)==zone_basename+".D1")
      assert(I.getValue(gc1)==zone_basename+".D0")
      if gc0_name == "match1_0.0": #Join0 => perio*2 and join1 => not perio
        gcp0 = I.getNodeFromType1(gc0,"GridConnectivityProperty_t")
        gcp1 = I.getNodeFromType1(gc1,"GridConnectivityProperty_t")
        assert(gcp1 is None)
        perio0 = I.getNodeFromType1(gcp0,"Periodic_t")
        rotation_center0 = I.getVal(I.getNodeFromName1(perio0,"RotationCenter"))
        rotation_angle0  = I.getVal(I.getNodeFromName1(perio0,"RotationAngle"))
        translation0     = I.getVal(I.getNodeFromName1(perio0,"Translation"))
        assert((rotation_center0 == np.array([0.0, 0.0,  0.0])  ).all())
        assert((rotation_angle0  == np.array([0.0, 0.0,  0.0])  ).all())
        assert((translation0     == np.array([0.0, 0.0, -2.0])*2).all())
      else: #Join0 => not perio and join1 => perio*2
        gcp0 = I.getNodeFromType1(gc0,"GridConnectivityProperty_t")
        gcp1 = I.getNodeFromType1(gc1,"GridConnectivityProperty_t")
        assert(gcp0 is None)
        perio1 = I.getNodeFromType1(gcp1,"Periodic_t")
        rotation_center1 = I.getVal(I.getNodeFromName1(perio1,"RotationCenter"))
        rotation_angle1  = I.getVal(I.getNodeFromName1(perio1,"RotationAngle"))
        translation1     = I.getVal(I.getNodeFromName1(perio1,"Translation"))
        assert((rotation_center1 == np.array([0.0, 0.0, 0.0])  ).all())
        assert((rotation_angle1  == np.array([0.0, 0.0, 0.0])  ).all())
        assert((translation1     == np.array([0.0, 0.0, 2.0])*2).all())
    else:
      assert(False)

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
  
  zone_basename = I.getName(I.getZones(tree)[0])
  
  duplicate.duplicate_zones_from_periodic_join_by_rotation_to_360(tree,I.getZones(tree),
                                                                   JN_for_duplication_paths)
  
  assert (len(I.getZones(tree)) == 4)
  
  zone0 = I.getZones(tree)[0]
  zone1 = I.getZones(tree)[1]
  zone2 = I.getZones(tree)[2]
  zone3 = I.getZones(tree)[3]
  assert((I.getName(zone0) == zone_basename+".D0") and (I.getName(zone1) == zone_basename+".D1"))
  assert((I.getName(zone2) == zone_basename+".D2") and (I.getName(zone3) == zone_basename+".D3"))

  coord0_x = I.getVal(I.getNodeFromName(zone0,"CoordinateX"))
  coord0_y = I.getVal(I.getNodeFromName(zone0,"CoordinateY"))
  coord0_z = I.getVal(I.getNodeFromName(zone0,"CoordinateZ"))
  coord1_x = I.getVal(I.getNodeFromName(zone1,"CoordinateX"))
  coord1_y = I.getVal(I.getNodeFromName(zone1,"CoordinateY"))
  coord1_z = I.getVal(I.getNodeFromName(zone1,"CoordinateZ"))
  coord2_x = I.getVal(I.getNodeFromName(zone2,"CoordinateX"))
  coord2_y = I.getVal(I.getNodeFromName(zone2,"CoordinateY"))
  coord2_z = I.getVal(I.getNodeFromName(zone2,"CoordinateZ"))
  coord3_x = I.getVal(I.getNodeFromName(zone3,"CoordinateX"))
  coord3_y = I.getVal(I.getNodeFromName(zone3,"CoordinateY"))
  coord3_z = I.getVal(I.getNodeFromName(zone3,"CoordinateZ"))
  assert(np.allclose(coord0_x,-coord2_x))
  assert(np.allclose(coord0_y,-coord2_y))
  assert(np.allclose(coord0_z, coord2_z))
  assert(np.allclose(coord1_x,-coord3_x))
  assert(np.allclose(coord1_y,-coord3_y))
  assert(np.allclose(coord1_z, coord3_z))
  assert(np.allclose(coord0_x,-coord1_y))
  assert(np.allclose(coord0_y, coord1_x))
  assert(np.allclose(coord0_z, coord1_z))
  assert(np.allclose(coord0_x, coord3_y))
  assert(np.allclose(coord0_y,-coord3_x))
  assert(np.allclose(coord0_z, coord3_z))
  
  zgc0 = I.getNodeFromType1(zone0,"ZoneGridConnectivity_t")
  zgc1 = I.getNodeFromType1(zone1,"ZoneGridConnectivity_t")
  zgc2 = I.getNodeFromType1(zone2,"ZoneGridConnectivity_t")
  zgc3 = I.getNodeFromType1(zone3,"ZoneGridConnectivity_t")
  for gc0 in I.getNodesFromType1(zgc0,"GridConnectivity_t"):
    gc0_name = I.getName(gc0)
    gc1 = I.getNodeFromNameAndType(zgc1,gc0_name,"GridConnectivity_t")
    gc2 = I.getNodeFromNameAndType(zgc2,gc0_name,"GridConnectivity_t")
    gc3 = I.getNodeFromNameAndType(zgc3,gc0_name,"GridConnectivity_t")
    ordinal0 = I.getVal(I.getNodeFromNameAndType(gc0,"Ordinal","UserDefinedData_t"))
    ordinal1 = I.getVal(I.getNodeFromNameAndType(gc1,"Ordinal","UserDefinedData_t"))
    ordinal2 = I.getVal(I.getNodeFromNameAndType(gc2,"Ordinal","UserDefinedData_t"))
    ordinal3 = I.getVal(I.getNodeFromNameAndType(gc3,"Ordinal","UserDefinedData_t"))
    assert(ordinal0 == ordinal1-4)
    assert(ordinal0 == ordinal2-8)
    assert(ordinal0 == ordinal3-12)
    if gc0_name in ["match1_0","match1_1"]: #Joins by rotation => not perio
      if gc0_name == "match1_0":
        assert(I.getValue(gc0)==zone_basename+".D3")
        assert(I.getValue(gc1)==zone_basename+".D0")
        assert(I.getValue(gc2)==zone_basename+".D1")
        assert(I.getValue(gc3)==zone_basename+".D2")
      else:
        assert(I.getValue(gc0)==zone_basename+".D1")
        assert(I.getValue(gc1)==zone_basename+".D2")
        assert(I.getValue(gc2)==zone_basename+".D3")
        assert(I.getValue(gc3)==zone_basename+".D0")
      gcp0 = I.getNodeFromType1(gc0,"GridConnectivityProperty_t")
      gcp1 = I.getNodeFromType1(gc1,"GridConnectivityProperty_t")
      gcp2 = I.getNodeFromType1(gc2,"GridConnectivityProperty_t")
      gcp3 = I.getNodeFromType1(gc3,"GridConnectivityProperty_t")
      assert(gcp0 is None)
      assert(gcp1 is None)
      assert(gcp2 is None)
      assert(gcp3 is None)
    elif gc0_name in ["match1_0.0","match1_1.0"]: #Joins by translation => no change execpt value
      # TODO : need developpement to correct value of gcs for gcs not concerned by duplication
      # assert(I.getValue(gc0)==zone_basename+".D0")
      # assert(I.getValue(gc1)==zone_basename+".D1")
      # assert(I.getValue(gc2)==zone_basename+".D2")
      # assert(I.getValue(gc3)==zone_basename+".D3")
      pass
    else:
      assert(False)
