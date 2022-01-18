import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import Converter.Internal as I

from maia.generate.dcube_generator import dcube_generate

from maia.geometry import geometry

@mark_mpi_test(1)
def test_compute_cell_center(sub_comm):
  #Test U
  tree = dcube_generate(3, 1., [0,0,0], sub_comm)
  zoneU = I.getZones(tree)[0]
  #On partitions, element are supposed to be I4
  for elt_node in I.getNodesFromType1(zoneU, 'Elements_t'):
    for name in ['ElementConnectivity', 'ParentElements', 'ElementStartOffset']:
      node = I.getNodeFromName1(elt_node, name)
      node[1] = node[1].astype(np.int32)

  cell_center = geometry.compute_cell_center(zoneU)
  expected_cell_center = np.array([0.25, 0.25, 0.25, 
                                   0.75, 0.25, 0.25, 
                                   0.25, 0.75, 0.25, 
                                   0.75, 0.75, 0.25, 
                                   0.25, 0.25, 0.75, 
                                   0.75, 0.25, 0.75, 
                                   0.25, 0.75, 0.75, 
                                   0.75, 0.75, 0.75])
  assert (cell_center == expected_cell_center).all()

  #Test S
  cx_s = I.getNodeFromName(zoneU, 'CoordinateX')[1].reshape((3,3,3), order='F')
  cy_s = I.getNodeFromName(zoneU, 'CoordinateY')[1].reshape((3,3,3), order='F')
  cz_s = I.getNodeFromName(zoneU, 'CoordinateZ')[1].reshape((3,3,3), order='F')

  zoneS = I.newZone(zsize=[[3,2,0], [3,2,0], [3,2,0]], ztype='Structured')
  grid_coords = I.newGridCoordinates(parent=zoneS)
  I.newDataArray('CoordinateX', cx_s, parent=grid_coords)
  I.newDataArray('CoordinateY', cy_s, parent=grid_coords)
  I.newDataArray('CoordinateZ', cz_s, parent=grid_coords)
  cell_center = geometry.compute_cell_center(zoneS)
  assert (cell_center == expected_cell_center).all()

  #Test wrong case
  zone_no_ng = I.rmNodesByType(zoneU, 'Elements_t')
  with pytest.raises(NotImplementedError):
    geometry.compute_cell_center(zone_no_ng)


###############################################################################
class Test_apply_transformation_on_concatenated_components_of_cartesian_vectors_simple():
    #    
    #                    (1.,1.,0.)(2.,1.,0.)
    #                       D+---------+C      
    #                        |         |              
    #                        |         |              
    #                        |         |              
    #                        |         |              
    #              +        A+---------+B               
    #          (0.,0.,0.)(1.,0.,0.)(2.,0.,0.)
    #    
  vx = [1.,2.,2.,1.]
  vy = [0.,0.,1.,1.]
  vz = [0.,0.,0.,0.]
  vectors = np.array([vx,vy,vz], order='F')
  # --------------------------------------------------------------------------- #
  def test_apply_transformation_on_concatenated_components_of_cartesian_vectors_simple_none(self):
    rotation_center = [0.,0.,0.]
    rotation_angle  = [0.,0.,0.]
    translation     = [0.,0.,0.]
    modified_vectors = geometry.apply_transformation_on_concatenated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, self.vectors)
    assert (modified_vectors == self.vectors).all()
  # --------------------------------------------------------------------------- #  
  def test_apply_transformation_on_concatenated_components_of_cartesian_vectors_simple_rotation_without_rotation_center(self):
    #                Before rotation                            After rotation
    #                                                                                               
    #                                                C'+---------+B'                                 
    #                                                  |         |                                  
    #                                                  |         |                                  
    #                                                  |         |                                  
    #                                                  |         |                                   
    #                       D+---------+C       =>   D'+---------+A'      D+---------+C      
    #                        |         |                                   |         |              
    #                        |         |                                   |         |              
    #                        |         |                                   |         |              
    #                        |         |                                   |         |              
    #              +        A+---------+B                        +        A+---------+B               
    #          (0.,0.,0.)(1.,0.,0.)(2.,0.,0.)                (0.,0.,0.)(1.,0.,0.)(2.,0.,0.)
    #     
    rotation_center = [0.,0.,0.      ]
    rotation_angle  = [0.,0.,np.pi/2.]
    translation     = [0.,0.,0.      ]
    expected_vx     = [0.,0.,-1.,-1.]
    expected_vy     = [1.,2., 2., 1.]
    expected_vz     = [0.,0., 0., 0.]
    expected_vectors = np.array([expected_vx,expected_vy,expected_vz], order='F')
    modified_vectors = geometry.apply_transformation_on_concatenated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, self.vectors)
    assert (np.abs(modified_vectors-expected_vectors) < 5.e-15).all()
  # --------------------------------------------------------------------------- #  
  def test_apply_transformation_on_concatenated_components_of_cartesian_vectors_simple_rotation_with_rotation_center(self):
    #                 Before rotation                               After rotation
    #                                                                                                
    #                                                                    B' D
    #                       D+---------+C                      C'+---------+---------+C      
    #                        |         |                         |         |         |              
    #                        |         |        =>               |         |         |              
    #                        |         |                         |         |         |              
    #                        |         |                         |       A'|A        |              
    #              +        A+---------+B                      D'+---------+---------+B               
    #          (0.,0.,0.)(1.,0.,0.)(2.,0.,0.)                (0.,0.,0.)(1.,0.,0.)(2.,0.,0.)
    #     
    rotation_center = [1.,0.,0.      ]
    rotation_angle  = [0.,0.,np.pi/2.]
    translation     = [0.,0.,0.      ]
    expected_vx     = [1.,1.,0.,0.]
    expected_vy     = [0.,1.,1.,0.]
    expected_vz     = [0.,0.,0.,0.]
    expected_vectors = np.array([expected_vx,expected_vy,expected_vz], order='F')
    modified_vectors = geometry.apply_transformation_on_concatenated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, self.vectors)
    assert (np.abs(modified_vectors-expected_vectors) < 5.e-15).all()
  # --------------------------------------------------------------------------- #  
  def test_apply_transformation_on_concatenated_components_of_cartesian_vectors_simple_translation(self):
    #              Before translation                                  After translation
    #                                                                       
    #                                                                                        D'+---------+C'
    #                                                                                          |         | 
    #                                                                                          |         | 
    #                                                                                          |         | 
    #                                           =>                                             |         | 
    #                       D+---------+C                                  +---------+C      A'+---------+B''
    #                        |         |                                   |         |              
    #                        |         |                                   |         |              
    #                        |         |                                   |         |              
    #                        |         |                                   |A        |              
    #              +        A+---------+B                        +         +---------+B               
    #          (0.,0.,0.)(1.,0.,0.)(2.,0.,0.)                (0.,0.,0.)(1.,0.,0.)(2.,0.,0.)
    #     
    rotation_center = [0.,0.,0.]
    rotation_angle  = [0.,0.,0.]
    translation     = [2.,1.,0.]
    expected_vx     = [3.,4.,4.,3.]
    expected_vy     = [1.,1.,2.,2.]
    expected_vz     = [0.,0.,0.,0.]
    expected_vectors = np.array([expected_vx,expected_vy,expected_vz], order='F')
    modified_vectors = geometry.apply_transformation_on_concatenated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, self.vectors)
    assert (np.abs(modified_vectors-expected_vectors) < 5.e-15).all()
  # --------------------------------------------------------------------------- #  
  def test_apply_transformation_on_concatenated_components_of_cartesian_vectors_simple_rotation_and_translation(self):
    #         Before rotation and translation               After rotation and translation
    #                                                                          
    #                                                                    D'+---------+C'
    #                                                                      |         | 
    #                                                                      |         | 
    #                                                                      |         | 
    #                                           =>                         |         |A'
    #                       D+---------+C                                  +---------+
    #                        |         |                                  D|         |C 
    #                        |         |                                   |         |  
    #                        |         |                                   |         |  
    #                        |         |                                   |         |  
    #              +        A+---------+B                        +        A+---------+B 
    #          (0.,0.,0.)(1.,0.,0.)(2.,0.,0.)                (0.,0.,0.)(1.,0.,0.)(2.,0.,0.)
    #     
    rotation_center = [1.,0.,0.      ]
    rotation_angle  = [0.,0.,np.pi/2.]
    translation     = [1.,1.,0.      ]
    expected_vx     = [2.,2.,1.,1.]
    expected_vy     = [1.,2.,2.,1.]
    expected_vz     = [0.,0.,0.,0.]
    expected_vectors = np.array([expected_vx,expected_vy,expected_vz], order='F')
    modified_vectors = geometry.apply_transformation_on_concatenated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, self.vectors)
    assert (np.abs(modified_vectors-expected_vectors) < 5.e-15).all()
###############################################################################

###############################################################################
class Test_apply_transformation_on_separated_components_of_cartesian_vectors():
    #    
    #                    (1.,1.,0.)(2.,1.,0.)
    #                       D+---------+C      
    #                        |         |              
    #                        |         |              
    #                        |         |              
    #                        |         |              
    #              +        A+---------+B               
    #          (0.,0.,0.)(1.,0.,0.)(2.,0.,0.)
    #    
  vx = [1.,2.,2.,1.]
  vy = [0.,0.,1.,1.]
  vz = [0.,0.,0.,0.]
  # --------------------------------------------------------------------------- #
  def test_apply_transformation_on_separated_components_of_cartesian_vectors_none(self):
    rotation_center = [0.,0.,0.]
    rotation_angle  = [0.,0.,0.]
    translation     = [0.,0.,0.]
    (mod_vx, mod_vy, mod_vz) = geometry.apply_transformation_on_separated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, self.vx, self.vy, self.vz)
    assert (mod_vx == self.vx).all()
    assert (mod_vy == self.vy).all()
    assert (mod_vz == self.vz).all()
  # --------------------------------------------------------------------------- #  
  def test_apply_transformation_on_separated_components_of_cartesian_vectors_rotation_without_rotation_center(self):
    #                Before rotation                            After rotation
    #                                                                                               
    #                                                C'+---------+B'                                 
    #                                                  |         |                                  
    #                                                  |         |                                  
    #                                                  |         |                                  
    #                                                  |         |                                   
    #                       D+---------+C       =>   D'+---------+A'      D+---------+C      
    #                        |         |                                   |         |              
    #                        |         |                                   |         |              
    #                        |         |                                   |         |              
    #                        |         |                                   |         |              
    #              +        A+---------+B                        +        A+---------+B               
    #          (0.,0.,0.)(1.,0.,0.)(2.,0.,0.)                (0.,0.,0.)(1.,0.,0.)(2.,0.,0.)
    #     
    rotation_center = [0.,0.,0.      ]
    rotation_angle  = [0.,0.,np.pi/2.]
    translation     = [0.,0.,0.      ]
    expected_vx     = [0.,0.,-1.,-1.]
    expected_vy     = [1.,2., 2., 1.]
    expected_vz     = [0.,0., 0., 0.]
    (mod_vx, mod_vy, mod_vz) = geometry.apply_transformation_on_separated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, self.vx, self.vy, self.vz)
    assert (np.abs(mod_vx-expected_vx) < 5.e-15).all()
    assert (np.abs(mod_vy-expected_vy) < 5.e-15).all()
    assert (np.abs(mod_vz-expected_vz) < 5.e-15).all()
  # --------------------------------------------------------------------------- #  
  def test_apply_transformation_on_separated_components_of_cartesian_vectors_rotation_with_rotation_center(self):
    #                 Before rotation                               After rotation
    #                                                                                                
    #                                                                    B' D
    #                       D+---------+C                      C'+---------+---------+C      
    #                        |         |                         |         |         |              
    #                        |         |        =>               |         |         |              
    #                        |         |                         |         |         |              
    #                        |         |                         |       A'|A        |              
    #              +        A+---------+B                      D'+---------+---------+B               
    #          (0.,0.,0.)(1.,0.,0.)(2.,0.,0.)                (0.,0.,0.)(1.,0.,0.)(2.,0.,0.)
    #     
    rotation_center = [1.,0.,0.      ]
    rotation_angle  = [0.,0.,np.pi/2.]
    translation     = [0.,0.,0.      ]
    expected_vx     = [1.,1.,0.,0.]
    expected_vy     = [0.,1.,1.,0.]
    expected_vz     = [0.,0.,0.,0.]
    (mod_vx, mod_vy, mod_vz) = geometry.apply_transformation_on_separated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, self.vx, self.vy, self.vz)
    assert (np.abs(mod_vx-expected_vx) < 5.e-15).all()
    assert (np.abs(mod_vy-expected_vy) < 5.e-15).all()
    assert (np.abs(mod_vz-expected_vz) < 5.e-15).all()
  # --------------------------------------------------------------------------- #  
  def test_apply_transformation_on_separated_components_of_cartesian_vectors_translation(self):
    #              Before translation                                  After translation
    #                                                                       
    #                                                                                        D'+---------+C'
    #                                                                                          |         | 
    #                                                                                          |         | 
    #                                                                                          |         | 
    #                                           =>                                             |         | 
    #                       D+---------+C                                  +---------+C      A'+---------+B''
    #                        |         |                                   |         |              
    #                        |         |                                   |         |              
    #                        |         |                                   |         |              
    #                        |         |                                   |A        |              
    #              +        A+---------+B                        +         +---------+B               
    #          (0.,0.,0.)(1.,0.,0.)(2.,0.,0.)                (0.,0.,0.)(1.,0.,0.)(2.,0.,0.)
    #     
    rotation_center = [0.,0.,0.]
    rotation_angle  = [0.,0.,0.]
    translation     = [2.,1.,0.]
    expected_vx     = [3.,4.,4.,3.]
    expected_vy     = [1.,1.,2.,2.]
    expected_vz     = [0.,0.,0.,0.]
    (mod_vx, mod_vy, mod_vz) = geometry.apply_transformation_on_separated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, self.vx, self.vy, self.vz)
    assert (np.abs(mod_vx-expected_vx) < 5.e-15).all()
    assert (np.abs(mod_vy-expected_vy) < 5.e-15).all()
    assert (np.abs(mod_vz-expected_vz) < 5.e-15).all()
  # --------------------------------------------------------------------------- #  
  def test_apply_transformation_on_separated_components_of_cartesian_vectors_rotation_and_translation(self):
    #         Before rotation and translation               After rotation and translation
    #                                                                          
    #                                                                    D'+---------+C'
    #                                                                      |         | 
    #                                                                      |         | 
    #                                                                      |         | 
    #                                           =>                         |         |A'
    #                       D+---------+C                                  +---------+
    #                        |         |                                  D|         |C 
    #                        |         |                                   |         |  
    #                        |         |                                   |         |  
    #                        |         |                                   |         |  
    #              +        A+---------+B                        +        A+---------+B 
    #          (0.,0.,0.)(1.,0.,0.)(2.,0.,0.)                (0.,0.,0.)(1.,0.,0.)(2.,0.,0.)
    #     
    rotation_center = [1.,0.,0.      ]
    rotation_angle  = [0.,0.,np.pi/2.]
    translation     = [1.,1.,0.      ]
    expected_vx     = [2.,2.,1.,1.]
    expected_vy     = [1.,2.,2.,1.]
    expected_vz     = [0.,0.,0.,0.]
    (mod_vx, mod_vy, mod_vz) = geometry.apply_transformation_on_separated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, self.vx, self.vy, self.vz)
    assert (np.abs(mod_vx-expected_vx) < 5.e-15).all()
    assert (np.abs(mod_vy-expected_vy) < 5.e-15).all()
    assert (np.abs(mod_vz-expected_vz) < 5.e-15).all()
###############################################################################

###############################################################################
class Test_apply_transformation_on_concatenated_components_of_cartesian_vectors():
  vx = [ 0. , 1. , 2. ,3. ,4.5, 6.7]
  vy = [-1. ,-2  ,-3.5,4. ,9.8, 7. ]
  vz = [ 0.0, 1.1, 0.0,1.1,2.2,-3.3]
  vectors = np.array([vx,vy,vz], order='F')
  # --------------------------------------------------------------------------- #
  def test_apply_transformation_on_concatenated_components_of_cartesian_vectors_none(self):
    rotation_center = [0.,0.,0.]
    rotation_angle  = [0.,0.,0.]
    translation     = [0.,0.,0.]
    modified_vectors = geometry.apply_transformation_on_concatenated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, self.vectors)
    assert (modified_vectors == self.vectors).all()
  # --------------------------------------------------------------------------- #  
  def test_apply_transformation_on_concatenated_components_of_cartesian_vectors_rotation1(self):
    rotation_center = [0.,1.,2.]
    rotation_angle  = [0.,0.,0.]
    translation     = [0.,0.,0.]
    expected_vx     = [ 0., 1. , 2. ,3. ,4.5, 6.7]
    expected_vy     = [-1.,-2. ,-3.5,4. ,9.8, 7. ]
    expected_vz     = [ 0., 1.1, 0. ,1.1,2.2,-3.3]
    expected_vectors = np.array([expected_vx,expected_vy,expected_vz], order='F')
    modified_vectors = geometry.apply_transformation_on_concatenated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, self.vectors)
    assert (np.abs(modified_vectors-expected_vectors) < 5.e-15).all()
  # --------------------------------------------------------------------------- #  
  def test_apply_transformation_on_concatenated_components_of_cartesian_vectors_rotation2(self):
    rotation_center = [0. ,0.,0.]
    rotation_angle  = [2.5,6.,3.]
    translation     = [0. ,0.,0.]
    expected_vx     = [ 0.13549924,-0.98691995,-1.42687542,-3.70103814,-6.22013284,-6.39518477]
    expected_vy     = [-0.81672459,-2.21305635,-2.75355304, 2.79227425, 6.97591548, 7.9650614 ]
    expected_vz     = [ 0.56089294, 0.5816963 , 2.57526158,-2.17152509,-5.81175968, 0.66287908]
    expected_vectors = np.array([expected_vx,expected_vy,expected_vz], order='F')
    modified_vectors = geometry.apply_transformation_on_concatenated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, self.vectors)
    assert (np.abs(modified_vectors-expected_vectors) < 5.e-8).all()
  # --------------------------------------------------------------------------- #  
  def test_apply_transformation_on_concatenated_components_of_cartesian_vectors_rotation3(self):
    rotation_center = [0. ,1.,2.]
    rotation_angle  = [2.5,6.,3.]
    translation     = [0. ,0.,0.]
    expected_vx     = [0.82982947,-0.29258972,-0.73254519,-3.00670791,-5.52580261,-5.70085453]
    expected_vy     = [0.51582115,-0.88051061,-1.42100729, 4.12481999, 8.30846123, 9.29760715]
    expected_vz     = [4.66025448, 4.68105784, 6.67462311, 1.92783644,-1.71239815, 4.76224062]
    expected_vectors = np.array([expected_vx,expected_vy,expected_vz], order='F')
    modified_vectors = geometry.apply_transformation_on_concatenated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, self.vectors)
    assert (np.abs(modified_vectors-expected_vectors) < 5.e-8).all()
  # --------------------------------------------------------------------------- #  
  def test_apply_transformation_on_concatenated_components_of_cartesian_vectors_translation(self):
    rotation_center = [0. ,0. , 0. ]
    rotation_angle  = [0. ,0. , 0. ]
    translation     = [1.2,2.3,-0.1]
    expected_vx     = [ 1.2,2.2, 3.2,4.2, 5.7, 7.9]
    expected_vy     = [ 1.3,0.3,-1.2,6.3,12.1, 9.3]
    expected_vz     = [-0.1,1. ,-0.1,1.0, 2.1,-3.4]
    expected_vectors = np.array([expected_vx,expected_vy,expected_vz], order='F')
    modified_vectors = geometry.apply_transformation_on_concatenated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, self.vectors)
    assert (np.abs(modified_vectors-expected_vectors) < 5.e-15).all()
  # --------------------------------------------------------------------------- #  
  def test_apply_transformation_on_concatenated_components_of_cartesian_vectors_rotation_and_translation(self):
    rotation_center = [0. ,1. , 2. ]
    rotation_angle  = [2.5,6. , 3. ]
    translation     = [4. ,2.5,-1.3]
    expected_vx     = [4.82982947,3.70741028,3.26745481,0.99329209,-1.52580261,-1.70085453]
    expected_vy     = [3.01582115,1.61948939,1.07899271,6.62481999,10.80846123,11.79760715]
    expected_vz     = [3.36025448,3.38105784,5.37462311,0.62783644,-3.01239815, 3.46224062]
    expected_vectors = np.array([expected_vx,expected_vy,expected_vz], order='F')
    modified_vectors = geometry.apply_transformation_on_concatenated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, self.vectors)
    assert (np.abs(modified_vectors-expected_vectors) < 5.e-8).all()
###############################################################################

###############################################################################
class Test_apply_transformation_on_separated_components_of_cartesian_vectors():
  vx = [ 0. , 1. , 2. ,3. ,4.5, 6.7]
  vy = [-1. ,-2  ,-3.5,4. ,9.8, 7. ]
  vz = [ 0.0, 1.1, 0.0,1.1,2.2,-3.3]
  # --------------------------------------------------------------------------- #
  def test_apply_transformation_on_separated_components_of_cartesian_vectors_none(self):
    rotation_center = [0.,0.,0.]
    rotation_angle  = [0.,0.,0.]
    translation     = [0.,0.,0.]
    (mod_vx, mod_vy, mod_vz) = geometry.apply_transformation_on_separated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, self.vx, self.vy, self.vz)
    assert (mod_vx == self.vx).all()
    assert (mod_vy == self.vy).all()
    assert (mod_vz == self.vz).all()
  # --------------------------------------------------------------------------- #  
  def test_apply_transformation_on_separated_components_of_cartesian_vectors_rotation1(self):
    rotation_center = [0.,1.,2.]
    rotation_angle  = [0.,0.,0.]
    translation     = [0.,0.,0.]
    expected_vx     = [ 0., 1. , 2. ,3. ,4.5, 6.7]
    expected_vy     = [-1.,-2. ,-3.5,4. ,9.8, 7. ]
    expected_vz     = [ 0., 1.1, 0. ,1.1,2.2,-3.3]
    (mod_vx, mod_vy, mod_vz) = geometry.apply_transformation_on_separated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, self.vx, self.vy, self.vz)
    assert (np.abs(mod_vx-expected_vx) < 5.e-15).all()
    assert (np.abs(mod_vy-expected_vy) < 5.e-15).all()
    assert (np.abs(mod_vz-expected_vz) < 5.e-15).all()
  # --------------------------------------------------------------------------- #  
  def test_apply_transformation_on_separated_components_of_cartesian_vectors_rotation2(self):
    rotation_center = [0. ,0.,0.]
    rotation_angle  = [2.5,6.,3.]
    translation     = [0. ,0.,0.]
    expected_vx     = [ 0.13549924,-0.98691995,-1.42687542,-3.70103814,-6.22013284,-6.39518477]
    expected_vy     = [-0.81672459,-2.21305635,-2.75355304, 2.79227425, 6.97591548, 7.9650614 ]
    expected_vz     = [ 0.56089294, 0.5816963 , 2.57526158,-2.17152509,-5.81175968, 0.66287908]
    (mod_vx, mod_vy, mod_vz) = geometry.apply_transformation_on_separated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, self.vx, self.vy, self.vz)
    assert (np.abs(mod_vx-expected_vx) < 5.e-8).all()
    assert (np.abs(mod_vy-expected_vy) < 5.e-8).all()
    assert (np.abs(mod_vz-expected_vz) < 5.e-8).all()
  # --------------------------------------------------------------------------- #  
  def test_apply_transformation_on_separated_components_of_cartesian_vectors_rotation3(self):
    rotation_center = [0. ,1.,2.]
    rotation_angle  = [2.5,6.,3.]
    translation     = [0. ,0.,0.]
    expected_vx     = [0.82982947,-0.29258972,-0.73254519,-3.00670791,-5.52580261,-5.70085453]
    expected_vy     = [0.51582115,-0.88051061,-1.42100729, 4.12481999, 8.30846123, 9.29760715]
    expected_vz     = [4.66025448, 4.68105784, 6.67462311, 1.92783644,-1.71239815, 4.76224062]
    (mod_vx, mod_vy, mod_vz) = geometry.apply_transformation_on_separated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, self.vx, self.vy, self.vz)
    assert (np.abs(mod_vx-expected_vx) < 5.e-8).all()
    assert (np.abs(mod_vy-expected_vy) < 5.e-8).all()
    assert (np.abs(mod_vz-expected_vz) < 5.e-8).all()
  # --------------------------------------------------------------------------- #  
  def test_apply_transformation_on_separated_components_of_cartesian_vectors_translation(self):
    rotation_center = [0. ,0. , 0. ]
    rotation_angle  = [0. ,0. , 0. ]
    translation     = [1.2,2.3,-0.1]
    expected_vx     = [ 1.2,2.2, 3.2,4.2, 5.7, 7.9]
    expected_vy     = [ 1.3,0.3,-1.2,6.3,12.1, 9.3]
    expected_vz     = [-0.1,1. ,-0.1,1.0, 2.1,-3.4]
    (mod_vx, mod_vy, mod_vz) = geometry.apply_transformation_on_separated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, self.vx, self.vy, self.vz)
    assert (np.abs(mod_vx-expected_vx) < 5.e-15).all()
    assert (np.abs(mod_vy-expected_vy) < 5.e-15).all()
    assert (np.abs(mod_vz-expected_vz) < 5.e-15).all()
  # --------------------------------------------------------------------------- #  
  def test_apply_transformation_on_separated_components_of_cartesian_vectors_rotation_and_translation(self):
    rotation_center = [0. ,1. , 2. ]
    rotation_angle  = [2.5,6. , 3. ]
    translation     = [4. ,2.5,-1.3]
    expected_vx     = [4.82982947,3.70741028,3.26745481,0.99329209,-1.52580261,-1.70085453]
    expected_vy     = [3.01582115,1.61948939,1.07899271,6.62481999,10.80846123,11.79760715]
    expected_vz     = [3.36025448,3.38105784,5.37462311,0.62783644,-3.01239815, 3.46224062]
    (mod_vx, mod_vy, mod_vz) = geometry.apply_transformation_on_separated_components_of_cartesian_vectors(rotation_center, rotation_angle, translation, self.vx, self.vy, self.vz)
    assert (np.abs(mod_vx-expected_vx) < 5.e-8).all()
    assert (np.abs(mod_vy-expected_vy) < 5.e-8).all()
    assert (np.abs(mod_vz-expected_vz) < 5.e-8).all()
###############################################################################
