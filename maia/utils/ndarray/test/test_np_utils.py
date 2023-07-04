import pytest
import numpy as np

from maia.utils.ndarray import np_utils

def test_interweave_arrays():
  first  = np.array([1,2,3], dtype=np.int32)
  second = np.array([11,22,33], dtype=np.int32)
  third  = np.array([111,222,333], dtype=np.int32)
  assert (np_utils.interweave_arrays([first]) == [1,2,3]).all()
  assert (np_utils.interweave_arrays([second, third]) == \
      [11,111,22,222,33,333]).all()
  assert (np_utils.interweave_arrays([first, second, third]) == \
      [1,11,111,2,22,222,3,33,333]).all()

def test_single_dim_pr_to_pl():
  no_dist = np_utils.single_dim_pr_to_pl(np.array([[20, 25]]))
  assert no_dist.dtype == np.array([[20,25]]).dtype
  assert (no_dist == np.arange(20, 25+1)).all()
  assert no_dist.ndim == 2 and no_dist.shape[0] == 1
  dist = np_utils.single_dim_pr_to_pl(np.array([[20, 25]], dtype=np.int32), np.array([10,15,20]))
  assert dist.dtype == np.int32
  assert (dist == np.arange(10+20, 15+20)).all()
  assert dist.ndim == 2 and dist.shape[0] == 1
  with pytest.raises(AssertionError):
    np_utils.single_dim_pr_to_pl(np.array([[20, 25], [1,1]]))

def test_compress():
  idx, val = np_utils.compress(np.array([1,1,1,2,2,3,4,4,4,4,4]))
  assert np.array_equal(idx, np.array([0,3,5,6,11]))
  assert np.array_equal(val, np.array([1,2,3,4]))
  idx, val = np_utils.compress(np.array([42,42,42,42]))
  assert np.array_equal(idx, np.array([0,4]))
  assert np.array_equal(val, np.array([42]))
  with pytest.raises(AssertionError):
    idx, val = np_utils.compress(np.array([]))

def test_sizes_to_indices():
  assert(np_utils.sizes_to_indices([]) == np.zeros(1))
  assert(np_utils.sizes_to_indices([5,3,5,10]) == np.array([0,5,8,13,23])).all()
  assert(np_utils.sizes_to_indices([5,0,0,10]) == np.array([0,5,5,5,15])).all()
  assert np_utils.sizes_to_indices([5,0,0,10], np.int32).dtype == np.int32
  assert np_utils.sizes_to_indices([5,0,0,10], np.int64).dtype == np.int64

def test_shift_nonzeros():
  array = np.array([2,0,1,-1,0,3])
  np_utils.shift_nonzeros(array, 4)
  assert (array == [2+4, 0, 1+4, -1+4, 0, 3+4]).all()
  array = np.array([[12,0], [11,12],[0,13], [0,0]])
  np_utils.shift_nonzeros(array, -10)
  assert (array == [[2,0], [1,2], [0,3], [0,0]]).all()

def test_interlaced_indexed():
  array = np.array([3, 11,12,13, 4, 9,8,7,6,  3, 77,88,99])
  idx, data = np_utils.interlaced_to_indexed(3, array)
  assert (idx == [0,3,7,10]).all()
  assert (data == [11,12,13, 9,8,7,6, 77,88,99]).all()
  assert data.dtype == array.dtype
  array2 = np_utils.indexed_to_interlaced(idx, data)
  assert np.array_equal(array, array2)

def test_shift_absvalue():
  array = np.array([2,9,1,-1,3,-8,0])
  id_bck = id(array)
  np_utils.shift_absvalue(array, 100)
  assert (array == [102,109,101,-101,103,-108,100]).all()
  assert id(array) == id_bck

def test_shifted_to_local():
  out, ids = np_utils.shifted_to_local(np.array([]), np.array([0, 10, 15, 20]))
  assert out.size == ids.size == 0
  out, ids = np_utils.shifted_to_local(np.array([1,16,8,4,12,20]), np.array([0, 10, 15, 20]))
  assert np.array_equal(out, [1,1,8,4,2,5])
  assert np.array_equal(ids, [1,3,1,1,2,3])

def test_reverse_connectivity():
  ids   = np.array([8,51,6,30,29])
  idx   = np.array([0,3,6,10,13,17])
  array = np.array([7,29,32, 32,11,13, 4,32,29,61, 32,4,13, 44,11,32,7])

  r_ids, r_idx, r_array = np_utils.reverse_connectivity(ids, idx, array)

  assert (r_ids == [4,7,11,13,29,32,44,61]).all()
  assert (r_idx == [0,2,4,6,8,10,15,16,17]).all()
  assert (r_array == [30,6, 8,29, 51,29, 51,30,  8, 6, 8,6,29,30,51, 29, 6]).all()

def test_multi_arange():
  # With only one start/stop, same as np.arange
  assert (np_utils.multi_arange([0], [10]) == [0,1,2,3,4,5,6,7,8,9]).all()

  assert (np_utils.multi_arange([0,100], [10,105]) == [0,1,2,3,4,5,6,7,8,9,  100,101,102,103,104]).all()

  # Empty aranges produce no values
  assert (np_utils.multi_arange([1,3,4,6], [1,5,7,6]) == [ 3,4, 4,5,6 ]).all()

  # No start/stop
  assert np_utils.multi_arange(np.empty(0, np.int64), np.empty(0, np.int64)).size == 0

def test_arange_with_jumps():
  assert (np_utils.arange_with_jumps([0         ,5   , 10      , 13  , 18   , 20], \
                                     [False     ,True, False   , True, False]) == \
                                     [0,1,2,3,4      , 10,11,12      , 18,19]).all()

def test_jagged_extract():
  idx_array = np.array([0,2,6,10,10])
  array = np.array([0,1, 2,3,4,5, 6,7,8,9  ])

  idx_e, array_e = np_utils.jagged_extract(idx_array, array, np.array([0,2]))
  assert (idx_e == [0,2,6]).all()
  assert (array_e == [0,1,  6,7,8,9]).all()
  idx_e, array_e = np_utils.jagged_extract(idx_array, array, np.array([0,2,3]))
  assert (idx_e == [0,2,6,6]).all()
  assert (array_e == [0,1, 6,7,8,9]).all()
  idx_e, array_e = np_utils.jagged_extract(idx_array, array, np.array([0,3]))
  assert (idx_e == [0,2,2]).all()
  assert (array_e == [0,1]).all()
  idx_e, array_e = np_utils.jagged_extract(idx_array, array, np.array([0,1,2,3]))
  assert (idx_e == idx_array).all()
  assert (array_e == array).all()
  idx_e, array_e = np_utils.jagged_extract(idx_array, array, np.array([], int))
  assert (idx_e == [0]).all()
  assert (array_e.size == 0)

def test_jagged_merge():
  idx1   = np.array([0, 1, 4], np.int32)
  array1 = np.array([.3,  .5,.6,.7])
  idx2   = np.array([0, 2, 3], np.int32)
  array2 = np.array([100.,101,  105])
  idx, array = np_utils.jagged_merge(idx1, array1, idx2, array2)
  assert (idx == [0, 3, 7]).all()
  assert idx.dtype == np.int32
  assert (array == [.3,100,101,  .5,.6,.7,105]).all()
  assert array.dtype == float

  idx1  = np.array([0, 2, 2, 3], np.int32)
  array1 = np.array([12,13,  15])
  idx2  = np.array([0, 1, 2, 3], np.int32)
  array2 = np.array([101, 102, 103])
  idx, array = np_utils.jagged_merge(idx1, array1, idx2, array2)
  assert (idx == [0, 3, 4, 6]).all()
  assert (array == [12,13,101, 102, 15,103]).all()
  assert array.dtype == array1.dtype

def test_roll_from():
  assert (np_utils.roll_from(np.array([2,4,8,16]), start_idx = 1) == [4,8,16,2]).all()
  assert (np_utils.roll_from(np.array([2,4,8,16]), start_value = 4) == [4,8,16,2]).all()
  assert (np_utils.roll_from(np.array([2,4,8,16]), start_value = 8, reverse=True) == [8,4,2,16]).all()
  with pytest.raises(AssertionError):
    np_utils.roll_from(np.array([2,4,8,16]), start_idx = 1, start_value = 8)

def test_others_mask():
  array = np.array([2,4,6,1,3,5])
  assert (np_utils.others_mask(array, np.empty(0, np.int32)) == [1,1,1,1,1,1]).all()
  assert (np_utils.others_mask(array, np.array([2,1]))       == [1,0,0,1,1,1]).all()
  assert (np_utils.others_mask(array, np.array([0,1,3,4,5])) == [0,0,1,0,0,0]).all()

def test_concatenate_np_arrays():
  a1 = np.array([2, 4, 6, 8])
  a2 = np.array([10, 20, 30, 40, 50, 60])
  a3 = np.array([100])
  av = np.empty(0, np.int64)
  array_idx, array = np_utils.concatenate_np_arrays([a1,a3,av,a2])
  assert (array_idx == [0,4,5,5,11]).all()
  assert (array == [2,4,6,8,100,10,20,30,40,50,60]).all()
  assert array.dtype == np.int64

  array_idx, array = np_utils.concatenate_np_arrays([av])
  assert (array_idx == [0,0]).all()
  assert (array == []).all()

def test_concatenate_point_list():
  pl1 = np.array([[2, 4, 6, 8]])
  pl2 = np.array([[10, 20, 30, 40, 50, 60]])
  pl3 = np.array([[100]])
  plvoid = np.empty((1,0))

  #No pl at all in the mesh
  with pytest.raises(ValueError):
    none_idx, none = np_utils.concatenate_point_list([])
  none_idx, none = np_utils.concatenate_point_list([], np.int32)
  assert none_idx == [0]
  assert isinstance(none, np.ndarray) and none.shape == (0,)

  #A pl, but with no data
  empty_idx, empty = np_utils.concatenate_point_list([plvoid])
  assert (none_idx == [0,0]).all()
  assert isinstance(empty, np.ndarray)
  assert empty.shape == (0,)

  # A pl with data
  one_idx, one = np_utils.concatenate_point_list([pl1])
  assert (one_idx == [0,4]).all()
  assert (one     == pl1[0]).all()

  # Several pl
  merged_idx, merged = np_utils.concatenate_point_list([pl1, pl2, pl3])
  assert (merged_idx == [0, pl1.size, pl1.size+pl2.size, pl1.size+pl2.size+pl3.size]).all()
  assert (merged[0:pl1.size]                 == pl1[0]).all()
  assert (merged[pl1.size:pl1.size+pl2.size] == pl2[0]).all()
  assert (merged[pl1.size+pl2.size:]         == pl3[0]).all()
  # Several pl, some with no data
  merged_idx, merged = np_utils.concatenate_point_list([pl1, plvoid, pl2])
  assert (merged_idx == [0, 4, 4, 10]).all()
  assert (merged[0:4 ] == pl1[0]).all()
  assert (merged[4:10] == pl2[0]).all()

def test_any_in_range():
  assert np_utils.any_in_range([3,4,1,6,12,3], 2, 20, strict=False)
  assert not np_utils.any_in_range([3,4,2,6,12,3], 15, 20, strict=False)
  assert np_utils.any_in_range([3,4,1,6,12,3], 12, 20, strict=False)
  assert not np_utils.any_in_range([3,4,1,6,12,3], 12, 20, strict=True)

def test_all_in_range():
  assert np_utils.all_in_range([3,4,5,6,12,3], 2, 20, strict=False)
  assert not np_utils.all_in_range([18,4,2,17,16,3], 15, 20, strict=False)
  assert not np_utils.all_in_range([3,4,1,6,12,3], 3, 20, strict=True)


def check_transform(expected_x, expected_y, expected_z, computed_matrix, computed_x, computed_y, computed_z, atol):
  # Check matrix
  expected_matrix = np.array([expected_x, expected_y, expected_z], order='F')
  assert np.allclose(expected_matrix, computed_matrix, rtol=0., atol=atol)
  # Check vectors
  assert np.allclose(expected_x, computed_x, rtol=0., atol=atol)
  assert np.allclose(expected_y, computed_y, rtol=0., atol=atol)
  assert np.allclose(expected_z, computed_z, rtol=0., atol=atol)

class Test_transform_simple():
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
  def test_none(self):
    rotation_center = [0.,0.,0.]
    rotation_angle  = [0.,0.,0.]
    translation     = [0.,0.,0.]

    modified_vectors = np_utils.transform_cart_matrix(self.vectors, translation, rotation_center, rotation_angle)
    (mod_vx, mod_vy, mod_vz) = np_utils.transform_cart_vectors(self.vx, self.vy, self.vz, translation, rotation_center, rotation_angle)
    check_transform(self.vx, self.vy, self.vz, modified_vectors, mod_vx, mod_vy, mod_vz, 0)

  # --------------------------------------------------------------------------- #  
  def test_rotation_without_rotation_center(self):
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

    modified_vectors = np_utils.transform_cart_matrix(self.vectors, translation, rotation_center, rotation_angle)
    (mod_vx, mod_vy, mod_vz) = np_utils.transform_cart_vectors(self.vx, self.vy, self.vz, translation, rotation_center, rotation_angle)
    check_transform(expected_vx, expected_vy, expected_vz, modified_vectors, mod_vx, mod_vy, mod_vz, 5.e-15)
  # --------------------------------------------------------------------------- #  
  def test_rotation_with_rotation_center(self):
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

    modified_vectors = np_utils.transform_cart_matrix(self.vectors, translation, rotation_center, rotation_angle)
    (mod_vx, mod_vy, mod_vz) = np_utils.transform_cart_vectors(self.vx, self.vy, self.vz, translation, rotation_center, rotation_angle)
    check_transform(expected_vx, expected_vy, expected_vz, modified_vectors, mod_vx, mod_vy, mod_vz, 5.e-15)
  # --------------------------------------------------------------------------- #  
  def test_translation(self):
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

    modified_vectors = np_utils.transform_cart_matrix(self.vectors, translation, rotation_center, rotation_angle)
    (mod_vx, mod_vy, mod_vz) = np_utils.transform_cart_vectors(self.vx, self.vy, self.vz, translation, rotation_center, rotation_angle)
    check_transform(expected_vx, expected_vy, expected_vz, modified_vectors, mod_vx, mod_vy, mod_vz, 5.e-15)
  # --------------------------------------------------------------------------- #  
  def test_rotation_and_translation(self):
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

    modified_vectors = np_utils.transform_cart_matrix(self.vectors, translation, rotation_center, rotation_angle)
    (mod_vx, mod_vy, mod_vz) = np_utils.transform_cart_vectors(self.vx, self.vy, self.vz, translation, rotation_center, rotation_angle)
    check_transform(expected_vx, expected_vy, expected_vz, modified_vectors, mod_vx, mod_vy, mod_vz, 5.e-15)


class Test_transform_cart_matrix():
  vx = [ 0. , 1. , 2. ,3. ,4.5, 6.7]
  vy = [-1. ,-2  ,-3.5,4. ,9.8, 7. ]
  vz = [ 0.0, 1.1, 0.0,1.1,2.2,-3.3]
  vectors = np.array([vx,vy,vz], order='F')
  # --------------------------------------------------------------------------- #
  def test_none(self):
    rotation_center = [0.,0.,0.]
    rotation_angle  = [0.,0.,0.]
    translation     = [0.,0.,0.]

    modified_vectors = np_utils.transform_cart_matrix(self.vectors, translation, rotation_center, rotation_angle)
    (mod_vx, mod_vy, mod_vz) = np_utils.transform_cart_vectors(self.vx, self.vy, self.vz, translation, rotation_center, rotation_angle)
    check_transform(self.vx, self.vy, self.vz, modified_vectors, mod_vx, mod_vy, mod_vz, 0)
  # --------------------------------------------------------------------------- #  
  def test_rotation1(self):
    rotation_center = [0.,1.,2.]
    rotation_angle  = [0.,0.,0.]
    translation     = [0.,0.,0.]
    expected_vx     = [ 0., 1. , 2. ,3. ,4.5, 6.7]
    expected_vy     = [-1.,-2. ,-3.5,4. ,9.8, 7. ]
    expected_vz     = [ 0., 1.1, 0. ,1.1,2.2,-3.3]
    expected_vectors = np.array([expected_vx,expected_vy,expected_vz], order='F')

    modified_vectors = np_utils.transform_cart_matrix(self.vectors, translation, rotation_center, rotation_angle)
    (mod_vx, mod_vy, mod_vz) = np_utils.transform_cart_vectors(self.vx, self.vy, self.vz, translation, rotation_center, rotation_angle)
    check_transform(expected_vx, expected_vy, expected_vz, modified_vectors, mod_vx, mod_vy, mod_vz, 5.e-15)
  # --------------------------------------------------------------------------- #  
  def test_rotation2(self):
    rotation_center = [0. ,0.,0.]
    rotation_angle  = [2.5,6.,3.]
    translation     = [0. ,0.,0.]
    expected_vx     = [ 0.13549924,-0.98691995,-1.42687542,-3.70103814,-6.22013284,-6.39518477]
    expected_vy     = [-0.81672459,-2.21305635,-2.75355304, 2.79227425, 6.97591548, 7.9650614 ]
    expected_vz     = [ 0.56089294, 0.5816963 , 2.57526158,-2.17152509,-5.81175968, 0.66287908]
    expected_vectors = np.array([expected_vx,expected_vy,expected_vz], order='F')

    modified_vectors = np_utils.transform_cart_matrix(self.vectors, translation, rotation_center, rotation_angle)
    (mod_vx, mod_vy, mod_vz) = np_utils.transform_cart_vectors(self.vx, self.vy, self.vz, translation, rotation_center, rotation_angle)
    check_transform(expected_vx, expected_vy, expected_vz, modified_vectors, mod_vx, mod_vy, mod_vz, 5.e-8)
  # --------------------------------------------------------------------------- #  
  def test_rotation3(self):
    rotation_center = [0. ,1.,2.]
    rotation_angle  = [2.5,6.,3.]
    translation     = [0. ,0.,0.]
    expected_vx     = [0.82982947,-0.29258972,-0.73254519,-3.00670791,-5.52580261,-5.70085453]
    expected_vy     = [0.51582115,-0.88051061,-1.42100729, 4.12481999, 8.30846123, 9.29760715]
    expected_vz     = [4.66025448, 4.68105784, 6.67462311, 1.92783644,-1.71239815, 4.76224062]
    expected_vectors = np.array([expected_vx,expected_vy,expected_vz], order='F')

    modified_vectors = np_utils.transform_cart_matrix(self.vectors, translation, rotation_center, rotation_angle)
    (mod_vx, mod_vy, mod_vz) = np_utils.transform_cart_vectors(self.vx, self.vy, self.vz, translation, rotation_center, rotation_angle)
    check_transform(expected_vx, expected_vy, expected_vz, modified_vectors, mod_vx, mod_vy, mod_vz, 5.e-8)
  # --------------------------------------------------------------------------- #  
  def test_translation(self):
    rotation_center = [0. ,0. , 0. ]
    rotation_angle  = [0. ,0. , 0. ]
    translation     = [1.2,2.3,-0.1]
    expected_vx     = [ 1.2,2.2, 3.2,4.2, 5.7, 7.9]
    expected_vy     = [ 1.3,0.3,-1.2,6.3,12.1, 9.3]
    expected_vz     = [-0.1,1. ,-0.1,1.0, 2.1,-3.4]
    expected_vectors = np.array([expected_vx,expected_vy,expected_vz], order='F')

    modified_vectors = np_utils.transform_cart_matrix(self.vectors, translation, rotation_center, rotation_angle)
    (mod_vx, mod_vy, mod_vz) = np_utils.transform_cart_vectors(self.vx, self.vy, self.vz, translation, rotation_center, rotation_angle)
    check_transform(expected_vx, expected_vy, expected_vz, modified_vectors, mod_vx, mod_vy, mod_vz, 5.e-15)
  # --------------------------------------------------------------------------- #  
  def test_rotation_and_translation(self):
    rotation_center = [0. ,1. , 2. ]
    rotation_angle  = [2.5,6. , 3. ]
    translation     = [4. ,2.5,-1.3]
    expected_vx     = [4.82982947,3.70741028,3.26745481,0.99329209,-1.52580261,-1.70085453]
    expected_vy     = [3.01582115,1.61948939,1.07899271,6.62481999,10.80846123,11.79760715]
    expected_vz     = [3.36025448,3.38105784,5.37462311,0.62783644,-3.01239815, 3.46224062]
    expected_vectors = np.array([expected_vx,expected_vy,expected_vz], order='F')

    modified_vectors = np_utils.transform_cart_matrix(self.vectors, translation, rotation_center, rotation_angle)
    (mod_vx, mod_vy, mod_vz) = np_utils.transform_cart_vectors(self.vx, self.vy, self.vz, translation, rotation_center, rotation_angle)
    check_transform(expected_vx, expected_vy, expected_vz, modified_vectors, mod_vx, mod_vy, mod_vz, 5.e-8)


def test_safe_int_cast():
  array32 = np.random.randint(-10000, 10000, size=25, dtype=np.int32)
  assert np_utils.safe_int_cast(array32, np.int32) is array32
  array32_to_64 = np_utils.safe_int_cast(array32, np.int64)
  assert array32_to_64 is not array32
  assert np.array_equal(array32_to_64, array32) and array32_to_64.dtype == np.int64
  with pytest.raises(ValueError):
    assert np_utils.safe_int_cast(array32, float)

  array64 = np.random.randint(-10000, 10000, size=25, dtype=np.int64)
  assert np_utils.safe_int_cast(array64, np.int64) is array64
  array64_to_32 = np_utils.safe_int_cast(array64, np.int32)
  assert np.array_equal(array64_to_32, array64) and array64_to_32.dtype == np.int32

  array64 = np.random.randint(3000000000, 4000000000, size=25, dtype=np.int64)
  with pytest.raises(OverflowError):
    assert np_utils.safe_int_cast(array64, np.int32)

  empty = np.empty(0, np.int64)
  assert np_utils.safe_int_cast(empty, np.int32).dtype == np.int32
