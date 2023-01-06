from maia.io.hdf import hdf_dataspace

def test_create_flat_dataspace():
  assert hdf_dataspace.create_flat_dataspace([0, 10, 100]) == \
      [[0], [1], [10], [1], [0], [1], [10], [1], [100], [0]]
  assert hdf_dataspace.create_flat_dataspace([50, 60, 100]) == \
      [[0], [1], [60-50], [1], [50], [1], [60-50], [1], [100], [0]]
  assert hdf_dataspace.create_flat_dataspace([0, 100, 100]) == \
      [[0], [1], [100], [1], [0], [1], [100], [1], [100], [0]]
  assert hdf_dataspace.create_flat_dataspace([50, 50, 100]) == \
      [[0], [1], [0], [1], [50], [1], [0], [1], [100], [0]]
  assert hdf_dataspace.create_flat_dataspace([0, 0, 0]) == \
      [[0], [1], [0], [1], [0], [1], [0], [1], [0], [0]]

def test_create_pointlist_dataspace():
  assert hdf_dataspace.create_pointlist_dataspace([0, 10, 100]) == \
      [[0,0], [1,1], [1,10], [1,1], [0,0], [1,1], [1,10], [1,1], [1,100], [0]]
  assert hdf_dataspace.create_pointlist_dataspace([0, 10, 100],3) == \
      [[0,0], [1,1], [3,10], [1,1], [0,0], [1,1], [3,10], [1,1], [3,100], [0]]
  assert hdf_dataspace.create_pointlist_dataspace([0, 100, 100]) == \
      [[0,0], [1,1], [1,100], [1,1], [0,0], [1,1], [1,100], [1,1], [1,100], [0]]
  assert hdf_dataspace.create_pointlist_dataspace([50, 50, 100]) == \
      [[0,0], [1,1], [1,0], [1,1], [0,50], [1,1], [1,0], [1,1], [1,100], [0]]
  assert hdf_dataspace.create_pointlist_dataspace([0, 0, 0]) == \
      [[0,0], [1,1], [1,0], [1,1], [0,0], [1,1], [1,0], [1,1], [1,0], [0]]

def test_create_pe_dataspace():
  assert hdf_dataspace.create_pe_dataspace([0, 10, 100]) == \
      [[0,0], [1,1], [10,2], [1,1], [0,0], [1,1], [10,2], [1,1], [100,2], [1]]
  assert hdf_dataspace.create_pe_dataspace([0, 100, 100]) == \
      [[0,0], [1,1], [100,2], [1,1], [0,0], [1,1], [100,2], [1,1], [100,2], [1]]
  assert hdf_dataspace.create_pe_dataspace([0, 0, 0]) == \
      [[0,0], [1,1], [0,2], [1,1], [0,0], [1,1], [0,2], [1,1], [0,2], [1]]

def test_create_3D_combined_dataspace():
  assert hdf_dataspace.create_combined_dataspace([10,2,5], [0,10,100]) == \
      [[0], [1], [10], [1], [[0, 0, 0], [1, 1, 1], [10, 1, 1], [1, 1, 1]], [10, 2, 5], [0]]
  assert hdf_dataspace.create_combined_dataspace([2,5,10], [0,10,100]) == \
      [[0], [1], [10], [1], [[0, 0, 0], [1, 1, 1], [2, 5, 1], [1, 1, 1]], [2, 5, 10], [0]]
  assert hdf_dataspace.create_combined_dataspace([2,5,10], [50,50,100]) == \
      [[0], [1], [0], [1], [], [2, 5, 10], [0]]

def test_create_2D_combined_dataspace():
  assert hdf_dataspace.create_combined_dataspace([10,2], [0,10,20]) == \
      [[0], [1], [10], [1], [[0, 0], [1, 1], [10, 1], [1, 1]], [10, 2], [0]]
  assert hdf_dataspace.create_combined_dataspace([2,5], [0,3,10]) == \
      [[0], [1], [3], [1], [[0, 0], [1, 1], [2, 1], [1, 1], [0, 1], [1, 1], [1, 1], [1, 1]], [2, 5], [0]]
  assert hdf_dataspace.create_combined_dataspace([2,5], [5,5,10]) == \
      [[0], [1], [0], [1], [], [2, 5], [0]]

def test_create_data_array_filter():
  assert hdf_dataspace.create_data_array_filter([0,10,100]) == \
      hdf_dataspace.create_flat_dataspace([0,10,100])
  assert hdf_dataspace.create_data_array_filter([0,10,100], [5,4,5]) == \
      hdf_dataspace.create_combined_dataspace([5,4,5], [0,10,100])

