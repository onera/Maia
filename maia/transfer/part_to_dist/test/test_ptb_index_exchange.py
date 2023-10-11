import pytest
import pytest_parallel
import numpy      as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.pytree.yaml import parse_yaml_cgns
from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype
import maia.transfer.part_to_dist.index_exchange as IPTB

from maia import npy_pdm_gnum_dtype as pdm_dtype
dtype = 'I4' if pdm_dtype == np.int32 else 'I8'

@pytest_parallel.mark.parallel(4)
def test_create_part_pl_gnum_unique(comm):
  part_zones = [PT.new_Zone('Zone.P{0}.N0'.format(comm.Get_rank()))]
  if comm.Get_rank() == 0:
    PT.new_ZoneSubRegion("ZSR", point_list=[[2,4,6,8]], loc='Vertex', parent=part_zones[0])
  if comm.Get_rank() == 2:
    part_zones.append(PT.new_Zone('Zone.P2.N1'))
    PT.new_ZoneSubRegion("ZSR", point_list=[[1,3]], loc='Vertex', parent=part_zones[0])
    PT.new_ZoneSubRegion("ZSR", point_list=[[2,4,6]], loc='Vertex', parent=part_zones[1])
  if comm.Get_rank() == 3:
    part_zones = []
  IPTB.create_part_pl_gnum_unique(part_zones, "ZSR", comm)

  for p_zone in part_zones:
    if PT.get_child_from_name(p_zone, "ZSR") is not None:
      assert PT.get_node_from_path(p_zone, "ZSR/:CGNS#GlobalNumbering/Index") is not None 
  if comm.Get_rank() == 0:
    assert (PT.get_node_from_name(part_zones[0], 'Index')[1] == [1,2,3,4]).all()
  if comm.Get_rank() == 2:
    assert (PT.get_node_from_name(part_zones[1], 'Index')[1] == [7,8,9]).all()
    assert PT.get_node_from_name(part_zones[0], 'Index')[1].dtype == pdm_gnum_dtype

@pytest_parallel.mark.parallel(4)
def test_create_part_pl_gnum(comm):
  dist_zone = PT.new_Zone('Zone')
  part_zones = [PT.new_Zone('Zone.P{0}.N0'.format(comm.Get_rank()))]
  distri_ud0 = MT.newGlobalNumbering(parent=part_zones[0])
  if comm.Get_rank() == 0:
    PT.new_ZoneSubRegion("ZSR", point_list=[[1,8,5,2]], loc='Vertex', parent=part_zones[0])
    PT.new_DataArray('Vertex', np.array([22,18,5,13,9,11,6,4], pdm_dtype), parent=distri_ud0)
  elif comm.Get_rank() == 1:
    PT.new_DataArray('Vertex', np.array([5,16,9,17,22], pdm_dtype), parent=distri_ud0)
  elif comm.Get_rank() == 2:
    PT.new_DataArray('Vertex', np.array([13,8,9,6,2], pdm_dtype), parent=distri_ud0)
    part_zones.append(PT.new_Zone('Zone.P2.N1'))
    distri_ud1 = MT.newGlobalNumbering(parent=part_zones[1])
    PT.new_DataArray('Vertex', np.array([4,9,13,1,7,6], pdm_dtype), parent=distri_ud1)
    PT.new_ZoneSubRegion("ZSR", point_list=[[1,3]], loc='Vertex', parent=part_zones[0])
    PT.new_ZoneSubRegion("ZSR", point_list=[[2,4,6]], loc='Vertex', parent=part_zones[1])
  elif comm.Get_rank() == 3:
    part_zones = []

  IPTB.create_part_pl_gnum(dist_zone, part_zones, "ZSR", comm)

  for p_zone in part_zones:
    if PT.get_child_from_name(p_zone, "ZSR") is not None:
      assert PT.get_node_from_path(p_zone, "ZSR/:CGNS#GlobalNumbering/Index") is not None 
  if comm.Get_rank() == 0:
    assert (PT.get_node_from_name(part_zones[0], 'Index')[1] == [7,2,4,6]).all()
    assert PT.get_node_from_name(part_zones[0], 'Index')[1].dtype == pdm_gnum_dtype
  if comm.Get_rank() == 2:
    assert (PT.get_node_from_name(part_zones[0], 'Index')[1] == [5,4]).all()
    assert (PT.get_node_from_name(part_zones[1], 'Index')[1] == [4,1,3]).all()

@pytest_parallel.mark.parallel(3)
def test_create_part_pr_gnum(comm):
  i_rank = comm.Get_rank()
  dist_zone = PT.new_Zone('Zone', type="Structured", size=[[3,2,0], [3,2,0], [3,2,0]])

  if i_rank == 0:
    part_zones = [PT.new_Zone(f'Zone.P{i_rank}.N0', size=[[2,1,0], [3,2,0], [3,2,0]])]
    distri_vtx = np.array([40, 18, 22, 12, 11, 49,  4, 42, 27, 24, 37, 19,  1, 35, 7, 36, 41, 3], pdm_dtype)
    #                                                       ^       ^              ^       ^
    MT.newGlobalNumbering({'Vertex': distri_vtx}, part_zones[0])
    PT.new_ZoneSubRegion("ZSR", point_range=[[1,1], [2,3], [2,3]], loc='Vertex', parent=part_zones[0])
  elif i_rank == 1:
    part_zones = [PT.new_Zone(f'Zone.P{i_rank}.N0', size=[[2,1,0], [3,2,0], [3,2,0]])]
    distri_vtx = np.array([43, 31, 14, 41, 35, 18, 39,  4,  8,  7, 30, 32, 47,  6, 26, 23, 10, 46], pdm_dtype)
    #                           ^       ^               ^       ^
    MT.newGlobalNumbering({'Vertex': distri_vtx}, part_zones[0])
    PT.new_ZoneSubRegion("ZSR", point_range=[[2,2], [1,2], [1,2]], loc='Vertex', parent=part_zones[0])
  elif i_rank == 2:
    part_zones = []

  IPTB.create_part_pr_gnum(dist_zone, part_zones, "ZSR", comm)

  # Two gnum are shared, so index should go from 1 to 6
  for p_zone in part_zones:
    if PT.get_child_from_name(p_zone, "ZSR") is not None:
      assert PT.get_node_from_path(p_zone, "ZSR/:CGNS#GlobalNumbering/Index") is not None 
  if comm.Get_rank() == 0:
    assert (PT.get_node_from_name(part_zones[0], 'Index')[1] == [3,5,2,6]).all()
    assert PT.get_node_from_name(part_zones[0], 'Index')[1].dtype == pdm_gnum_dtype
  if comm.Get_rank() == 1:
    assert (PT.get_node_from_name(part_zones[0], 'Index')[1] == [4,6,1,2]).all()
    assert PT.get_node_from_name(part_zones[0], 'Index')[1].dtype == pdm_gnum_dtype

@pytest_parallel.mark.parallel(1)
@pytest.mark.parametrize("idx_dim", [1, 2])
def test_create_part_pr_gnum_lowdim(idx_dim, comm):

  dist_zone = PT.new_Zone('Zone', type="Structured", size=[[3,2,0], [3,2,0]])
  if idx_dim == 1:
    dist_zone = PT.new_Zone('Zone', type="Structured", size=[[3,2,0]])

  part_zones = [PT.new_Zone(f'Zone.P0.N{i}', size=[[2,1,0], [3,2,0]]) for i in range(2)]
  MT.newGlobalNumbering({'Vertex': np.array([1,2,4,5,7,8], pdm_dtype)}, part_zones[0])
  MT.newGlobalNumbering({'Vertex': np.array([2,3,5,6,8,9], pdm_dtype)}, part_zones[1])
  for p_zone in part_zones:
    if idx_dim == 2:
      PT.new_ZoneSubRegion("ZSR", point_range=[[1,2], [1,1]], loc='Vertex', parent=p_zone)
    elif idx_dim == 1:
      PT.new_ZoneSubRegion("ZSR", point_range=[[1,2]], loc='Vertex', parent=p_zone)

  IPTB.create_part_pr_gnum(dist_zone, part_zones, "ZSR", comm)

  for p_zone in part_zones:
    assert (PT.get_node_from_name(part_zones[0], 'Index')[1] == [1,2]).all()
    assert (PT.get_node_from_name(part_zones[1], 'Index')[1] == [2,3]).all()

@pytest_parallel.mark.parallel(4)
def test_part_pl_to_dist_pl(comm):
  dist_zone = PT.new_Zone('Zone', type='Unstructured')
  dist_zsr = PT.new_ZoneSubRegion("ZSR", loc='Vertex', parent=dist_zone)
  part_zones = [PT.new_Zone('Zone.P{0}.N0'.format(comm.Get_rank()), type='Unstructured')]
  distri_ud0 = MT.newGlobalNumbering(parent=part_zones[0])
  if comm.Get_rank() == 0:
    PT.new_DataArray('Vertex', np.array([22,18,5,13,9,11,6,4], pdm_dtype), parent=distri_ud0)
    zsr = PT.new_ZoneSubRegion("ZSR", point_list=[[1,8,5,2]], loc='Vertex', parent=part_zones[0])
    MT.newGlobalNumbering({'Index' : np.array([7,2,4,6], pdm_dtype)}, zsr)
  elif comm.Get_rank() == 1:
    PT.new_DataArray('Vertex', np.array([5,16,9,17,22], pdm_dtype), parent=distri_ud0)
  elif comm.Get_rank() == 2:
    PT.new_DataArray('Vertex', np.array([13,8,9,6,2], pdm_dtype), parent=distri_ud0)
    part_zones.append(PT.new_Zone('Zone.P2.N1', type='Unstructured'))
    distri_ud1 = MT.newGlobalNumbering(parent=part_zones[1])
    PT.new_DataArray('Vertex', np.array([4,9,13,1,7,6], pdm_dtype), parent=distri_ud1)
    zsr = PT.new_ZoneSubRegion("ZSR", point_list=[[1,3]], loc='Vertex', parent=part_zones[0])
    MT.newGlobalNumbering({'Index' : np.array([5,4], pdm_dtype)}, zsr)
    zsr = PT.new_ZoneSubRegion("ZSR", point_list=[[2,4,6]], loc='Vertex', parent=part_zones[1])
    MT.newGlobalNumbering({'Index' : np.array([4,1,3], pdm_dtype)}, zsr)
  elif comm.Get_rank() == 3:
    part_zones = []

  IPTB.part_pl_to_dist_pl(dist_zone, part_zones, "ZSR", comm)

  dist_pl     = PT.get_node_from_path(dist_zsr, 'PointList')[1]
  dist_distri = PT.get_value(MT.getDistribution(dist_zsr, 'Index'))
  assert dist_distri.dtype == pdm_gnum_dtype

  if comm.Get_rank() == 0:
    assert (dist_distri == [0,2,7]).all()
    assert (dist_pl     == [1,4]  ).all()
  elif comm.Get_rank() == 1:
    assert (dist_distri == [2,4,7]).all()
    assert (dist_pl     == [6,9]  ).all()
  elif comm.Get_rank() == 2:
    assert (dist_distri == [4,6,7]).all()
    assert (dist_pl     == [13,18]).all()
  elif comm.Get_rank() == 3:
    assert (dist_distri == [6,7,7]).all()
    assert (dist_pl     == [22]   ).all()

@pytest_parallel.mark.parallel(1)
def test_part_pl_to_dist_pl_S(comm):
  dist_zone = PT.new_Zone('Zone', type='Structured', size=[[3,2,0],[3,2,0],[3,2,0]])
  dist_zsr = PT.new_ZoneSubRegion("ZSR", loc='KFaceCenter', parent=dist_zone)
  
  part_zone = parse_yaml_cgns.to_node("""
  Zone.P0.N0 Zone_t [[3,2,0], [2,1,0], [3,2,0]]:
    ZoneType ZoneType_t "Structured":
    ZSR ZoneSubRegion_t:
      GridLocation GridLocation_t "KFaceCenter":
      :CGNS#GlobalNumbering UserDefinedData_t:
        Index DataArray_t [2,1]:
      PointList IndexArray_t [[1,2], [1,1], [3,3]]: #(1,1,3) and (2,1,3)
    :CGNS#GlobalNumbering UserDefinedData_t:
      Face DataArray_t [4,5,6,10,11,12, 15,16,17,18,21,22,23,24, 27,28,31,32,35,36]:
  """)

  IPTB.part_pl_to_dist_pl(dist_zone, [part_zone], "ZSR", comm)

  dist_pl = PT.get_node_from_name(dist_zone, 'PointList')
  assert (PT.get_value(dist_pl) == [[2,1],[2,2],[3,3]]).all()

@pytest_parallel.mark.parallel(4)
@pytest.mark.parametrize("allow_mult", [False, True])
def test_part_pr_to_dist_pr(comm, allow_mult):
  yt = """
  Zone Zone_t [[3,2,0], [3,2,0], [5,4,0]]: #Cube 2*2*4 cells
    ZoneType ZoneType_t "Structured":
    ZoneBC ZoneBC_t:
      bc.0 BC_t:
  """
  dist_zone = parse_yaml_cgns.to_node(yt)

  suffix = '.0' if allow_mult else ''
  if comm.Get_rank() == 0:
    yt = f""" # Know the BC on one part
    Zone.P0.N0 Zone_t [[3,2,0], [2,1,0], [3,2,0]]:
      ZoneType ZoneType_t "Structured":
      :CGNS#GlobalNumbering UserDefinedData_t:
        Vertex DataArray_t [1,2,3,4,5,6,10,11,12,13,14,15,19,20,21,22,23,24]: 
      ZoneBC ZoneBC_t:
        bc.0{suffix} BC_t:
          PointRange IndexRange_t [[1,3], [1,2], [1,1]]:
    """
  elif comm.Get_rank() == 1:
    yt = f""" # Know the zone, but has no BC
    Zone.P1.N0 Zone_t [[3,2,0], [3,2,0], [2,1,0]]:
      ZoneType ZoneType_t "Structured":
    Zone.P1.N1 Zone_t [[3,2,0], [3,2,0], [2,1,0]]:
      ZoneType ZoneType_t "Structured":
      ZoneBC ZoneBC_t:
        otherbc{suffix} BC_t:
    """
  elif comm.Get_rank() == 2:
    if allow_mult:
      yt = """ # Know the BC on several partitions
      Zone.P2.N0 Zone_t [[3,2,0], [2,1,0], [3,2,0]]:
        ZoneType ZoneType_t "Structured":
        :CGNS#GlobalNumbering UserDefinedData_t:
          Vertex DataArray_t [4,5,6,7,8,9,13,14,15,16,17,18,22,23,24,25,26,27]:
        ZoneBC ZoneBC_t:
          bc.0 BC_t: # Some other bc (called initially bc)
            PointRange IndexRange_t [[1,1], [1,3], [1,3]]: 
          bc.0.0 BC_t:
            PointRange IndexRange_t [[2,3], [1,2], [1,1]]:
          bc.0.1 BC_t:
            PointRange IndexRange_t [[1,2], [1,2], [1,1]]:
      """
    else:
      yt = """ # Know the BC on several partitions
      Zone.P2.N0 Zone_t [[2,1,0], [2,1,0], [3,2,0]]:
        ZoneType ZoneType_t "Structured":
        :CGNS#GlobalNumbering UserDefinedData_t:
          Vertex DataArray_t [4,5,7,8,10,11,16,17,22,23,25,26]:
        ZoneBC ZoneBC_t:
          bc.0 BC_t:
            PointRange IndexRange_t [[1,2], [1,2], [1,1]]:
      Zone.P2.N1 Zone_t [[2,1,0], [2,1,0], [3,2,0]]:
        ZoneType ZoneType_t "Structured":
        :CGNS#GlobalNumbering UserDefinedData_t:
          Vertex DataArray_t [5,6,8,9,14,15,17,18,23,24,26,27]:
        ZoneBC ZoneBC_t:
          bc.0 BC_t:
            PointRange IndexRange_t [[1,2], [1,2], [1,1]]:
      """
  if comm.Get_rank() == 3:
    part_zones = [] # Not concerned by this zone
  else:
    part_zones = parse_yaml_cgns.to_nodes(yt)

  IPTB.part_pr_to_dist_pr(dist_zone, part_zones, "ZoneBC/bc.0", comm, allow_mult)
  assert np.array_equal(PT.get_node_from_name(dist_zone, 'PointRange')[1], [[1,3], [1,3], [1,1]])

@pytest_parallel.mark.parallel(2)
def test_part_pr_to_dist_pr_2d(comm):
  dist_zone = parse_yaml_cgns.to_node("""
  Zone Zone_t [[4,3,0], [3,2,0]]: #Cube 3*2 cells
    ZoneType ZoneType_t "Structured":
    ZoneBC ZoneBC_t:
      bc BC_t:
  """)

  if comm.Get_rank() == 0:
    part_zones = parse_yaml_cgns.to_nodes(""" # Know the BC on one part
    Zone.P0.N0 Zone_t [[4,3,0], [2,1,0]]:
      ZoneType ZoneType_t "Structured":
      :CGNS#GlobalNumbering UserDefinedData_t:
        Vertex DataArray_t [1,2,3,4,5,6,7,8]:
      ZoneBC ZoneBC_t:
        bc BC_t:
          PointRange IndexRange_t [[4,4], [1,2]]:
    """)
  elif comm.Get_rank() == 1:
    part_zones = parse_yaml_cgns.to_nodes(""" # Know the zone, but has no BC
    Zone.P1.N0 Zone_t [[4,3,0], [2,1,0]]:
      ZoneType ZoneType_t "Structured":
      :CGNS#GlobalNumbering UserDefinedData_t:
        Vertex DataArray_t [5,6,7,8,9,10,11,12]:
      ZoneBC ZoneBC_t:
        bc BC_t:
          PointRange IndexRange_t [[4,4], [1,2]]:
    """)
  IPTB.part_pr_to_dist_pr(dist_zone, part_zones, "ZoneBC/bc", comm)
  assert np.array_equal(PT.get_node_from_name(dist_zone, 'PointRange')[1], [[4,4], [1,3]])

@pytest_parallel.mark.parallel(3)
def test_part_elt_to_dist_elt(comm):
  rank = comm.Get_rank()
  size = comm.Get_size()

  dist_zone = PT.new_Zone('Zone')
  if rank == 0:
    yt = """
Zone.P0.N0 Zone_t:
  Quad Elements_t [7,0]:
    ElementRange IndexRange_t [1, 4]:
    ElementConnectivity DataArray_t:
      I4 : [10,15,17,11, 15,16,12,17, 11,17,13,18, 17,12,14,13]
    :CGNS#GlobalNumbering UserDefinedData_t:
      Element DataArray_t {0} [5,6,7,8]:
      Sections DataArray_t {0} [15,16,17,18]:
  :CGNS#GlobalNumbering UserDefinedData_t:
    Vertex DataArray_t {0} [19,20,21,22,23,24,25,26,27,10,13,15,17,18,11,12,14,16]:
    Cell DataArray_t {0} [5,6,7,4]:
  """.format(dtype)
    expected_ec=[2,5,4,1, 5,8,7,4, 10,13,14,11]
  elif rank == 1:
    yt = ""
    expected_ec=[10,11,14,13,11,12,15,14]
  elif rank == 2:
    yt = """
Zone.P2.N0 Zone_t:
  Quad Elements_t [7,0]:
    ElementRange IndexRange_t [12, 14]:
    ElementConnectivity DataArray_t:
      I4 : [4,5,2,1, 5,6,3,2, 10,11,8,7]
    :CGNS#GlobalNumbering UserDefinedData_t:
      Element DataArray_t {0} [1,2,3]:
      Sections DataArray_t {0} [11,12,13]:
  :CGNS#GlobalNumbering UserDefinedData_t:
    Vertex DataArray_t {0} [1,4,7,2,5,8,11,14,16,10,13,17]:
    Cell DataArray_t {0} [1,3]:
  """.format(dtype)
    expected_ec = [13,14,17,16,14,15,18,17]

  expected_elt_distri_full  = np.array([0, 3, 6, 8])

  pT = parse_yaml_cgns.to_cgns_tree(yt)

  IPTB.part_elt_to_dist_elt(dist_zone, PT.get_all_Zone_t(pT), 'Quad', comm)

  elt = PT.get_node_from_name(dist_zone, 'Quad')
  assert (PT.Element.Range(elt) == [11,18]).all()
  assert (elt[1] == [7,0]).all()
  assert (PT.get_child_from_name(elt, 'ElementConnectivity')[1] == expected_ec).all()
  distri_elt  = PT.get_value(MT.getDistribution(elt, 'Element'))
  assert distri_elt.dtype == pdm_gnum_dtype
  assert (distri_elt  == expected_elt_distri_full [[rank, rank+1, size]]).all()

@pytest_parallel.mark.parallel(3)
def test_part_ngon_to_dist_ngon(comm):
  rank = comm.Get_rank()
  size = comm.Get_size()

  dist_zone = PT.new_Zone('Zone')
  if rank == 0:
    yt = """
Zone.P0.N0 Zone_t:
  Ngon Elements_t [22,0]:
    ElementRange IndexRange_t [1, 20]:
    ElementConnectivity DataArray_t:
      I4 : [10,15,17,11, 15,16,12,17, 11,17,13,18, 17,12,14,13, 1,4,5,2,
            2,5,6,3, 4,7,8,5, 5,8,9,6, 10,11,4,1, 11,18,7,4,
            2,5,17,15, 5,8,13,17, 3,6,12,16, 6,9,14,12, 1,2,15,10,
            2,3,16,15, 11,17,5,4, 17,12,6,5, 18,13,8,7, 13,14,9,8]
    ElementStartOffset DataArray_t:
      I4 : [0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80]
    ParentElements DataArray_t:
      I4 : [[21,0],  [22,0],  [23,0], [24,0], [21,0], [22,0], [23,0],  [24,0],  [21,0], [23,0],
            [21,22], [23,24], [22,0], [24,0], [21,0], [22,0], [21,23], [22,24], [23,0], [24,0]]
    :CGNS#GlobalNumbering UserDefinedData_t:
      Element DataArray_t:
        {0} : [5,6,7,8,9,10,11,12,15,16,19,20,23,24,26,28,30,32,34,36]
  :CGNS#GlobalNumbering UserDefinedData_t:
    Vertex DataArray_t {0} [19,20,21,22,23,24,25,26,27,10,13,15,17,18,11,12,14,16]:
    Cell DataArray_t {0} [5,6,7,8]:
  """.format(dtype)
    expected_ec=[2,5,4,1,3,6,5,2,5,8,7,4,6,9,8,5,10,13,14,11,11,14,15,12,
                 13,16,17,14,14,17,18,15,19,22,23,20,20,23,24,21,22,25,26,23,23,26,27,24]
    expected_pe = np.array([[37,0], [38,0], [39,0], [40,0], [37,41], [38,42],
                            [39,43], [40,44], [41,0], [42,0], [43,0], [44,0]])
  elif rank == 1:
    yt = ""
    expected_pe = np.array([[37,0], [39,0], [41,0], [43,0], [37,38], [39,40],
                            [41,42], [43,44], [38,0], [40,0], [42,0], [44,0]])
    expected_ec=[1,4,13,10,4,7,16,13,10,13,22,19,13,16,25,22,11,14,5,2,14,17,8,5,
                 20,23,14,11,23,26,17,14,12,15,6,3,15,18,9,6,21,24,15,12,24,27,18,15]
  elif rank == 2:
    yt = """
Zone.P2.N0 Zone_t:
  Ngon Elements_t [22,0]:
    ElementRange IndexRange_t [12, 23]:
    ElementConnectivity DataArray_t:
      I4 : [4,5,2,1, 5,6,3,2, 10,11,8,7, 11,9,12,8, 1,2,11,10, 2,3,9,11,
            7,8,5,4, 8,12,6,5, 10,7,4,1, 2,5,8,11, 3,6,12,9]
    ElementStartOffset DataArray_t:
      I4 : [0,4,8,12,16,20,24,28,32,36,40,44]
    ParentElements DataArray_t:
      I4 : [[1,0], [2,0], [1,0], [2,0], [1,0], [2,0], [1,0], [2,0], [1,0], [1,2], [2,0]]
    :CGNS#GlobalNumbering UserDefinedData_t:
      Element DataArray_t {0} [1,3,5,7,13,14,17,18,25,29,33]:
  :CGNS#GlobalNumbering UserDefinedData_t:
    Vertex DataArray_t {0} [1,4,7,2,5,8,11,14,16,10,13,17]:
    Cell DataArray_t {0} [1,3]:
Zone.P2.N1 Zone_t:
  Ngon Elements_t [22,0]:
    ElementRange IndexRange_t [12, 23]:
    ElementConnectivity DataArray_t:
      I4 : [1,2,5,4, 2,3,6,5, 7,9,10,8, 9,11,12,10, 7,4,5,9, 9,5,6,11,
            8,10,2,1, 10,12,3,2, 7,8,1,4, 5,2,10,9, 6,3,12,11]
    ElementStartOffset DataArray_t:
      I4 : [0,4,8,12,16,20,24,28,32,36,40,44]
    ParentElements DataArray_t:
      I4 : [[1,0], [2,0], [1,0], [2,0], [1,0], [2,0], [1,0], [2,0], [1,0], [1,2], [2,0]]
    :CGNS#GlobalNumbering UserDefinedData_t:
      Element DataArray_t {0} [2,4,6,8,17,18,21,22,27,31,35]:
  :CGNS#GlobalNumbering UserDefinedData_t:
    Vertex DataArray_t {0} [3,6,9,2,5,8,11,12,14,15,17,18]:
    Cell DataArray_t {0} [2,4]:
  """.format(dtype)
    expected_pe = np.array([[37,0], [41,0], [38,0], [42,0], [37,39], [41,43],
                            [38,40], [42,44], [39,0], [43,0], [40,0], [44,0]])
    expected_ec=[10,11,2,1,19,20,11,10,11,12,3,2,20,21,12,11,4,5,14,13,13,14,23,22,
                 5,6,15,14,14,15,24,23,7,8,17,16,16,17,26,25,8,9,18,17,17,18,27,26]
  expected_eso = np.array([0,4,8,12,16,20,24,28,32,36,40,44,48]) + 48*comm.Get_rank()
  expected_elt_distri_full  = np.array([0, 12, 24, 36])
  expected_eltc_distri_full = np.array([0, 48, 96, 144])

  pT = parse_yaml_cgns.to_cgns_tree(yt)

  IPTB.part_ngon_to_dist_ngon(dist_zone, PT.get_all_Zone_t(pT), 'Ngon', comm)

  ngon = PT.request_node_from_name(dist_zone, 'Ngon')
  assert (PT.get_child_from_name(ngon, 'ElementStartOffset')[1] == expected_eso).all()
  assert (PT.get_child_from_name(ngon, 'ParentElements')[1] == expected_pe).all()
  assert (PT.get_child_from_name(ngon, 'ElementConnectivity')[1] == expected_ec).all()
  distri_elt  = PT.get_value(MT.getDistribution(ngon, 'Element'))
  distri_eltc = PT.get_value(MT.getDistribution(ngon, 'ElementConnectivity'))
  assert distri_elt.dtype == distri_eltc.dtype == pdm_gnum_dtype
  assert (distri_elt  == expected_elt_distri_full [[rank, rank+1, size]]).all()
  assert (distri_eltc == expected_eltc_distri_full[[rank, rank+1, size]]).all()

@pytest_parallel.mark.parallel(3)
def test_part_nface_to_dist_nface(comm):
  rank = comm.Get_rank()
  size = comm.Get_size()

  dist_zone = PT.new_Zone('Zone')
  if rank == 0:
    yt = """
Zone.P0.N0 Zone_t:
  Ngon Elements_t [22,0]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Element DataArray_t:
        {0} : [5,6,7,8,9,10,11,12,15,16,19,20,23,24,26,28,30,32,34,36]
  NFace Elements_t [23,0]:
    ElementConnectivity DataArray_t:
      I4 : [1,5,9,11,15,17,2,6,-11,13,16,18,3,7,10,12,-17,19,4,8,-12,14,-18,20]
    ElementStartOffset DataArray_t [0,6,12,18,24]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Element DataArray_t {0} [5,6,7,8]:
  :CGNS#GlobalNumbering UserDefinedData_t:
    Vertex DataArray_t {0} [19,20,21,22,23,24,25,26,27,10,13,15,17,18,11,12,14,16]:
    Cell DataArray_t {0} [5,6,7,8]:
  """.format(dtype)
    expected_ec  = [1,5,13,17,25,29,2,6,17,21,27,31,3,7,14,18,-29,33]
    expected_eso = [0,6,12,18]
  elif rank == 1:
    yt = ""
    expected_ec  = [4,8,18,22,-31,35,5,9,15,19,26,30,6,10,-19,23,28,32]
    expected_eso = [18,24,30,36]
  elif rank == 2:
    yt = """
Zone.P2.N0 Zone_t:
  Ngon Elements_t [22,0]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Element DataArray_t {0} [1,3,5,7,13,14,17,18,25,29,33]:
  NFace Elements_t [23,0]:
    ElementConnectivity DataArray_t I4 [1,3,5,7,9,10,2,4,6,8,-10,11]:
    ElementStartOffset DataArray_t [0,6,12]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Element DataArray_t {0} [1,3]:
  :CGNS#GlobalNumbering UserDefinedData_t:
    Vertex DataArray_t {0} [1,4,7,2,5,8,11,14,16,10,13,17]:
    Cell DataArray_t {0} [1,3]:
Zone.P2.N1 Zone_t:
  Ngon Elements_t [22,0]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Element DataArray_t {0} [2,4,6,8,17,18,21,22,27,31,35]:
  NFace Elements_t [23,0]:
    ElementConnectivity DataArray_t I4 [1,3,5,7,9,10,2,4,6,8,-10,11]:
    ElementStartOffset DataArray_t [0,6,12]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Element DataArray_t {0} [2,4]:
  :CGNS#GlobalNumbering UserDefinedData_t:
    Vertex DataArray_t {0} [3,6,9,2,5,8,11,12,14,15,17,18]:
    Cell DataArray_t {0} [2,4]:
  """.format(dtype)
    expected_ec  = [7,11,16,20,-30,34,8,12,-20,24,-32,36]
    expected_eso = [36,42,48]

  expected_elt_distri_full  = np.array([0,3,6,8])
  expected_eltc_distri_full = np.array([0,18,36,48])

  pT = parse_yaml_cgns.to_cgns_tree(yt)

  IPTB.part_nface_to_dist_nface(dist_zone, PT.get_all_Zone_t(pT), 'NFace', 'Ngon', comm)

  nface = PT.get_node_from_name(dist_zone, 'NFace')
  assert nface is not None
  assert (PT.get_child_from_name(nface, 'ElementStartOffset')[1] == expected_eso).all()
  assert (PT.get_child_from_name(nface, 'ElementConnectivity')[1] == expected_ec).all()
  distri_elt  = PT.get_value(MT.getDistribution(nface, 'Element'))
  distri_eltc = PT.get_value(MT.getDistribution(nface, 'ElementConnectivity'))
  assert distri_elt.dtype == distri_eltc.dtype == pdm_gnum_dtype
  assert (distri_elt  == expected_elt_distri_full [[rank, rank+1, size]]).all()
  assert (distri_eltc == expected_eltc_distri_full[[rank, rank+1, size]]).all()
