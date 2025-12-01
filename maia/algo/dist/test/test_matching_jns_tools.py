from mpi4py import MPI
import pytest
import pytest_parallel
import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

import maia
from maia.algo.dist import matching_jns_tools as MJT

pl_val  = lambda n : PT.get_np_value(PT.find_child_from_name(n, 'PointList'))
pld_val = lambda n : PT.get_np_value(PT.find_child_from_name(n, 'PointListDonor'))

def test_gc_is_reference():
  pr  = np.array([[1,1], [1,10], [1,10]], order='F')
  prd = np.array([[20,10], [1,10], [5,5]], order='F')
  gc = PT.new_GridConnectivity1to1(donor_name='Base/ZoneB', point_range=pr, point_range_donor=prd)
  assert MJT.gc_is_reference(gc, 'Base/ZoneA') == True
  gc = PT.new_GridConnectivity1to1(donor_name='Base/ZoneA', point_range=pr, point_range_donor=prd)
  assert MJT.gc_is_reference(gc, 'Base/ZoneB') == False
  gc = PT.new_GridConnectivity1to1(donor_name='Aase/ZoneA', point_range=pr, point_range_donor=prd)
  assert MJT.gc_is_reference(gc, 'Base/ZoneA') == False
  gc = PT.new_GridConnectivity1to1(donor_name='Base/ZoneA', point_range=pr, point_range_donor=prd)
  assert MJT.gc_is_reference(gc, 'Base/ZoneA') == True
  with pytest.raises(ValueError):
    gc = PT.new_GridConnectivity1to1(donor_name='Base/ZoneA', point_range=pr, point_range_donor=pr)
    MJT.gc_is_reference(gc, 'Base/ZoneA')

class Test_compare_pointrange():
  def test_ok(self):
    jn1 = PT.new_GridConnectivity1to1(point_range      =[[17,17],[3,9],[1,5]], point_range_donor=[[7,1],[9,9],[5,1]])
    jn2 = PT.new_GridConnectivity1to1(point_range_donor=[[17,17],[3,9],[1,5]], point_range      =[[7,1],[9,9],[5,1]])
    assert(MJT._compare_pointrange(jn1, jn2) == True)
    jn1 = PT.new_GridConnectivity1to1(point_range      =[[17,17],[3,9],[1,5]], point_range_donor=[[7,1],[9,9],[5,1]])
    jn2 = PT.new_GridConnectivity1to1(point_range_donor=[[17,17],[3,9],[1,5]], point_range      =[[1,7],[9,9],[1,5]])
    assert(MJT._compare_pointrange(jn1, jn2) == True)
  def test_ko(self):
    jn1 = PT.new_GridConnectivity1to1(point_range      =[[17,17],[3,9],[1,5]], point_range_donor=[[7,1],[9,9],[5,1]])
    jn2 = PT.new_GridConnectivity1to1(point_range_donor=[[17,17],[3,7],[1,5]], point_range      =[[1,5],[9,9],[1,5]])
    assert(MJT._compare_pointrange(jn1, jn2) == False)
    jn1 = PT.new_GridConnectivity1to1(point_range      =[[17,17],[3,9]],       point_range_donor=[[7,1],[9,9]])
    jn2 = PT.new_GridConnectivity1to1(point_range_donor=[[17,17],[3,9],[1,5]], point_range      =[[7,1],[9,9],[5,1]])
    assert(MJT._compare_pointrange(jn1, jn2) == False)

@pytest_parallel.mark.parallel(1)
class Test_compare_pointlist():
  def test_ok(self, comm):
    jn1 = PT.new_GridConnectivity(type='Abutting1to1', point_list      =[[12,14,16,18]], point_list_donor=[[9,7,5,3]])
    jn2 = PT.new_GridConnectivity(type='Abutting1to1', point_list_donor=[[12,14,16,18]], point_list      =[[9,7,5,3]])
    MT.new_Distribution({'Index' : np.array([0,4,4])}, jn1)
    MT.new_Distribution({'Index' : np.array([0,4,4])}, jn2)
    assert(MJT._compare_pointlist(jn1, jn2, comm) == True)
  def test_ko(self, comm):
    jn1 = PT.new_GridConnectivity(type='Abutting1to1', point_list      =[[12,14,16,18]], point_list_donor=[[9,7,5,3]])
    jn2 = PT.new_GridConnectivity(type='Abutting1to1', point_list_donor=[[12,14,16,18]], point_list      =[[3,9,5,7]])
    MT.new_Distribution({'Index' : np.array([0,4,4])}, jn1)
    MT.new_Distribution({'Index' : np.array([0,4,4])}, jn2)
    assert(MJT._compare_pointlist(jn1, jn2, comm) == False)
  def test_empty(self, comm):
    jn1 = PT.new_GridConnectivity(type='Abutting1to1', point_list      =np.empty((1,0), np.int32), point_list_donor=np.empty((1,0), np.int32))
    jn2 = PT.new_GridConnectivity(type='Abutting1to1', point_list_donor=np.empty((1,0), np.int32), point_list      =np.empty((1,0), np.int32))
    MT.new_Distribution({'Index' : np.array([0,0,0])}, jn1)
    MT.new_Distribution({'Index' : np.array([0,0,0])}, jn2)
    assert(MJT._compare_pointlist(jn1, jn2, comm) == True)

@pytest_parallel.mark.parallel(2)
def test_compare_pl_non_sym_ok(comm):
  if comm.rank == 0:
    jn1 = PT.new_GridConnectivity(type='Abutting1to1', point_list      =[[12]], point_list_donor=[[9]])
    jn2 = PT.new_GridConnectivity(type='Abutting1to1', point_list_donor=[[18]], point_list      =[[3]])
    MT.new_Distribution({'Index' : np.array([0,1,4])}, jn1)
    MT.new_Distribution({'Index' : np.array([0,1,4])}, jn2)
  else:
    jn1 = PT.new_GridConnectivity(type='Abutting1to1', point_list      =[[14,16,18]], point_list_donor=[[7,5,3]])
    jn2 = PT.new_GridConnectivity(type='Abutting1to1', point_list_donor=[[16,14,12]], point_list      =[[5,7,9]])
    MT.new_Distribution({'Index' : np.array([1,4,4])}, jn1)
    MT.new_Distribution({'Index' : np.array([1,4,4])}, jn2)

  assert MJT._compare_pointlist(jn1, jn2, comm) == True

@pytest_parallel.mark.parallel(2)
def test_compare_pl_non_sym_ko(comm):
  if comm.rank == 0:
    jn1 = PT.new_GridConnectivity(type='Abutting1to1', point_list      =[[12]], point_list_donor=[[9]])
    jn2 = PT.new_GridConnectivity(type='Abutting1to1', point_list_donor=[[18]], point_list      =[[3]])
    MT.new_Distribution({'Index' : np.array([0,1,4])}, jn1)
    MT.new_Distribution({'Index' : np.array([0,1,4])}, jn2)
  else:
    jn1 = PT.new_GridConnectivity(type='Abutting1to1', point_list      =[[14,16,18]], point_list_donor=[[7,5,3]])
    jn2 = PT.new_GridConnectivity(type='Abutting1to1', point_list_donor=[[16,14,120]], point_list      =[[5,7,9]])
    MT.new_Distribution({'Index' : np.array([1,4,4])}, jn1)
    MT.new_Distribution({'Index' : np.array([1,4,4])}, jn2)

  assert comm.allreduce(MJT._compare_pointlist(jn1, jn2, comm), MPI.LAND) == False

  if comm.rank == 0:
    jn1 = PT.new_GridConnectivity(type='Abutting1to1', point_list      =[[12]], point_list_donor=[[9]])
    jn2 = PT.new_GridConnectivity(type='Abutting1to1', point_list_donor=[[18]], point_list      =[[3]])
    MT.new_Distribution({'Index' : np.array([0,1,4])}, jn1)
    MT.new_Distribution({'Index' : np.array([0,1,4])}, jn2)
  else:
    #                                                                   # Permutation => False           x x 
    jn1 = PT.new_GridConnectivity(type='Abutting1to1', point_list      =[[14,16,18]], point_list_donor=[[5,7,3]])
    jn2 = PT.new_GridConnectivity(type='Abutting1to1', point_list_donor=[[16,14,12]], point_list      =[[5,7,9]])
    MT.new_Distribution({'Index' : np.array([1,4,4])}, jn1)
    MT.new_Distribution({'Index' : np.array([1,4,4])}, jn2)

  assert comm.allreduce(MJT._compare_pointlist(jn1, jn2, comm), MPI.LAND) == False


@pytest_parallel.mark.parallel([1,3])
def test_add_joins_donor_name(comm):
  yt = """
Base0 CGNSBase_t [3,3]:
  ZoneA Zone_t [[27,8,0]]:
    ZoneType ZoneType_t "Unstructured":
    ZGC ZoneGridConnectivity_t:
      matchAB GridConnectivity1to1_t "ZoneB":
        GridLocation GridLocation_t "FaceCenter":
        PointList      IndexArray_t [[1,4,7,10]]:    # HERE
        PointListDonor IndexArray_t [[13,16,7,10]]:  # HERE
  ZoneB Zone_t [[27,8,0]]:
    ZoneType ZoneType_t "Unstructured":
    ZGC ZoneGridConnectivity_t:
      matchBA GridConnectivity_t "ZoneA":
        GridLocation GridLocation_t "FaceCenter":
        PointList      IndexArray_t [[13,16,7,10]]:  # HERE
        PointListDonor IndexArray_t [[1,4,7,10]]:    # HERE
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
      matchBC1 GridConnectivity_t "Base1/ZoneC":
        GridLocation GridLocation_t "FaceCenter":
        PointList      IndexArray_t [[32,34]]:
        PointListDonor IndexArray_t [[1,3]]:
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
      matchBC2 GridConnectivity_t "Base1/ZoneC":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[33,35]]:
        PointListDonor IndexArray_t [[2,4]]:
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
Base1 CGNSBase_t [3,3]:
  ZoneC Zone_t [[18,4,0]]:
    ZoneType ZoneType_t "Unstructured":
    ZGC ZoneGridConnectivity_t:
      matchCB2 GridConnectivity1to1_t "Base0/ZoneB":
        GridLocation GridLocation_t "FaceCenter":
        PointList      IndexArray_t [[2,4]]:         # HERE
        PointListDonor IndexArray_t [[33,35]]:       # HERE
      matchCB1 GridConnectivity1to1_t "Base0/ZoneB":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[1,3]]:
        PointListDonor IndexArray_t [[32,34]]:
"""
  full_tree = PT.yaml.to_cgns_tree(yt)
  dist_tree = maia.factory.full_to_dist_tree(full_tree, comm)

  MJT.add_joins_donor_name(dist_tree, comm)

  expected_donor_names = ['matchBA', 'matchAB', 'matchCB1', 'matchCB2', 'matchBC2', 'matchBC1']
  for i, jn in enumerate(PT.iter_nodes_from_predicate(dist_tree, PT.pred.IS_GC)):
    assert PT.get_value(PT.get_child_from_name(jn, 'GridConnectivityDonorName')) == expected_donor_names[i]

@pytest_parallel.mark.parallel(1)
def test_force(comm):
  yt = """
Base0 CGNSBase_t:
  ZoneA Zone_t:
    ZGC ZoneGridConnectivity_t:
      matchAB GridConnectivity_t "ZoneB":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        PointList IndexArray_t [[1,4,7,10]]:
        PointListDonor IndexArray_t [[13,16,7,10]]:
        GridConnectivityDonorName Descriptor_t "WrongOldValue":
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t [0,4,4]:
  ZoneB Zone_t:
    ZGC ZoneGridConnectivity_t:
      matchBA GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        PointList IndexArray_t [[13,16,7,10]]:
        PointListDonor IndexArray_t [[1,4,7,10]]:
        GridConnectivityDonorName Descriptor_t "WrongOldValue":
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t [0,4,4]:
"""
  dist_tree = PT.yaml.to_cgns_tree(yt)
  jn_donor_path = 'Base0/ZoneA/ZGC/matchAB/GridConnectivityDonorName'
  assert PT.get_value(PT.get_node_from_path(dist_tree, jn_donor_path)) == 'WrongOldValue'
  MJT.add_joins_donor_name(dist_tree, comm)
  assert PT.get_value(PT.get_node_from_path(dist_tree, jn_donor_path)) == 'WrongOldValue'
  MJT.add_joins_donor_name(dist_tree, comm, force=True)
  assert PT.get_value(PT.get_node_from_path(dist_tree, jn_donor_path)) == 'matchBA'

@pytest_parallel.mark.parallel(1)
def test_some_computed(comm):
  yt = """
Base0 CGNSBase_t:
  ZoneA Zone_t:
    ZGC ZoneGridConnectivity_t:
      matchAB.0 GridConnectivity_t "ZoneB":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        PointList IndexArray_t [[1,4,7,10]]:
        PointListDonor IndexArray_t [[13,16,7,10]]:
        GridConnectivityDonorName Descriptor_t "matchBA.0":
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t [0,4,4]:
      matchAB.1 GridConnectivity_t "ZoneB":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        PointList IndexArray_t [[7,10]]:
        PointListDonor IndexArray_t [[7,10]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t [0,2,2]:
  ZoneB Zone_t:
    ZGC ZoneGridConnectivity_t:
      matchBA.0 GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        PointList IndexArray_t [[13,16,7,10]]:
        PointListDonor IndexArray_t [[1,4,7,10]]:
        GridConnectivityDonorName Descriptor_t "matchAB.0":
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t [0,4,4]:
      matchBA.1 GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        PointList IndexArray_t [[7,10]]:
        PointListDonor IndexArray_t [[7,10]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t [0,2,2]:
"""
  dist_tree = PT.yaml.to_cgns_tree(yt)
  MJT.add_joins_donor_name(dist_tree, comm)

class Test_gcdonorname_utils:
  dt = """
Base CGNSBase_t:
  ZoneA Zone_t:
    ZGC ZoneGridConnectivity_t:
      perio1 GridConnectivity_t 'ZoneA':
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridConnectivityDonorName Descriptor_t 'perio2':
        PointList IndexArray_t [[1,3]]:
      perio2 GridConnectivity_t 'Base/ZoneA':
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridConnectivityDonorName Descriptor_t 'perio1':
        PointList IndexArray_t [[2,4]]:
      match1 GridConnectivity_t 'ZoneB':
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridConnectivityDonorName Descriptor_t 'match2':
        PointList IndexArray_t [[10,100]]:
        GridLocation GridLocation_t "FaceCenter":
  ZoneB Zone_t:
    ZGC ZoneGridConnectivity_t:
      match2 GridConnectivity_t 'ZoneA':
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridConnectivityDonorName Descriptor_t 'match1':
        PointList IndexArray_t [[-100,-10]]:
        GridLocation GridLocation_t "FaceCenter":
  """
  dist_tree = PT.yaml.to_cgns_tree(dt)

  def test_get_jn_donor_path(self):
    assert MJT.get_jn_donor_path(self.dist_tree, 'Base/ZoneA/ZGC/perio2') == 'Base/ZoneA/ZGC/perio1'
    assert MJT.get_jn_donor_path(self.dist_tree, 'Base/ZoneA/ZGC/match1') == 'Base/ZoneB/ZGC/match2'

  def test_update_jn_name(self):
    dist_tree = PT.deep_copy(self.dist_tree)
    ini_gc = PT.get_node_from_path(dist_tree, 'Base/ZoneA/ZGC/perio2')
    MJT.update_jn_name(dist_tree, 'Base/ZoneA/ZGC/perio2', 'PERIO2')
    assert ini_gc[0] == 'PERIO2'
    assert PT.get_value(PT.get_node_from_path(dist_tree, 'Base/ZoneA/ZGC/perio1/GridConnectivityDonorName')) == 'PERIO2'

  def test_get_matching_jns(self):
    pathes = MJT.get_matching_jns(self.dist_tree)
    assert pathes[0] == ('Base/ZoneA/ZGC/perio1', 'Base/ZoneA/ZGC/perio2')
    assert pathes[1] == ('Base/ZoneA/ZGC/match1', 'Base/ZoneB/ZGC/match2')
    pathes = MJT.get_matching_jns(self.dist_tree, PT.pred.has_location('Vertex'))
    assert len(pathes) == 1
    assert pathes[0] == ('Base/ZoneA/ZGC/perio1', 'Base/ZoneA/ZGC/perio2')

  def test_match_jn_from_ordinals(self):
    dist_tree = PT.deep_copy(self.dist_tree)
    MJT.copy_donor_subset(dist_tree)
    expected_pl_opp = [[2,4], [1,3], [-100,-10], [10,100]]
    for i, jn in enumerate(PT.iter_nodes_from_label(dist_tree, 'GridConnectivity_t')):
      assert (PT.get_child_from_name(jn, 'PointListDonor')[1] == expected_pl_opp[i]).all()

  def test_store_interfaces_ids(self):
    dist_tree = PT.deep_copy(self.dist_tree)
    MJT.store_interfaces_ids(dist_tree)
    expected_id = ['1','1','2','2']
    expected_pos = ['0','1','0','1']
    for i, jn in enumerate(PT.iter_nodes_from_label(dist_tree, 'GridConnectivity_t')):
      assert PT.get_value(PT.get_child_from_name(jn, 'DistInterfaceId')) == expected_id[i]
      assert PT.get_value(PT.get_child_from_name(jn, 'DistInterfaceOrd')) == expected_pos[i]


def test_clear_interfaces_ids():
  yt = """
Base0 CGNSBase_t:
  ZoneA Zone_t:
    ZGC ZoneGridConnectivity_t:
      matchAB GridConnectivity_t "ZoneB":
        DistInterfaceId Descriptor_t "1":
  ZoneB Zone_t:
    ZGC ZoneGridConnectivity_t:
      matchBA GridConnectivity_t "ZoneA":
        PointList IndexArray_t [[13,16,7,10]]:
"""
  dist_tree = PT.yaml.to_cgns_tree(yt)
  MJT.clear_interface_ids(dist_tree)
  assert PT.get_node_from_name(dist_tree, 'DistInterfaceId')  is None
  assert PT.get_node_from_name(dist_tree, 'DistInterfaceOrd') is None

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize('has_zsr', [0, 1, 2])
def test_enforce_symmetric_jns(has_zsr, comm):
  yt = """
Base CGNSBase_t:
  ZoneU Zone_t [[3, 2, 0]]:
    ZoneType ZoneType_t "Unstructured":
    ZGC ZoneGridConnectivity_t:
      perio1 GridConnectivity_t 'ZoneU':
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        PointList IndexArray_t [[7,5,3]]:
        PointListDonor IndexArray_t [[11,12,13]]:
      perio2 GridConnectivity_t 'Base/ZoneU':
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        PointList IndexArray_t [[11,13,12]]:
        PointListDonor IndexArray_t [[7,3,5]]:
      matchA GridConnectivity_t 'ZoneS':
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        PointList IndexArray_t [[5,9,7,8,12,4]]:
        PointListDonor IndexArray_t [[1,1,1,1,1,1], [5,1,2,4,6,3]]:
  ZoneS Zone_t [[3,2,0], [6,5,0]]:
    ZoneType ZoneType_t "Structured":
    ZGC ZoneGridConnectivity_t:
      matchB GridConnectivity_t 'ZoneU':
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        PointList IndexArray_t [[1,1,1,1,1,1], [3,5,6,1,2,4]]:
        PointListDonor IndexArray_t [[4,5,12,9,7,8]]:
"""
  full_tree = PT.yaml.to_cgns_tree(yt)
  if has_zsr > 0:
    # ZSR to perio1
    PT.new_ZoneSubRegion('GCZSR1', gc_name='perio1', parent=PT.get_all_Zone_t(full_tree)[0])
  if has_zsr > 1:
    # ZSR to both perio1 and perio2 (should raise)
    PT.new_ZoneSubRegion('GCZSR2', gc_name='perio2', parent=PT.get_all_Zone_t(full_tree)[0])
  dist_tree = maia.factory.full_to_dist_tree(full_tree, comm)

  if has_zsr == 2:
    with pytest.raises(RuntimeError):
      MJT.enforce_symmetric_jns(dist_tree, comm)

  else:
    MJT.enforce_symmetric_jns(dist_tree, comm)

    ftree = maia.factory.dist_to_full_tree(dist_tree, comm, 0)
    if comm.rank == 0:
      perio1 = PT.find_node_from_name(ftree, 'perio1')
      perio2 = PT.find_node_from_name(ftree, 'perio2')
      expt_pl = [[7,5,3]] if has_zsr == 0 else [[7,3,5]]
      expt_pld = [[11,12,13]] if has_zsr == 0 else [[11,13,12]]
      assert np.array_equal(v:=pl_val(perio1), pld_val(perio2)) and (v == expt_pl).all()
      assert np.array_equal(v:=pld_val(perio1), pl_val(perio2)) and (v == expt_pld).all()

      matchA = PT.find_node_from_name(ftree, 'matchA')
      matchB = PT.find_node_from_name(ftree, 'matchB')
      assert np.array_equal(v:=pl_val(matchA), pld_val(matchB)) and (v == [[4,5,12,9,7,8]]).all()
      assert np.array_equal(v:=pld_val(matchA), pl_val(matchB)) and (v == [[1,1,1,1,1,1], [3,5,6,1,2,4]]).all()