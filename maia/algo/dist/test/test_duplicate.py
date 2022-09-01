import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np
import os

import Converter.Internal as I
import maia.pytree as PT

from maia.io          import file_to_dist_tree
from maia.utils       import test_utils as TU

from maia.algo.dist   import duplicate

###############################################################################
@mark_mpi_test([1,3])
def test_duplicate_from_periodic_jns(sub_comm):

  def get_perio_values(perio_node):
    return [I.getVal(PT.get_child_from_name(perio_node, name)) for name in ["RotationCenter", "RotationAngle", "Translation"]]
  def get_coords_values(zone_node):
    return [I.getVal(PT.get_node_from_name(zone_node, f"Coordinate{c}")) for c in ['X', 'Y', 'Z']]

  yaml_path = os.path.join(TU.sample_mesh_dir, 'quarter_crown_square_8.yaml')
  dist_tree = file_to_dist_tree(yaml_path, sub_comm)
  
  match_perio_by_trans_a = 'Base/Zone/ZoneGridConnectivity/MatchTranslationA'
  match_perio_by_trans_b = 'Base/Zone/ZoneGridConnectivity/MatchTranslationB'
  jn_paths_for_dupl = [[match_perio_by_trans_a],[match_perio_by_trans_b]]
  
  zone_basename = I.getName(I.getZones(dist_tree)[0])
  zone_paths = ['Base/' + I.getName(zone) for zone in I.getZones(dist_tree)]
  
  duplicate.duplicate_from_periodic_jns(dist_tree, zone_paths, jn_paths_for_dupl, 1, sub_comm)
  
  assert len(I.getZones(dist_tree)) == 2
  
  zone0, zone1 = I.getZones(dist_tree)
  assert (I.getName(zone0) == zone_basename+".D0") and (I.getName(zone1) == zone_basename+".D1") 

  coord0 = get_coords_values(zone0)
  coord1 = get_coords_values(zone1)
  assert np.allclose(coord0[0], coord1[0]   )
  assert np.allclose(coord0[1], coord1[1]   )
  assert np.allclose(coord0[2], coord1[2]+2.)
  
  zgc0 = PT.get_child_from_label(zone0, "ZoneGridConnectivity_t")
  zgc1 = PT.get_child_from_label(zone1, "ZoneGridConnectivity_t")
  for gc0 in PT.get_children_from_label(zgc0, "GridConnectivity_t"):
    gc0_name = I.getName(gc0)
    gc1 = PT.get_child_from_name_and_label(zgc1, gc0_name, "GridConnectivity_t")
    oppname0 = I.getValue(PT.get_child_from_name(gc0, "GridConnectivityDonorName"))
    oppname1 = I.getValue(PT.get_child_from_name(gc1, "GridConnectivityDonorName"))
    assert oppname0 == oppname1
    if gc0_name in ["MatchRotationA","MatchRotationB"]: #Joins by rotation
      assert I.getValue(gc0) == zone_basename+".D0"
      assert I.getValue(gc1) == zone_basename+".D1"
      gcp0 = PT.get_child_from_label(gc0, "GridConnectivityProperty_t")
      gcp1 = PT.get_child_from_label(gc1, "GridConnectivityProperty_t")
      perio0 = PT.get_child_from_label(gcp0, "Periodic_t")
      perio1 = PT.get_child_from_label(gcp1, "Periodic_t")
      rotation_center0, rotation_angle0, translation0 = get_perio_values(perio0)
      rotation_center1, rotation_angle1, translation1 = get_perio_values(perio1)
      assert (rotation_center0 == rotation_center1).all()
      assert (rotation_angle0  == rotation_angle1).all()
      assert (translation0     == translation1).all()
    elif gc0_name in ["MatchTranslationA","MatchTranslationB"]: #Joins by translation
      assert I.getValue(gc0) == zone_basename+".D1"
      assert I.getValue(gc1) == zone_basename+".D0"
      if gc0_name == "MatchTranslationA": #Join0 => perio*2 and join1 => not perio
        gcp0 = PT.get_child_from_label(gc0, "GridConnectivityProperty_t")
        gcp1 = PT.get_child_from_label(gc1, "GridConnectivityProperty_t")
        assert gcp1 is None
        perio0 = PT.get_child_from_label(gcp0, "Periodic_t")
        rotation_center0, rotation_angle0, translation0 = get_perio_values(perio0)
        assert (rotation_center0 == np.zeros(3)).all()
        assert (rotation_angle0  == np.zeros(3)).all()
        assert (translation0     == np.array([0.0, 0.0, -2.0])*2).all()
      else: #Join0 => not perio and join1 => perio*2
        gcp0 = PT.get_child_from_label(gc0, "GridConnectivityProperty_t")
        gcp1 = PT.get_child_from_label(gc1, "GridConnectivityProperty_t")
        assert gcp0 is None 
        perio1 = PT.get_child_from_label(gcp1,"Periodic_t")
        rotation_center1, rotation_angle1, translation1 = get_perio_values(perio1)
        assert (rotation_center1 == np.zeros(3)).all()
        assert (rotation_angle1  == np.zeros(3)).all()
        assert (translation1     == np.array([0.0, 0.0, 2.0])*2).all()
    else:
      assert False 

###############################################################################

###############################################################################
@mark_mpi_test(2)
def test_duplicate_zones_from_periodic_join_by_rotation_to_360(sub_comm):
  def get_coords_values(zone_node):
    return [I.getVal(PT.get_node_from_name(zone_node, f"Coordinate{c}")) for c in ['X', 'Y', 'Z']]

  yaml_path = os.path.join(TU.sample_mesh_dir, 'quarter_crown_square_8.yaml')
  dist_tree = file_to_dist_tree(yaml_path,sub_comm)
  
  match_perio_by_rot_a = 'Base/Zone/ZoneGridConnectivity/MatchRotationA'
  match_perio_by_rot_b = 'Base/Zone/ZoneGridConnectivity/MatchRotationB'
  jn_paths_for_dupl = [[match_perio_by_rot_a],[match_perio_by_rot_b]]
  
  zone_basename = I.getName(I.getZones(dist_tree)[0])
  zone_paths = ['Base/' + I.getName(zone) for zone in I.getZones(dist_tree)]
  
  duplicate.duplicate_from_rotation_jns_to_360(dist_tree, zone_paths, 
      jn_paths_for_dupl, sub_comm, conformize=True)
  
  zones = I.getZones(dist_tree)
  assert len(zones) == 4
  assert all([zone[0] == f"{zone_basename}.D{i}" for i, zone in enumerate(zones)])

  coord0, coord1, coord2, coord3 = [get_coords_values(zone) for zone in zones]
  assert np.allclose(coord0[0], -coord2[0])
  assert np.allclose(coord0[1], -coord2[1])
  assert np.allclose(coord0[2],  coord2[2])
  assert np.allclose(coord1[0], -coord3[0])
  assert np.allclose(coord1[1], -coord3[1])
  assert np.allclose(coord1[2],  coord3[2])
  assert np.allclose(coord0[0], -coord1[1])
  assert np.allclose(coord0[1],  coord1[0])
  assert np.allclose(coord0[2],  coord1[2])
  assert np.allclose(coord0[0],  coord3[1])
  assert np.allclose(coord0[1], -coord3[0])
  assert np.allclose(coord0[2],  coord3[2])
  
  zgc0 = PT.get_child_from_label(zones[0], "ZoneGridConnectivity_t")
  zgc1 = PT.get_child_from_label(zones[1], "ZoneGridConnectivity_t")
  zgc2 = PT.get_child_from_label(zones[2], "ZoneGridConnectivity_t")
  zgc3 = PT.get_child_from_label(zones[3], "ZoneGridConnectivity_t")
  for gc0 in PT.get_children_from_label(zgc0, "GridConnectivity_t"):
    gc0_name = I.getName(gc0)
    gc1 = PT.get_child_from_name_and_label(zgc1, gc0_name, "GridConnectivity_t")
    gc2 = PT.get_child_from_name_and_label(zgc2, gc0_name, "GridConnectivity_t")
    gc3 = PT.get_child_from_name_and_label(zgc3, gc0_name, "GridConnectivity_t")
    oppname0 = I.getValue(PT.get_child_from_name(gc0, "GridConnectivityDonorName"))
    oppname1 = I.getValue(PT.get_child_from_name(gc1, "GridConnectivityDonorName"))
    oppname2 = I.getValue(PT.get_child_from_name(gc1, "GridConnectivityDonorName"))
    oppname3 = I.getValue(PT.get_child_from_name(gc1, "GridConnectivityDonorName"))
    assert oppname0 == oppname1 == oppname2 == oppname3
    if gc0_name in ["MatchRotationA","MatchRotationB"]: #Joins by rotation => not perio
      if gc0_name == "MatchRotationA":
        assert I.getValue(gc0) == zone_basename+".D3"
        assert I.getValue(gc1) == zone_basename+".D0"
        assert I.getValue(gc2) == zone_basename+".D1"
        assert I.getValue(gc3) == zone_basename+".D2"
      else:
        assert I.getValue(gc0) == zone_basename+".D1"
        assert I.getValue(gc1) == zone_basename+".D2"
        assert I.getValue(gc2) == zone_basename+".D3"
        assert I.getValue(gc3) == zone_basename+".D0"
      assert PT.get_child_from_label(gc0, "GridConnectivityProperty_t") is None
      assert PT.get_child_from_label(gc1, "GridConnectivityProperty_t") is None
      assert PT.get_child_from_label(gc2, "GridConnectivityProperty_t") is None
      assert PT.get_child_from_label(gc3, "GridConnectivityProperty_t") is None
    elif gc0_name in ["MatchTranslationA","MatchTranslationB"]: #Joins by translation => no change execpt value
      assert I.getValue(gc0) == zone_basename+".D0"
      assert I.getValue(gc1) == zone_basename+".D1"
      assert I.getValue(gc2) == zone_basename+".D2"
      assert I.getValue(gc3) == zone_basename+".D3"
      pass
    else:
      assert False
