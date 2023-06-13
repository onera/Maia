import pytest
import pytest_parallel
import numpy as np
import os

import maia.pytree as PT

from maia.io          import file_to_dist_tree
from maia.utils       import test_utils as TU

from maia.algo.dist   import duplicate

###############################################################################
@pytest_parallel.mark.parallel([1,3])
def test_duplicate_from_periodic_jns(comm):

  yaml_path = os.path.join(TU.sample_mesh_dir, 'quarter_crown_square_8.yaml')
  dist_tree = file_to_dist_tree(yaml_path, comm)
  
  match_perio_by_trans_a = 'Base/Zone/ZoneGridConnectivity/MatchTranslationA'
  match_perio_by_trans_b = 'Base/Zone/ZoneGridConnectivity/MatchTranslationB'
  jn_paths_for_dupl = [[match_perio_by_trans_a],[match_perio_by_trans_b]]
  
  zone_basename = PT.get_name(PT.get_all_Zone_t(dist_tree)[0])
  zone_paths = ['Base/' + PT.get_name(zone) for zone in PT.get_all_Zone_t(dist_tree)]
  
  duplicate.duplicate_from_periodic_jns(dist_tree, zone_paths, jn_paths_for_dupl, 1, comm)
  
  assert len(PT.get_all_Zone_t(dist_tree)) == 2
  
  zone0, zone1 = PT.get_all_Zone_t(dist_tree)
  assert (PT.get_name(zone0) == zone_basename+".D0") and (PT.get_name(zone1) == zone_basename+".D1") 

  coord0 = PT.Zone.coordinates(zone0)
  coord1 = PT.Zone.coordinates(zone1)
  assert np.allclose(coord0[0], coord1[0]   )
  assert np.allclose(coord0[1], coord1[1]   )
  assert np.allclose(coord0[2], coord1[2]+2.)
  
  zgc0 = PT.get_child_from_label(zone0, "ZoneGridConnectivity_t")
  zgc1 = PT.get_child_from_label(zone1, "ZoneGridConnectivity_t")
  for gc0 in PT.get_children_from_label(zgc0, "GridConnectivity_t"):
    gc0_name = PT.get_name(gc0)
    gc1 = PT.get_child_from_name_and_label(zgc1, gc0_name, "GridConnectivity_t")
    oppname0 = PT.get_value(PT.get_child_from_name(gc0, "GridConnectivityDonorName"))
    oppname1 = PT.get_value(PT.get_child_from_name(gc1, "GridConnectivityDonorName"))
    assert oppname0 == oppname1
    if gc0_name in ["MatchRotationA","MatchRotationB"]: #Joins by rotation
      assert PT.get_value(gc0) == zone_basename+".D0"
      assert PT.get_value(gc1) == zone_basename+".D1"
      rotation_center0, rotation_angle0, translation0 = PT.GridConnectivity.get_perio_values(gc0)
      rotation_center1, rotation_angle1, translation1 = PT.GridConnectivity.get_perio_values(gc1)
      assert (rotation_center0 == rotation_center1).all()
      assert (rotation_angle0  == rotation_angle1).all()
      assert (translation0     == translation1).all()
    elif gc0_name in ["MatchTranslationA","MatchTranslationB"]: #Joins by translation
      assert PT.get_value(gc0) == zone_basename+".D1"
      assert PT.get_value(gc1) == zone_basename+".D0"
      if gc0_name == "MatchTranslationA": #Join0 => perio*2 and join1 => not perio
        gcp1 = PT.get_child_from_label(gc1, "GridConnectivityProperty_t")
        assert gcp1 is None
        rotation_center0, rotation_angle0, translation0 = PT.GridConnectivity.get_perio_values(gc0)
        assert (rotation_center0 == np.zeros(3)).all()
        assert (rotation_angle0  == np.zeros(3)).all()
        assert (translation0     == np.array([0.0, 0.0, -2.0])*2).all()
      else: #Join0 => not perio and join1 => perio*2
        gcp0 = PT.get_child_from_label(gc0, "GridConnectivityProperty_t")
        assert gcp0 is None 
        rotation_center1, rotation_angle1, translation1 = PT.GridConnectivity.get_perio_values(gc1)
        assert (rotation_center1 == np.zeros(3)).all()
        assert (rotation_angle1  == np.zeros(3)).all()
        assert (translation1     == np.array([0.0, 0.0, 2.0])*2).all()
    else:
      assert False 

###############################################################################

###############################################################################
@pytest_parallel.mark.parallel(2)
def test_duplicate_zones_from_periodic_join_by_rotation_to_360(comm):

  yaml_path = os.path.join(TU.sample_mesh_dir, 'quarter_crown_square_8.yaml')
  dist_tree = file_to_dist_tree(yaml_path,comm)
  
  match_perio_by_rot_a = 'Base/Zone/ZoneGridConnectivity/MatchRotationA'
  match_perio_by_rot_b = 'Base/Zone/ZoneGridConnectivity/MatchRotationB'
  jn_paths_for_dupl = [[match_perio_by_rot_a],[match_perio_by_rot_b]]
  
  zone_basename = PT.get_name(PT.get_all_Zone_t(dist_tree)[0])
  zone_paths = ['Base/' + PT.get_name(zone) for zone in PT.get_all_Zone_t(dist_tree)]
  
  duplicate.duplicate_from_rotation_jns_to_360(dist_tree, zone_paths, 
      jn_paths_for_dupl, comm, conformize=True)
  
  zones = PT.get_all_Zone_t(dist_tree)
  assert len(zones) == 4
  assert all([zone[0] == f"{zone_basename}.D{i}" for i, zone in enumerate(zones)])

  coord0, coord1, coord2, coord3 = [PT.Zone.coordinates(zone) for zone in zones]
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
    gc0_name = PT.get_name(gc0)
    gc1 = PT.get_child_from_name_and_label(zgc1, gc0_name, "GridConnectivity_t")
    gc2 = PT.get_child_from_name_and_label(zgc2, gc0_name, "GridConnectivity_t")
    gc3 = PT.get_child_from_name_and_label(zgc3, gc0_name, "GridConnectivity_t")
    oppname0 = PT.get_value(PT.get_child_from_name(gc0, "GridConnectivityDonorName"))
    oppname1 = PT.get_value(PT.get_child_from_name(gc1, "GridConnectivityDonorName"))
    oppname2 = PT.get_value(PT.get_child_from_name(gc1, "GridConnectivityDonorName"))
    oppname3 = PT.get_value(PT.get_child_from_name(gc1, "GridConnectivityDonorName"))
    assert oppname0 == oppname1 == oppname2 == oppname3
    if gc0_name in ["MatchRotationA","MatchRotationB"]: #Joins by rotation => not perio
      if gc0_name == "MatchRotationA":
        assert PT.get_value(gc0) == zone_basename+".D3"
        assert PT.get_value(gc1) == zone_basename+".D0"
        assert PT.get_value(gc2) == zone_basename+".D1"
        assert PT.get_value(gc3) == zone_basename+".D2"
      else:
        assert PT.get_value(gc0) == zone_basename+".D1"
        assert PT.get_value(gc1) == zone_basename+".D2"
        assert PT.get_value(gc2) == zone_basename+".D3"
        assert PT.get_value(gc3) == zone_basename+".D0"
      assert PT.get_child_from_label(gc0, "GridConnectivityProperty_t") is None
      assert PT.get_child_from_label(gc1, "GridConnectivityProperty_t") is None
      assert PT.get_child_from_label(gc2, "GridConnectivityProperty_t") is None
      assert PT.get_child_from_label(gc3, "GridConnectivityProperty_t") is None
    elif gc0_name in ["MatchTranslationA","MatchTranslationB"]: #Joins by translation => no change execpt value
      assert PT.get_value(gc0) == zone_basename+".D0"
      assert PT.get_value(gc1) == zone_basename+".D1"
      assert PT.get_value(gc2) == zone_basename+".D2"
      assert PT.get_value(gc3) == zone_basename+".D3"
      pass
    else:
      assert False
