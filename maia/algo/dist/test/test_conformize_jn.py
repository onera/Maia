import pytest
import pytest_parallel
import numpy as np

import maia.pytree as PT

import maia
from maia.pytree.yaml import parse_yaml_cgns
from maia.factory     import full_to_dist as F2D
from maia.factory     import dcube_generator  as DCG

from maia.algo.dist  import conformize_jn as CJN

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("from_loc", ["Vertex", "FaceCenter"])
def test_conformize_jn_pair(comm, from_loc):
  dist_tree = DCG.dcube_generate(5, 1., [0., 0., 0.], comm)
  isize = 'I8' if maia.npy_pdm_gnum_dtype == np.int64 else 'I4'
  # Add a fake join
  if from_loc == "Vertex":
    loc = 'Vertex'
    pl = [[1,2,3,4,5]]
    pld = [[125,124,123,122,121]]
  if from_loc == "FaceCenter":
    loc = "FaceCenter"
    pl = [[1,2,3,4]]
    pld = [[80,79,78,77]]
  yt = f"""
  matchA GridConnectivity_t "zone":
    GridConnectivityType GridConnectivityType_t "Abutting1to1":
    GridLocation GridLocation_t "{loc}":
    PointList IndexArray_t {isize} {pl}:
    PointListDonor IndexArray_t {isize} {pld}:
  matchB GridConnectivity_t "zone":
    GridConnectivityType GridConnectivityType_t "Abutting1to1":
    GridLocation GridLocation_t "{loc}":
    PointList IndexArray_t {isize} {pld}:
    PointListDonor IndexArray_t {isize} {pl}:
  """
  gcs = [F2D.distribute_pl_node(gc, comm) for gc in parse_yaml_cgns.to_nodes(yt)]
  zone = PT.get_all_Zone_t(dist_tree)[0]
  PT.new_child(zone, "ZGC", "ZoneGridConnectivity_t", children=gcs)

  CJN.conformize_jn_pair(dist_tree, ['Base/zone/ZGC/matchA', 'Base/zone/ZGC/matchB'], comm)

  if comm.rank == 0:
    start, end = 0, 5
  elif comm.rank == 1:
    start, end = 125-5, 125

  assert (PT.get_node_from_name(zone, "CoordinateX")[1][start:end] == 0.5).all()
  assert (PT.get_node_from_name(zone, "CoordinateY")[1][start:end] == 0.5).all()
  assert (PT.get_node_from_name(zone, "CoordinateZ")[1][start:end] == 0.5).all()

