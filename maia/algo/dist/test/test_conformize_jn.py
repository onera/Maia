import pytest
from pytest_mpi_check._decorator import mark_mpi_test

import Converter.Internal as I

from maia.utils.yaml import parse_yaml_cgns
from maia.factory    import full_to_dist as F2D
from maia.factory    import dcube_generator  as DCG

from maia.algo.dist  import conformize_jn as CJN

@mark_mpi_test(2)
@pytest.mark.parametrize("from_loc", ["Vertex", "FaceCenter"])
def test_conformize_jn_pair(sub_comm, from_loc):
  dist_tree = DCG.dcube_generate(5, 1., [0., 0., 0.], sub_comm)
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
    PointList IndexArray_t {pl}:
    PointListDonor IndexArray_t {pld}:
  matchB GridConnectivity_t "zone":
    GridConnectivityType GridConnectivityType_t "Abutting1to1":
    GridLocation GridLocation_t "{loc}":
    PointList IndexArray_t {pld}:
    PointListDonor IndexArray_t {pl}:
  """
  gcs = [F2D.distribute_pl_node(gc, sub_comm) for gc in parse_yaml_cgns.to_nodes(yt)]
  zone = I.getZones(dist_tree)[0]
  I.createChild(zone, "ZGC", "ZoneGridConnectivity_t", children=gcs)

  CJN.conformize_jn_pair(dist_tree, ['Base/zone/ZGC/matchA', 'Base/zone/ZGC/matchB'], sub_comm)

  if sub_comm.rank == 0:
    start, end = 0, 5
  elif sub_comm.rank == 1:
    start, end = 125-5, 125

  assert (I.getNodeFromName(zone, "CoordinateX")[1][start:end] == 0.5).all()
  assert (I.getNodeFromName(zone, "CoordinateY")[1][start:end] == 0.5).all()
  assert (I.getNodeFromName(zone, "CoordinateZ")[1][start:end] == 0.5).all()

