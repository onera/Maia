import numpy as np
import Converter.Internal as I
from   maia.utils        import parse_yaml_cgns
from maia.partitioning.split_S import part_zone as splitS

def test_collect_S_bnd_per_dir():
  yt = """
Zone Zone_t:
  ZBC ZoneBC_t:
    bc1 BC_t:
      PointRange IndexRange_t [[1,1],[1,8],[1,6]]:
      GridLocation GridLocation_t "IFaceCenter":
    bc2 BC_t:
      PointRange IndexRange_t [[1,9],[1,1],[1,7]]:
      GridLocation GridLocation_t "Vertex":
    bc8 BC_t:
      PointRange IndexRange_t [[9,17],[1,1],[1,7]]:
      GridLocation GridLocation_t "Vertex":
    bc3 BC_t:
      PointRange IndexRange_t [[1,16],[9,9],[1,6]]:
      GridLocation GridLocation_t "JFaceCenter":
    bc4 BC_t:
      PointRange IndexRange_t [[1,16],[1,8],[1,1]]:
      GridLocation GridLocation_t "CellCenter":
    bc5 BC_t:
      PointRange IndexRange_t [[1,16],[1,8],[6,6]]:
      GridLocation GridLocation_t "CellCenter":
    bc6 BC_t:
      PointRange IndexRange_t [[17,17],[1,3],[1,7]]:
    bc7 BC_t:
      PointRange IndexRange_t [[17,17],[3,8],[5,6]]:
      GridLocation GridLocation_t "IFaceCenter":
  ZGC ZoneGridConnectivity_t:
    gc1 GridConnectivity1to1_t:
      PointRange IndexRange_t [[17,17],[3,9],[1,5]]:
"""
  dist_tree = parse_yaml_cgns.to_complete_pytree(yt)
  zone      = I.getZones(dist_tree)[0]

  out = splitS.collect_S_bnd_per_dir(zone)
  assert out["xmin"] == ['bc1']
  assert out["ymin"] == ['bc2', 'bc8']
  assert out["zmin"] == ['bc4']
  assert out["xmax"] == ['bc6', 'bc7', 'gc1']
  assert out["ymax"] == ['bc3']
  assert out["zmax"] == ['bc5']

def test_intersect_pr():
  print("coucou")
  assert splitS.intersect_pr(np.array([[7,11],[1,5]]), np.array([[12,16],[1,5]])) is None
  assert (splitS.intersect_pr(np.array([[7,11],[1,5]]), np.array([[7,11],[1,5]])) \
      == np.array([[7,11],[1,5]])).all()
  assert (splitS.intersect_pr(np.array([[1,5],[1,3]]), np.array([[1,4],[3,6]])) \
      == np.array([[1,4],[3,3]])).all()
