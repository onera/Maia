from maia.pytree import yaml
from maia.pytree import walk as W
from maia.pytree import node as N

from maia.pytree.sids import adjust

def test_enforceDonorAsPath():
  yt = """
  Base CGNSBase_t:
    ZoneA Zone_t:
      ZGC ZoneGridConnectivity_t:
        match1 GridConnectivity_t "ZoneA":
        match2 GridConnectivity_t "Base/ZoneA":
        match3 GridConnectivity1to1_t "ZoneB":
    ZoneB Zone_t:
      ZGC1 ZoneGridConnectivity_t:
        match4 GridConnectivity1to1_t "ZoneA":
  """
  tree = yaml.parse_yaml_cgns.to_cgns_tree(yt)
  adjust.enforceDonorAsPath(tree)
  assert N.get_value(W.get_node_from_name(tree, "match1")) == "Base/ZoneA"
  assert N.get_value(W.get_node_from_name(tree, "match2")) == "Base/ZoneA"
  assert N.get_value(W.get_node_from_name(tree, "match3")) == "Base/ZoneB"
  assert N.get_value(W.get_node_from_name(tree, "match4")) == "Base/ZoneA"

