from maia.utils.yaml   import parse_yaml_cgns
from maia.pytree.sids import adjust
from maia.pytree      import walk
from maia.pytree      import nodes_attr as NA

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
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  adjust.enforceDonorAsPath(tree)
  assert NA.get_value(walk.get_node_from_name(tree, "match1")) == "Base/ZoneA"
  assert NA.get_value(walk.get_node_from_name(tree, "match2")) == "Base/ZoneA"
  assert NA.get_value(walk.get_node_from_name(tree, "match3")) == "Base/ZoneB"
  assert NA.get_value(walk.get_node_from_name(tree, "match4")) == "Base/ZoneA"


