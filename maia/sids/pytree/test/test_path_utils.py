import os
from maia.utils        import parse_yaml_cgns
import maia.sids.pytree as PT

from maia.sids.pytree import path_utils as PU

dir_path = os.path.dirname(os.path.realpath(__file__))

def test_predicates_to_pathes():
  with open(os.path.join(dir_path, "minimal_bc_tree.yaml"), 'r') as yt:
    tree = parse_yaml_cgns.to_cgns_tree(yt)

  pathes = PU.predicates_to_pathes(tree, ["Base", "Zone_t", "ZBC*", lambda n: int(n[0][-1]) >= 2 and int(n[0][-1]) <= 4])
  assert pathes == ['Base/ZoneI/ZBCA/bc2', 'Base/ZoneI/ZBCB/bc3', 'Base/ZoneI/ZBCB/bc4']
  assert PU.predicates_to_pathes(tree, 'Nope/*') == []

def test_concretise_pathes():
  with open(os.path.join(dir_path, "minimal_bc_tree.yaml"), 'r') as yt:
    tree = parse_yaml_cgns.to_cgns_tree(yt)
  pathes = PU.concretise_pathes(tree, ["Base/Zone*/ZBCA", "Base/ZoneI/*", "Nope/Zone/*"], ['CGNSBase_t', 'Zone_t', 'ZoneBC_t'])
  assert pathes == ['Base/ZoneI/ZBCA', 'Base/ZoneI/ZBCB']
  assert PU.concretise_pathes(tree, ["Nope/Zone/*"], ['CGNSBase_t', 'Zone_t', 'ZoneBC_t']) == []

def test_pathes_to_tree():
    yt = """
    Ro0t None:
      first None:
        B None:
          1 None:
          4 None:
        A None:
          1 None:
      second None:
        B None:
          2 None:
    """
    expected = parse_yaml_cgns.to_node(yt)
    #Fix None
    for node in PT.iter_nodes_from_predicate(expected, lambda n: True, explore='deep'):
      node[3] = None

    pathes = ['first/B/1/', 'second/B/2', 'first/B/4', 'first/A/1']
    path_tree = PU.pathes_to_tree(pathes, "Ro0t")
    assert path_tree == expected

    assert PU.pathes_to_tree([], root_name='Root') == ['Root', None, [], None]
