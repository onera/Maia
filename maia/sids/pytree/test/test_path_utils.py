import os
from maia.utils        import parse_yaml_cgns
import maia.sids.pytree as PT

from maia.sids.pytree import path_utils as PU

dir_path = os.path.dirname(os.path.realpath(__file__))

def test_path_head():
  assert PU.path_head('some/path/to/node', 2) == 'some/path'
  assert PU.path_head('some/path/to/node', 4) == 'some/path/to/node'
  assert PU.path_head('some/path/to/node', 0) == ''
  assert PU.path_head('some/path/to/node', -2) == 'some/path'
  assert PU.path_head('some/path/to/node') == 'some/path/to'

def test_path_tail():
  assert PU.path_tail('some/path/to/node', 2) == 'to/node'
  assert PU.path_tail('some/path/to/node', 0) == 'some/path/to/node'
  assert PU.path_tail('some/path/to/node', -1) == 'node'

def test_update_path_elt():
  path = 'some/path/to/node'
  assert PU.update_path_elt(path, 3, lambda n : 'something') == 'some/path/to/something'
  assert PU.update_path_elt(path, -1, lambda n : n.upper()) == 'some/path/to/NODE'
  assert PU.update_path_elt(path, 1, lambda n : 'crazy' + n) == 'some/crazypath/to/node'

def test_predicates_to_paths():
  with open(os.path.join(dir_path, "minimal_bc_tree.yaml"), 'r') as yt:
    tree = parse_yaml_cgns.to_cgns_tree(yt)

  paths = PU.predicates_to_paths(tree, ["Base", "Zone_t", "ZBC*", lambda n: int(n[0][-1]) >= 2 and int(n[0][-1]) <= 4])
  assert paths == ['Base/ZoneI/ZBCA/bc2', 'Base/ZoneI/ZBCB/bc3', 'Base/ZoneI/ZBCB/bc4']
  assert PU.predicates_to_paths(tree, 'Nope/*') == []

def test_concretize_paths():
  with open(os.path.join(dir_path, "minimal_bc_tree.yaml"), 'r') as yt:
    tree = parse_yaml_cgns.to_cgns_tree(yt)
  paths = PU.concretize_paths(tree, ["Base/Zone*/ZBCA", "Base/ZoneI/*", "Nope/Zone/*"], ['CGNSBase_t', 'Zone_t', 'ZoneBC_t'])
  assert paths == ['Base/ZoneI/ZBCA', 'Base/ZoneI/ZBCB']
  assert PU.concretize_paths(tree, ["Nope/Zone/*"], ['CGNSBase_t', 'Zone_t', 'ZoneBC_t']) == []

def test_paths_to_tree():
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

    paths = ['first/B/1/', 'second/B/2', 'first/B/4', 'first/A/1']
    path_tree = PU.paths_to_tree(paths, "Ro0t")
    assert path_tree == expected

    assert PU.paths_to_tree([], root_name='Root') == ['Root', None, [], None]
