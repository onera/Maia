import os

import maia.pytree as PT
from maia.pytree.utils import path_utils as PU

dir_path = PT.__path__[0]

def test_path_len():
  assert PU.path_len('some/path') == 2
  assert PU.path_len('some/path/to/node') == 4
  assert PU.path_len('') == 0

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

def test_concretize_paths():
  with open(os.path.join(dir_path, "test", "minimal_tree.yaml"), 'r') as yt:
    tree = PT.yaml.to_cgns_tree(yt)
  paths = PU.concretize_paths(tree, ["Base/Zone*/ZGCA", "Base/ZoneI/*", "Nope/Zone/*"], ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t'])
  assert paths == ['Base/ZoneI/ZGCA', 'Base/ZoneI/ZGCB']
  assert PU.concretize_paths(tree, ["Nope/Zone/*"], ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t']) == []

def test_paths_to_tree():
    # Nodes are not really BC, but the yaml loader required an existing label
    yt = """
    Ro0t BC_t:
      first BC_t:
        B BC_t:
          1 BC_t:
          4 BC_t:
        A BC_t:
          1 BC_t:
      second BC_t:
        B BC_t:
          2 BC_t:
    """
    expected = PT.yaml.to_node(yt)
    #Fix None
    for node in PT.iter_nodes_from_predicate(expected, lambda n: True, explore='deep'):
      node[3] = None

    paths = ['first/B/1/', 'second/B/2', 'first/B/4', 'first/A/1']
    path_tree = PU.paths_to_tree(paths, "Ro0t")
    assert path_tree == expected

    assert PU.paths_to_tree([], root_name='Root') == ['Root', None, [], None]
