from maia.pytree.graph.build import depth_first_build

from maia.pytree.graph.io_graph import VALUE, rooted_tree_example, depth_first_build_io_tree


def indent(s):
  indented_s = ''
  lines = s.split('\n')
  for line in lines[:-1]:
    indented_s += '  ' + line + '\n'
  return indented_s

def test_indent():
  s = 'A\n  A0\n  A1\nB\nC\n'
  expected_s = '  A\n    A0\n    A1\n  B\n  C\n'
  assert indent(s) == expected_s


def test_depth_first_build():
  #   Reminder:
  #         1               lvl 3
  #      /  |  \
  #     |   |    3          lvl 2
  #     |   | /  |  \
  #     2\_ | 8  |   \      lvl 1
  #   /  \ \| |  |    \
  #   |  |  \ |  |    \
  #  4    7  \9  10   11    lvl 0

  g = rooted_tree_example()

  def indented_tree_ctor(node, sub_strings):
    return str(node[VALUE]) + '\n' + ''.join([indent(s) for s in sub_strings])

  s = depth_first_build(g, indented_tree_ctor)

  expected_s = \
    ('1\n'
     '  2\n'
     '    4\n'
     '    7\n'
     '    9\n'
     '  3\n'
     '    8\n'
     '      9\n'
     '    10\n'
     '    11\n'
     '  9\n'
    )

  assert s == expected_s


def test_depth_first_build_io_tree():
  #   Reminder:
  #         1               lvl 3
  #      /  |  \
  #     |   |    3          lvl 2
  #     |   | /  |  \
  #     2\_ | 8  |   \      lvl 1
  #   /  \ \| |  |    \
  #   |  |  \ |  |    \
  #  4    7  \9  10   11    lvl 0

  g = rooted_tree_example()

  io_tree = depth_first_build_io_tree(g)

  expected_io_tree = [
    [4 , [3 ], []       ], # 0
    [7 , [3 ], []       ], # 1
    [9 , [3 ], []       ], # 2
    [2 , [10], [0, 1, 2]], # 3
    [9 , [5 ], []       ], # 4
    [8 , [8 ], [4]      ], # 5
    [10, [8 ], []       ], # 6
    [11, [8 ], []       ], # 7
    [3 , [10], [5, 6, 7]], # 8
    [9 , [10], []       ], # 9
    [1 , []  , [3, 8, 9]], # 10
  ]
  # expected_io_tree is:
  #             1
  #          /  |  \
  #         /   \    \
  #        /     \     9
  #       /      3
  #      /     / | \
  #     2      8 |  \
  #   / | \    | |   \
  # 4   7   9  9 10   11

  assert io_tree == expected_io_tree
