from maia.pytree.graph.build import depth_first_build_trees, depth_first_build
from maia.pytree.graph.algo import step

from maia.pytree.graph.f_graph import VALUE, rooted_f_graph_example, multiply_rooted_f_graph_example


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
  #         1
  #      /  |  \
  #     |   |    3
  #     |   | /  |  \
  #     2\_ | 8  |   \
  #   /  \ \| |  |    \
  #   |  |  \ |  |    \
  #  4    7  \9  10   11

  g = rooted_f_graph_example()

  def indented_tree_ctor(node, sub_strings):
    return str(node[VALUE]) + '\n' + ''.join([indent(s) for s in sub_strings])

  # 1. Test with a complete scan
  s = depth_first_build(g, indented_tree_ctor)

  expected_s = \
    ('1\n'
     '  2\n'
     '    4\n'
     '    7\n'
     '    9\n'
     '  9\n'
     '  3\n'
     '    8\n'
     '      9\n'
     '    10\n'
     '    11\n'
    )

  assert s == expected_s

  # 2. Exit early
  def step_over_2_and_out_10(node):
    if node[VALUE]==2:
      return step.over
    if node[VALUE]==10:
      return step.out
    return step.into

  s = depth_first_build(g, indented_tree_ctor, pre=step_over_2_and_out_10)

  expected_s = \
    ('1\n'
     '  2\n'
     '  9\n'
     '  3\n'
     '    8\n'
     '      9\n'
     '    10\n'
    )

  assert s == expected_s


def test_depth_first_build_trees():
  #   Reminder:
  #    root  root
  #     |     |
  #     v     v
  #
  #     2\_   8
  #   /  \ \  |
  #   |  |  \ |
  #  4    7  \9
  g = multiply_rooted_f_graph_example()

  def indented_tree_ctor(node, sub_strings):
    return str(node[VALUE]) + '\n' + ''.join([indent(s) for s in sub_strings])
  s = depth_first_build_trees(g, indented_tree_ctor)

  assert s == [('2\n'
                '  4\n'
                '  7\n'
                '  9\n'
               ),
               ('8\n'
                '  9\n'
               )
              ]
