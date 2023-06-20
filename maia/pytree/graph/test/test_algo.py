from maia.pytree.graph.algo import step, depth_first_search

from maia.pytree.graph.f_graph import rooted_f_graph_example, VALUE


class visitor_for_testing_depth_first_scan:
  def __init__(self):
    self.s = ''

  def pre(self, x):
    self.s += '[pre ] ' + str(x[VALUE]) + '\n'
  def post(self, x):
    self.s += '[post] ' + str(x[VALUE]) + '\n'
  def up(self, below, above):
    self.s += '[up  ] ' + str(below[VALUE]) + ' -> ' + str(above[VALUE]) + '\n'
  def down(self, above, below):
    self.s += '[down] ' + str(above[VALUE]) + ' -> ' + str(below[VALUE]) + '\n'

  def accumulation_string(self):
    return self.s


def test_depth_first_scan():
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
  v = visitor_for_testing_depth_first_scan()
  depth_first_search(g,v)

  expected_s = \
    '[pre ] 1\n' \
    '[down] 1 -> 2\n' \
    '[pre ] 2\n' \
    '[down] 2 -> 4\n' \
    '[pre ] 4\n' \
    '[post] 4\n' \
    '[up  ] 4 -> 2\n' \
    '[down] 2 -> 7\n' \
    '[pre ] 7\n' \
    '[post] 7\n' \
    '[up  ] 7 -> 2\n' \
    '[down] 2 -> 9\n' \
    '[pre ] 9\n' \
    '[post] 9\n' \
    '[up  ] 9 -> 2\n' \
    '[post] 2\n' \
    '[up  ] 2 -> 1\n' \
    '[down] 1 -> 9\n' \
    '[pre ] 9\n' \
    '[post] 9\n' \
    '[up  ] 9 -> 1\n' \
    '[down] 1 -> 3\n' \
    '[pre ] 3\n' \
    '[down] 3 -> 8\n' \
    '[pre ] 8\n' \
    '[down] 8 -> 9\n' \
    '[pre ] 9\n' \
    '[post] 9\n' \
    '[up  ] 9 -> 8\n' \
    '[post] 8\n' \
    '[up  ] 8 -> 3\n' \
    '[down] 3 -> 10\n' \
    '[pre ] 10\n' \
    '[post] 10\n' \
    '[up  ] 10 -> 3\n' \
    '[down] 3 -> 11\n' \
    '[pre ] 11\n' \
    '[post] 11\n' \
    '[up  ] 11 -> 3\n' \
    '[post] 3\n' \
    '[up  ] 3 -> 1\n' \
    '[post] 1\n' \

  assert v.accumulation_string() == expected_s


class visitor_for_testing_depth_first_find(visitor_for_testing_depth_first_scan):
  def pre(self, x) -> bool:
    visitor_for_testing_depth_first_scan.pre(self,x)
    if x[VALUE] == 3:
      return step.out
    else:
      return step.into

def test_depth_first_find():
  #  Reminder:
  #         1               lvl 3
  #      /  |  \
  #     |   |    3          lvl 2
  #     |   | /  |  \
  #     2\_ | 8  |   \      lvl 1
  #   /  \ \| |  |    \
  #   |  |  \ |  |    \
  #  4    7  \9  10   11    lvl 0

  g = rooted_f_graph_example()
  v = visitor_for_testing_depth_first_find()
  found = depth_first_search(g,v)

  expected_s = \
    '[pre ] 1\n' \
    '[down] 1 -> 2\n' \
    '[pre ] 2\n' \
    '[down] 2 -> 4\n' \
    '[pre ] 4\n' \
    '[post] 4\n' \
    '[up  ] 4 -> 2\n' \
    '[down] 2 -> 7\n' \
    '[pre ] 7\n' \
    '[post] 7\n' \
    '[up  ] 7 -> 2\n' \
    '[down] 2 -> 9\n' \
    '[pre ] 9\n' \
    '[post] 9\n' \
    '[up  ] 9 -> 2\n' \
    '[post] 2\n' \
    '[up  ] 2 -> 1\n' \
    '[down] 1 -> 9\n' \
    '[pre ] 9\n' \
    '[post] 9\n' \
    '[up  ] 9 -> 1\n' \
    '[down] 1 -> 3\n' \
    '[pre ] 3\n' \
    '[post] 3\n' \
    '[up  ] 3 -> 1\n' \
    '[post] 1\n'

  assert v.accumulation_string() == expected_s
  assert found[VALUE] == 3


class visitor_for_testing_depth_first_prune(visitor_for_testing_depth_first_scan):
  def pre(self, x) -> bool:
    visitor_for_testing_depth_first_scan.pre(self,x)
    if x[VALUE] == 2:
      return step.over
    else:
      return step.into

def test_depth_first_prune():
  #  Reminder:
  #         1               lvl 3
  #      /  |  \
  #     |   |    3          lvl 2
  #     |   | /  |  \
  #     2\_ | 8  |   \      lvl 1
  #   /  \ \| |  |    \
  #   |  |  \ |  |    \
  #  4    7  \9  10   11    lvl 0

  g = rooted_f_graph_example()
  v = visitor_for_testing_depth_first_prune()
  depth_first_search(g,v)

  expected_s = \
    '[pre ] 1\n' \
    '[down] 1 -> 2\n' \
    '[pre ] 2\n' \
    '[post] 2\n' \
    '[up  ] 2 -> 1\n' \
    '[down] 1 -> 9\n' \
    '[pre ] 9\n' \
    '[post] 9\n' \
    '[up  ] 9 -> 1\n' \
    '[down] 1 -> 3\n' \
    '[pre ] 3\n' \
    '[down] 3 -> 8\n' \
    '[pre ] 8\n' \
    '[down] 8 -> 9\n' \
    '[pre ] 9\n' \
    '[post] 9\n' \
    '[up  ] 9 -> 8\n' \
    '[post] 8\n' \
    '[up  ] 8 -> 3\n' \
    '[down] 3 -> 10\n' \
    '[pre ] 10\n' \
    '[post] 10\n' \
    '[up  ] 10 -> 3\n' \
    '[down] 3 -> 11\n' \
    '[pre ] 11\n' \
    '[post] 11\n' \
    '[up  ] 11 -> 3\n' \
    '[post] 3\n' \
    '[up  ] 3 -> 1\n' \
    '[post] 1\n'

  assert v.accumulation_string() == expected_s


class visitor_for_testing_dfs(visitor_for_testing_depth_first_scan):
  def pre(self,x):
    visitor_for_testing_depth_first_scan.pre(self,x)
    if x[VALUE] == 8: return step.out
    if x[VALUE] == 2: return step.over
    else: return step.into

def test_depth_first_search():
  #  Reminder:
  #         1               lvl 3
  #      /  |  \
  #     |   |    3          lvl 2
  #     |   | /  |  \
  #     2\_ | 8  |   \      lvl 1
  #   /  \ \| |  |    \
  #   |  |  \ |  |    \
  #  4    7  \9  10   11    lvl 0
  #
  g = rooted_f_graph_example()
  v = visitor_for_testing_dfs()
  found = depth_first_search(g,v)

  expected_s = \
    '[pre ] 1\n' \
    '[down] 1 -> 2\n' \
    '[pre ] 2\n' \
    '[post] 2\n' \
    '[up  ] 2 -> 1\n' \
    '[down] 1 -> 9\n' \
    '[pre ] 9\n' \
    '[post] 9\n' \
    '[up  ] 9 -> 1\n' \
    '[down] 1 -> 3\n' \
    '[pre ] 3\n' \
    '[down] 3 -> 8\n' \
    '[pre ] 8\n' \
    '[post] 8\n' \
    '[up  ] 8 -> 3\n' \
    '[post] 3\n' \
    '[up  ] 3 -> 1\n' \
    '[post] 1\n'

  assert v.accumulation_string() == expected_s
  assert found[VALUE] == 8


class modifying_visitor_for_testing_dfs(visitor_for_testing_depth_first_scan):
  def pre(self, x):
    if   x[VALUE] == 10: s = step.out
    elif x[VALUE] == 2 : s = step.over
    else               : s = step.into

    x[VALUE] += 100
    return s

  def post(self, x):
    x[VALUE] += 10000

def test_depth_first_search_inplace_modif():
  #   Reminder:
  #         1               lvl 3
  #      /  |  \
  #     |   |    3          lvl 2
  #     |   | /  |  \
  #     2\_ | 8  |   \      lvl 1
  #   /  \ \| |  |    \
  #   |  |  \ |  |    \
  #  4    7  \9  10   11    lvl 0

  g = rooted_f_graph_example()
  v = modifying_visitor_for_testing_dfs()
  depth_first_search(g,v)

  assert g.nodes() == [4,7,10102,20209,10108,10110,11,10103,10101]
