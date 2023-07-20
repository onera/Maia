from maia.pytree.graph.nested_tree import Tree, SYMBOLIC_ROOT
from maia.pytree.graph.algo import depth_first_search

t =  Tree('A', [
       Tree('B',[
         Tree('C'),
         Tree('D'),
       ]),
       Tree('E'),
     ])

def test_nested_tree():
  expected_str_t = ('A\n'
                    '    B\n'
                    '        C\n'
                    '        D\n'
                    '    E\n'
                   )

  assert str(t) == expected_str_t

def test_nested_tree_conforms_to_dfs_interface():
  class visitor:
    def __init__(self):
      self.s = ''
    def pre(self, node):
      self.s += str(node.node_value)

  v = visitor()
  depth_first_search(t, v)

  assert v.s == 'ABCDE'

def test_with_symbolic_root():
  t =  Tree(SYMBOLIC_ROOT, [
         Tree('B',[
           Tree('C'),
           Tree('D'),
         ]),
         Tree('E'),
       ])

  expected_str_t = ('B\n'
                    '    C\n'
                    '    D\n'
                    'E\n'
                   )

  assert str(t) == expected_str_t
