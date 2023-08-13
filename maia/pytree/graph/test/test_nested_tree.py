from maia.pytree.graph.nested_tree import Tree, ForwardBackwardTree, SYMBOLIC_ROOT
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


def test_forward_backward_nested_tree_base():
  A =  ForwardBackwardTree('A', [
         ForwardBackwardTree('B',[
           ForwardBackwardTree('C'),
           ForwardBackwardTree('D'),
         ]),
         ForwardBackwardTree('E'),
       ])

  B = A.sub_nodes[0]
  C = B.sub_nodes[0]

  assert A.parent is None
  assert B.parent is A
  assert C.parent is B

  super_tree = ForwardBackwardTree('S', [A])
  assert super_tree.parent is None
  assert A.parent is super_tree


def test_forward_backward_nested_tree_mutate_children():
  A =  ForwardBackwardTree('A', [
         ForwardBackwardTree('B',[
           ForwardBackwardTree('C'),
           ForwardBackwardTree('D'),
         ]),
         ForwardBackwardTree('E'),
       ])

  A.sub_nodes.append(ForwardBackwardTree('F'))

  F = A.sub_nodes[-1]
  assert F.node_value == 'F'
  assert F.parent is A

  B = A.sub_nodes[0]
  A.sub_nodes[0] = ForwardBackwardTree('G')
  assert A.sub_nodes[0].node_value == 'G'
  assert A.sub_nodes[0].parent is A
  assert B.parent is None # B has been orphaned
  assert str(B) == 'B\n    C\n    D\n' # but B can still be accessed

  A.sub_nodes += [ForwardBackwardTree('H'), ForwardBackwardTree('I')]
  assert len(A.sub_nodes) == 5
  assert A.sub_nodes[0].node_value == 'G'; assert A.sub_nodes[0].parent is A;
  assert A.sub_nodes[1].node_value == 'E'; assert A.sub_nodes[1].parent is A;
  assert A.sub_nodes[2].node_value == 'F'; assert A.sub_nodes[2].parent is A;
  assert A.sub_nodes[3].node_value == 'H'; assert A.sub_nodes[3].parent is A;
  assert A.sub_nodes[4].node_value == 'I'; assert A.sub_nodes[4].parent is A;

  A.sub_nodes = [ForwardBackwardTree('J'), ForwardBackwardTree('K')]
  assert len(A.sub_nodes) == 2
  assert A.sub_nodes[0].node_value == 'J'; assert A.sub_nodes[0].parent is A;
  assert A.sub_nodes[1].node_value == 'K'; assert A.sub_nodes[1].parent is A;


def test_forward_backward_nested_tree_ref_counting():
  import gc
  B = ForwardBackwardTree('B')
  A = ForwardBackwardTree('A', [B])

  gc.collect() # should not do anything fancy
  assert B.parent is A

  A = 'unrelated string' # `A` is now bound to a new value: the old `A` tree is unreachable
  gc.collect() # since the old `A` is unreachable, it should be collected

  assert B.parent is None # since the old `A` has been collected, `B` no longer has a parent
