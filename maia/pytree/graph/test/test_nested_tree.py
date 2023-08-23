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
    def pre(self, tree):
      self.s += str(tree.node)

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

  B = A.children[0]
  C = B.children[0]

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

  A.children.append(ForwardBackwardTree('F'))

  F = A.children[-1]
  assert F.node == 'F'
  assert F.parent is A

  B = A.children[0]
  A.children[0] = ForwardBackwardTree('G')
  assert A.children[0].node == 'G'
  assert A.children[0].parent is A
  assert B.parent is None # B has been orphaned
  assert str(B) == 'B\n    C\n    D\n' # but B can still be accessed

  A.children += [ForwardBackwardTree('H'), ForwardBackwardTree('I')]
  assert len(A.children) == 5
  assert A.children[0].node == 'G'; assert A.children[0].parent is A;
  assert A.children[1].node == 'E'; assert A.children[1].parent is A;
  assert A.children[2].node == 'F'; assert A.children[2].parent is A;
  assert A.children[3].node == 'H'; assert A.children[3].parent is A;
  assert A.children[4].node == 'I'; assert A.children[4].parent is A;

  A.children = [ForwardBackwardTree('J'), ForwardBackwardTree('K')]
  assert len(A.children) == 2
  assert A.children[0].node == 'J'; assert A.children[0].parent is A;
  assert A.children[1].node == 'K'; assert A.children[1].parent is A;


def test_forward_backward_nested_tree_ref_counting():
  import gc
  B = ForwardBackwardTree('B')
  A = ForwardBackwardTree('A', [B])

  gc.collect() # should not do anything fancy
  assert B.parent is A

  A = 'unrelated string' # `A` is now bound to a new value: the old `A` tree is unreachable
  gc.collect() # since the old `A` is unreachable, it should be collected

  assert B.parent is None # since the old `A` has been collected, `B` no longer has a parent
