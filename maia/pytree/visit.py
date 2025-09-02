from maia.pytree.typing import *

from .graph.cgns import depth_first_search
from .graph.algo import Step

__all__ = ['scan', 'visit', 'Step']

def scan(tree:CGNSTree, callable:Callable[[CGNSTree], None], ancestors:bool=False):
  """
  Recursively apply the callable function to every node of the input tree.

  If ``ancestors==False``, the callable must have the signature ``f(node:CGNSTree) -> None``.
  Otherwise, the expected signature is ``f(nodes:List[CGNSTree]) -> None``, and the function
  will be called on the current node and its ancestors.

  Args:
    tree (CGNSTree): Input tree
    callable (function): User function to apply on each node
    ancestors (bool): If ``True``, also pass ancestors to callable function
  Examples:
    >>> tree = PT.yaml.to_node('''
    ... Base CGNSBase_t:
    ...   Zone1 Zone_t:
    ...     ZoneGridConnectivity ZoneGridConnectivity_t:
    ...       match GridConnectivity1to1_t "Zone3":
    ... ''')
    >>> PT.scan(tree, lambda n: PT.update_node(n, name=PT.get_name(n).upper()))
    >>> PT.print_tree(tree)
    BASE CGNSBase_t 
    └───ZONE1 Zone_t 
        └───ZONEGRIDCONNECTIVITY ZoneGridConnectivity_t 
            └───MATCH GridConnectivity1to1_t "Zone3"

    >>> tree = PT.yaml.to_node('''
    ... Base CGNSBase_t:
    ...   Zone1 Zone_t:
    ...     ZoneGridConnectivity ZoneGridConnectivity_t:
    ...       match GridConnectivity1to1_t "Zone3":
    ... ''')
    >>> gc_paths = []
    >>> def gc_collector(nodes):
    ...   if PT.get_label(nodes[-1]) == 'GridConnectivity1to1_t':
    ...     gc_paths.append('/'.join([PT.get_name(n) for n in nodes]))
    >>> PT.scan(tree, gc_collector, ancestors=True)
    >>> print(gc_paths)
    ['Base/Zone1/ZoneGridConnectivity/match']
  """
  class visitor:
    def __init__(self, f):
      self.f = f
    def pre(self, arg):
      self.f(arg)

  v = visitor(callable)
  depth = 'all' if ancestors else 'node'
  depth_first_search(tree, v, depth)


def visit(tree:CGNSTree, visitor, ancestors:bool=False):
  """
  Recursively evaluate the visitor functions for every node of the tree.

  This is the complex (but more flexible) version of :func:`scan`;
  ``visitor`` must be an object exposing the following functions::

    class Visitor:
      # The visitor interface can expose the following functions.
      # You may omit some functions; in this case, an empty implementation is insered.

      def pre(self, node:CGNSTree) -> Optional[Step]:
        # Called when `node` is visited for the first time
      def down(self, parent:CGNSTree, child:CGNSTree) -> None:
        # Called when moving down from `parent` to `child`
      def up(self, child:CGNSTree, parent:CGNSTree) -> None:
        # Called when moving back from `child` to `parent`
      def post(self, node:CGNSTree) -> None:
        # Called when all the children of `node` are completed

  If ``ancestors=True``, the argument of ``pre`` and ``post`` becomes ``nodes:List[CGNSTree]``;
  these functions are called on the current node and its ancestors.

  At each level, the return value of ``pre`` function can be used to decide how to continue the recursion::

    class Step(Enum):
      INTO = 0 # continue normally: go down in children of current node
      OVER = 1 # stop current level: do not visit children, go up and continue
      OUT  = 2 # stop visit process: rewind to top level and exit

  If ``pre`` returns nothing, the recusion continues normally, which is equivalent to ``Step.INTO``.

  Args:
    tree (CGNSTree): Input tree
    visitor (visitor object): Functions to apply on each node
    ancestors (bool): If ``True``, also pass ancestors to visitor functions

  Examples:
    >>> tree = PT.yaml.to_node('''
    ... CGNSTree CGNSTree_t:
    ...   CGNSLibraryVersion CGNSLibraryVersion_t:
    ...   Base CGNSBase_t:
    ...     Zone2 Zone_t:
    ...     Zone1 Zone_t:
    ...       Tri  Elements_t [5, 0]:
    ...       Quad Elements_t [7, 0]:
    ... ''')
    >>> class TreeSorter:
    ...   # A visitor that sort children of nodes, until max level is reached
    ...   def __init__(self, level):
    ...     self.level = level
    ...   def pre(self, nodes):
    ...     last = nodes[-1]
    ...     PT.set_children(last, sorted(PT.get_children(last)))
    ...     if len(nodes) >= self.level:
    ...       return PT.Step.OVER
    >>> PT.visit(tree, TreeSorter(2), ancestors=True)
    >>> PT.print_tree(tree)
    CGNSTree CGNSTree_t 
    ├───Base CGNSBase_t 
    │   ├───Zone1 Zone_t 
    │   │   ├───Tri Elements_t I4 [5 0]
    │   │   └───Quad Elements_t I4 [7 0]
    │   └───Zone2 Zone_t 
    └───CGNSLibraryVersion CGNSLibraryVersion_t 
  """
  depth = 'all' if ancestors else 'node'
  depth_first_search(tree, visitor, depth)


