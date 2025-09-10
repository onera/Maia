"""
  Draft. The idea is to have an alternative to the current list-of-list PyTree representation of a CGNS tree that is
  - backward compatible with the list-of-list (idea: inherit from `list` and rely on duck-typing)
  - offers `.name`, `.label`,  `.value`, `.children` attributes, more explicit than [0], [3], [1], [2]
  - offers a `.parent` attribute that makes the tree backward iterable (i.e. a node can get access to its parents by itself)

  Advantages:
    - still very regular and lightweigh tree structure, independent from the SIDS
    - backward compatible
    - more explicit
    - we can have more check on admissible values
    - we can reconstruct the full path of a node without giving the root tree

  Limitations:
    - independent from the SIDS
       - good because no need to implement the SIDS
       - bad because no built-in checks related to the SIDS ("class invariants")
    - once a library function relies on more that the common PyTree interface (e.g. it uses `.parent`)
        then the function can't be called on a plain regular PyTree
"""

class cgns_tree(list):
  def __init__(self, name, label, value, children, parent=None):
    self.name = name
    self.label = label
    self.value = value
    self.children = children
    self.parent = parent

  def child(self, i):
    _child = self.children[i]
    return cgns_tree(_child.name, _child.label, _child.value, _child.children, self)

  def __getitem__(self, i):
    if i==0:
      return self.name
    if i==1:
      return self.value
    if i==2:
      return self.children
    if i==3:
      return self.label
    raise RuntimeError()


def pytree_to_cgns_tree(pytree, parent=None):
  children = [pytree_to_cgns_tree(pytree_child) for pytree_child in pytree[2]]
  return cgns_tree(pytree[0], pytree[3], pytree[1], children)


def test_super_cgns_interface():
  pytree = ['Zone', [[9,0,0],[4,0,0],[0,0,0]], [['Hexa',[17],[],'Elements_t']], 'Zone_t']
  t = pytree_to_cgns_tree(pytree)

  # new interface
  assert t.name == 'Zone'
  hexa = t.child(0)
  assert hexa.name == 'Hexa'
  assert hexa.parent.name == 'Zone'

  # backward-compatible old interface
  assert t[0] == 'Zone'
  assert isinstance(t, list)
