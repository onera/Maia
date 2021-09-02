from abc import abstractmethod
import queue

__NAME__     = 0
__VALUE__    = 1
__CHILDREN__ = 2
__LABEL__    = 3



# --------------------------------------------------------------------------
class NodeParserBase:

  DEFAULT="dfs"
  MAXDEPTH=30

  def __init__(self, depth=MAXDEPTH, sort=lambda children:children):
    self.depth = depth
    self.sort  = sort

  @abstractmethod
  def bfs(self, root, predicate):
    pass

  def dfs(self, root, predicate):
    # print(f"NodeParserBase.dfs: root = {I.getName(root)}")
    if predicate(root):
      return root
    return self._dfs(root, predicate)

  @abstractmethod
  def _dfs(self, parent, predicate, level=1):
    pass

# --------------------------------------------------------------------------
class NodeParser(NodeParserBase):

  def bfs(self, root, predicate):
    # print(f"NodeParser.bfs: root = {I.getName(root)}")
    temp = queue.Queue()
    temp.put(root)
    while not temp.empty():
      node = temp.get()
      # print(f"NodeParser.bfs: node = {I.getName(node)}")
      if predicate(node):
        return node
      for child in self.sort(node[__CHILDREN__]):
        temp.put(child)
    return None

  def _dfs(self, parent, predicate):
    # print(f"NodeParser._dfs: parent = {I.getName(parent)}")
    for child in self.sort(parent[__CHILDREN__]):
      if predicate(child):
        return child
      # Explore next level
      result = self._dfs(child, predicate)
      if result is not None:
        return result
    return None

class LevelNodeParser(NodeParserBase):

  def bfs(self, root, predicate, level=1):
    # print(f"LevelNodeParser.bfs: depth = {self.depth}: root = {I.getName(root)}")
    temp = queue.Queue()
    temp.put( (0, root,) )
    while not temp.empty():
      level, node = temp.get()
      # print(f"LevelNodeParser.bfs: level:{level} < depth:{self.depth}: node = {I.getName(node)}")
      if predicate(node):
        return node
      if level < self.depth:
        for child in self.sort(node[__CHILDREN__]):
          temp.put( (level+1, child) )
    return None

  def _dfs(self, parent, predicate, level=1):
    # print(f"LevelNodeParser.dfs: level = {level} < depth = {self.depth}: parent = {I.getName(parent)}")
    for child in self.sort(parent[__CHILDREN__]):
      if predicate(child):
        return child
      if level < self.depth:
        # Explore next level
        result = self._dfs(child, predicate, level=level+1)
        if result is not None:
          return result
    return None




# --------------------------------------------------------------------------
class NodesIteratorBase:

  MAXDEPTH=30
  DEFAULT='dfs'

  def __init__(self, depth=MAXDEPTH, sort=lambda children:children):
    self.depth = depth
    self.sort  = sort

  @abstractmethod
  def bfs(self, root, predicate):
    pass

  def dfs(self, root, predicate):
    # print(f"NodesIterator.dfs: root = {I.getName(root)}")
    if predicate(root):
      yield root
    yield from self._dfs(root, predicate)

  @abstractmethod
  def _dfs(self, parent, predicate, level=1):
    pass


# --------------------------------------------------------------------------
class NodesIterator(NodesIteratorBase):

  """ Stop exploration if something found at a level """

  def bfs(self, root, predicate):
    # print(f"ShallowNodesIterator.bfs: root = {I.getName(root)}")
    temp = queue.Queue()
    temp.put(root)
    while not temp.empty():
      node = temp.get()
      # print(f"ShallowNodesIterator.bfs: node = {I.getName(node)}")
      if predicate(node):
        yield node
      for child in self.sort(node[__CHILDREN__]):
        temp.put(child)

  def _dfs(self, parent, predicate, level=1):
    # print(f"ShallowNodesIterator._dfs: parent = {I.getName(parent)}")
    for child in self.sort(parent[__CHILDREN__]):
      if predicate(child):
        yield child
      # Explore next level
      yield from self._dfs(child, predicate)

class ShallowNodesIterator(NodesIteratorBase):

  """ Stop exploration if something found at a level """

  def bfs(self, root, predicate):
    # print(f"ShallowNodesIterator.bfs: root = {I.getName(root)}")
    temp = queue.Queue()
    temp.put(root)
    while not temp.empty():
      node = temp.get()
      # print(f"ShallowNodesIterator.bfs: node = {I.getName(node)}")
      if predicate(node):
        yield node
      else:
        for child in self.sort(node[__CHILDREN__]):
          temp.put(child)

  def _dfs(self, parent, predicate, level=1):
    # print(f"ShallowNodesIterator._dfs: parent = {I.getName(parent)}")
    for child in self.sort(parent[__CHILDREN__]):
      if predicate(child):
        yield child
      else:
        # Explore next level
        yield from self._dfs(child, predicate)

class LevelNodesIterator(NodesIteratorBase):

  """ Stop exploration until a limited level """

  def bfs(self, root, predicate):
    # print(f"LevelNodesIterator.bfs: depth = {self.depth}: root = {I.getName(root)}")
    temp = queue.Queue()
    temp.put( (0, root,) )
    while not temp.empty():
      level, node = temp.get()
      # print(f"LevelNodesIterator.bfs: level:{level} < depth:{self.depth}: node = {I.getName(node)}")
      if predicate(node):
        yield node
      if level < self.depth:
        for child in self.sort(node[__CHILDREN__]):
          temp.put( (level+1, child) )

  def _dfs(self, parent, predicate, level=1):
    # print(f"LevelNodesIterator._dfs: level = {level} < depth = {self.depth}: parent = {I.getName(parent)}")
    for child in self.sort(parent[__CHILDREN__]):
      if predicate(child):
        yield child
      if level < self.depth:
        # Explore next level
        yield from self._dfs(child, predicate, level=level+1)

class ShallowLevelNodesIterator(NodesIteratorBase):

  """ Stop exploration if something found at a level until a limited level """

  def bfs(self, root, predicate):
    # print(f"ShallowLevelNodesIterator.bfs: root = {I.getName(root)}")
    temp = queue.Queue()
    temp.put( (0, root,) )
    while not temp.empty():
      level, node = temp.get()
      # print(f"ShallowLevelNodesIterator.bfs: node = {I.getName(node)}")
      if predicate(node):
        yield node
      else:
        if level < self.depth:
          for child in self.sort(node[__CHILDREN__]):
            temp.put( (level+1, child) )

  def _dfs(self, parent, predicate, level=1):
    # print(f"ShallowLevelNodesIterator._dfs: parent = {I.getName(parent)}")
    for child in self.sort(parent[__CHILDREN__]):
      if predicate(child):
        yield child
      else:
        if level < self.depth:
          # Explore next level
          yield from self._dfs(child, predicate, level=level+1)
