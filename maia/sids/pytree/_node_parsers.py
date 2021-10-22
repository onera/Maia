from abc import abstractmethod
import queue

import Converter.Internal as I

__NAME__     = 0
__VALUE__    = 1
__CHILDREN__ = 2
__LABEL__    = 3


# --------------------------------------------------------------------------
class NodeParserBase:

  DEFAULT="dfs"

  def __init__(self, depth=[0,None], sort=lambda children:children):
    self.depth = depth
    self.sort  = sort
    self.cond1 = True if depth[1] is None else self.depth[1] > 0
    self.cond2 = (lambda l: True) if (depth[1] is None) else (lambda l: l < self.depth[1])

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


# --------------------------------------------------------------------------
class RangeLevelNodeParserBase(NodeParserBase):

  def dfs(self, root, predicate):
    # print(f"RangeLevelNodeParserBase.dfs: root = {I.getName(root)}, self.depth = {self.depth}")
    if self.depth[0] == 0 and predicate(root):
      # print(f"RangeLevelNodeParserBase.dfs:   parse root")
      return root
    if self.cond1:
      # print(f"RangeLevelNodeParserBase.dfs:   continue next level 1, self.cond1 = {self.cond1}")
      return self._dfs(root, predicate)


# --------------------------------------------------------------------------
class RangeLevelNodeParser(RangeLevelNodeParserBase):

  def bfs(self, root, predicate, level=1):
    # print(f"RangeLevelNodeParser.bfs:   depth = {self.depth}, root = {I.getName(root)}")
    temp = queue.Queue()
    temp.put( (0, root,) )
    while not temp.empty():
      level, node = temp.get()
      # print(f"RangeLevelNodeParser.bfs:   level={level}, depth={self.depth}, self.cond2(level)={self.cond2(level)}, node = {I.getName(node)}")
      if level >= self.depth[0] and predicate(node):
        return node
      if self.cond2(level):
        for child in self.sort(node[__CHILDREN__]):
          temp.put( (level+1, child) )
    return None

  def _dfs(self, parent, predicate, level=1):
    # print(f"RangeLevelNodeParser.dfs:   level={level}, depth={self.depth}, self.cond2(level)={self.cond2(level)}, parent={I.getName(parent)}")
    for child in self.sort(parent[__CHILDREN__]):
      if level >= self.depth[0] and predicate(child):
        return child
      if self.cond2(level):
        # Explore next level
        result = self._dfs(child, predicate, level=level+1)
        if result is not None:
          return result
    return None


# --------------------------------------------------------------------------
class NodesIteratorBase:

  DEFAULT='dfs'

  def __init__(self, depth=[0,None], sort=lambda children:children):
    self.depth = depth
    self.sort  = sort
    self.cond1 = True if depth[1] is None else self.depth[1] > 0
    self.cond2 = (lambda l: True) if (depth[1] is None) else (lambda l: l < self.depth[1])

  @abstractmethod
  def bfs(self, root, predicate):
    pass

  def dfs(self, root, predicate):
    # print(f"NodesIteratorBase.dfs: root = {I.getName(root)}")
    if predicate(root):
      # print(f"NodesIteratorBase.dfs: yield root")
      yield root
    # print(f"NodesIteratorBase.dfs:   continue under root...")
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


# --------------------------------------------------------------------------
class RangeLevelNodesIteratorBase(NodesIteratorBase):

  def dfs(self, root, predicate):
    # print(f"RangeLevelNodesIteratorBase.dfs: root = {I.getName(root)}, self.depth = {self.depth}")
    if self.depth[0] == 0 and predicate(root):
      # print(f"RangeLevelNodesIteratorBase.dfs:   parse root")
      yield root
    if self.cond1:
      # print(f"RangeLevelNodesIteratorBase.dfs:   continue next level 1, self.cond1 = {self.cond1}")
      yield from self._dfs(root, predicate)


# --------------------------------------------------------------------------
class RangeLevelNodesIterator(RangeLevelNodesIteratorBase):

  """ Stop exploration until a limited level """

  def bfs(self, root, predicate):
    # print(f"RangeLevelNodesIterator.bfs: depth = {self.depth}, root = {I.getName(root)}")
    temp = queue.Queue()
    temp.put( (0, root,) )
    while not temp.empty():
      level, node = temp.get()
      # print(f"RangeLevelNodesIterator.bfs:   level={level}, depth={self.depth}, self.cond2(level)={self.cond2(level)}, node = {I.getName(node)}")
      if level >= self.depth[0] and predicate(node):
        yield node
      if self.cond2(level):
        for child in self.sort(node[__CHILDREN__]):
          temp.put( (level+1, child) )

  def _dfs(self, parent, predicate, level=1):
    # print(f"RangeLevelNodesIterator._dfs:   level={level}, depth={self.depth}, self.cond2(level)={self.cond2(level)}, parent={I.getName(parent)}")
    for child in self.sort(parent[__CHILDREN__]):
      if level >= self.depth[0] and predicate(child):
        yield child
      if self.cond2(level):
        # Explore next level
        yield from self._dfs(child, predicate, level=level+1)


class ShallowRangeLevelNodesIterator(RangeLevelNodesIteratorBase):

  """ Stop exploration if something found at a level until a limited level """

  def bfs(self, root, predicate):
    # print(f"ShallowRangeLevelNodesIterator.bfs: depth = {self.depth}, root = {I.getName(root)}")
    temp = queue.Queue()
    temp.put( (0, root,) )
    while not temp.empty():
      level, node = temp.get()
      # print(f"ShallowRangeLevelNodesIterator.bfs:    level={level}, depth={self.depth}, self.cond2(level)={self.cond2(level)}, node = {I.getName(node)}")
      if level >= self.depth[0] and predicate(node):
        yield node
      else:
        if self.cond2(level):
          for child in self.sort(node[__CHILDREN__]):
            temp.put( (level+1, child) )

  def _dfs(self, parent, predicate, level=1):
    # print(f"ShallowRangeLevelNodesIterator._dfs:   level={level}, depth={self.depth}, self.cond2(level)={self.cond2(level)}, parent={I.getName(parent)}")
    for child in self.sort(parent[__CHILDREN__]):
      if level >= self.depth[0] and predicate(child):
        yield child
      else:
        if self.cond2(level):
          # Explore next level
          yield from self._dfs(child, predicate, level=level+1)
