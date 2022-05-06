# --------------------------------------------------------------------------
def getParentFromPredicate(start, node, predicate, prev=None):
    """Return thee first parent node matching type."""
    if id(start) == id(node):
      return prev
    if predicate(start):
      prev = start
    for n in start[2]:
        ret = getParentFromPredicate(n, node, parentType, prev)
        if ret is not None: return ret
    return None

def getParentsFromPredicate(start, node, predicate, l=[]):
    """Return all parent nodes matching type."""
    if id(start) == id(node):
      return l
    if predicate(start):
      l.append(start)
    for n in start[2]:
        ret = getParentsFromPredicate(n, node, predicate, l)
        if ret != []: return ret
    return []

# --------------------------------------------------------------------------

