def append_unique(L, item):
  """ Add an item in a list only if not already present"""
  if item not in L:
    L.append(item)

def bucket_split(l, f, compress=False, size=None):
  """ Dispatch the elements of list l into n sublists, according to the result of function f """
  if size is None: 
    size = max(f(e) for e in l) + 1
  result = [ [] for i in range(size)]
  for e in l:
    result[f(e)].append(e)
  if compress:
    result = [sub_l for sub_l in result if sub_l]
  return result

def are_overlapping(range1, range2, strict=False):
  """ Return True if range1 and range2 share a common element.
  If strict=True, case (eg) End1 == Start2 is not considered to overlap 
  https://is.gd/gTBuwu """
  assert range1[0] <= range1[1] and range2[0] <= range2[1]
  if strict:
    return range1[0] < range2[1] and range2[0] < range1[1]
  else:
    return range1[0] <= range2[1] and range2[0] <= range1[1]

def expects_one(L, err_msg=("elem", "list")):
  """
  Raise a RuntimeError if L does not contains exactly one element. Otherwise,
  return this element
  """
  assert isinstance(L, list)
  if len(L) == 0:
    raise RuntimeError(f"{err_msg[0]} not found in {err_msg[1]}")
  elif len(L) > 1:
    raise RuntimeError(f"Multiple {err_msg[0]} found in {err_msg[1]}")
  else:
    return L[0]

