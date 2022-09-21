import maia.pytree as PT

def apply_to_bases(f, t, *args):
  if PT.get_label(t)=="CGNSBase_t":
    b_iter = [t]
  elif PT.get_label(t)=="CGNSTree_t":
    b_iter = PT.iter_children_from_label(t, "CGNSBase_t")
  else:
    raise Exception("function \""+f.__name__+"\"" \
                    " can only be applied to a \"CGNSBase_t\" or on a complete \"CGNSTree_t\"," \
                    " not on a node of type \""+PT.get_label(t)+"\".")
  for b in b_iter:
    f(b, *args)

def zones_iterator(t):
  """
  Helper iterator to loop over zones from a tree, a base or a zone node
  """
  if PT.get_label(t)=="Zone_t":
    yield t
  elif PT.get_label(t)=="CGNSBase_t":
    yield from PT.iter_children_from_label(t, "Zone_t")
  elif PT.get_label(t)=="CGNSTree_t":
    yield from PT.iter_children_from_labels(t, ["CGNSBase_t", "Zone_t"])
  else:
    raise ValueError("Unvalid object for zones iterator")

def apply_to_zones(f, t, *args):
  if PT.get_label(t) not in ["Zone_t", "CGNSBase_t", "CGNSTree_t"]:
    raise Exception("function \""+f.__name__+"\"" \
                    " can only be applied to a \"Zone_t\", a \"CGNSBase_t\" or on a complete \"CGNSTree_t\"," \
                    " not on a node of type \""+PT.get_label(t)+"\".")
  else:
    for z in zones_iterator(t):
      f(z, *args)

