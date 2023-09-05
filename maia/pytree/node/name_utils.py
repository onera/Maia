import maia.pytree as PT

def shorten_names(t, quiet=False, labels_to_shorten=None):
  old_name = PT.get_name(t)
  can_shorten_label = labels_to_shorten is None or (PT.get_label(t) in labels_to_shorten)
  if can_shorten_label and len(old_name)>32:
    new_name = ""
    cnt = 0
    for c in old_name:
      if c.isupper():
        cnt = 0
        new_name += c
      else:
        cnt += 1
        if cnt < 4:
          new_name += c
    if not quiet:
      print("WARNING: field "+old_name+" is too long. It will be renamed "+new_name)
    PT.set_name(t,new_name)
  for x in PT.get_children(t):
    shorten_names(x,quiet,labels_to_shorten)

def shorten_field_names(t, quiet=False):
  shorten_names(t,quiet,labels_to_shorten=["DataArray_t"])

def rename_zone(t, name, new_name):
  """
  Rename a zone and its occurences in GCs
  """
  is_gc = lambda n: PT.get_label(n) in ['GridConnectivity1to1_t', 'GridConnectivity_t'] and \
                    PT.get_value(n) == name
  zones = PT.get_all_Zone_t(t)
  for zone in zones:
    if PT.get_name(zone) == name:
      PT.set_name(zone, new_name)
    for gc in PT.get_children_from_predicates(zone, ['ZoneGridConnectivity_t', is_gc]):
      PT.set_value(gc, new_name)
