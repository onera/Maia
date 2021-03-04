import Converter.Internal as I

def shorten_field_names(t, quiet=False):
  for x in I.getNodesFromType(t,"DataArray_t"):
    old_name = I.getName(x)
    if (len(old_name)>32):
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
      I.setName(x,new_name)

def shorten_names(t, quiet=False):
  old_name = I.getName(t)
  if (len(old_name)>32):
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
    I.setName(t,new_name)
  for x in I.getChildren(t):
    shorten_names(x,quiet)
