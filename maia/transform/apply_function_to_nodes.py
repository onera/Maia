import Converter.Internal as I

def apply_to_bases(t,f,*args):
  if I.getType(t)=="CGNSBase_t":
    f(t,*args)
  elif I.getType(t)=="CGNSTree_t":
    for b in I.getBases(t):
      f(b,*args)
  else:
    raise Exception("function \""+f.__name__+"\"" \
                    " can only be applied to a \"CGNSBase_t\" or on a complete \"CGNSTree_t\"," \
                    " not on a node of type \""+I.getType(t)+"\".")

def apply_to_zones(t,f,*args):
  if I.getType(t)=="Zone_t":
    f(t,*args)
  elif I.getType(t)=="CGNSTree_t" or I.getType(t)=="CGNSBase_t":
    for z in I.getZones(t):
      f(z,*args)
  else:
    raise Exception("function \""+f.__name__+"\"" \
                    " can only be applied to a \"Zone_t\", a \"CGNSBase_t\" or on a complete \"CGNSTree_t\"," \
                    " not on a node of type \""+I.getType(t)+"\".")

