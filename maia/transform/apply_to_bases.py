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
