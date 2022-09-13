import maia.pytree as PT
from . import explore

def enforceDonorAsPath(tree):
  """ Force the GCs to indicate their opposite zone under the form BaseName/ZoneName """
  predicates = ['Zone_t', 'ZoneGridConnectivity_t', lambda n: PT.get_label(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t']]
  for base in PT.iter_all_CGNSBase_t(tree):
    base_n = PT.get_name(base)
    for gc in PT.iter_children_from_predicates(base, predicates):
      PT.set_value(gc, explore.getZoneDonorPath(base_n, gc))

