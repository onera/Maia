import Converter.Internal as I

import maia.pytree as PT
from . import explore

def enforceDonorAsPath(tree):
  """ Force the GCs to indicate their opposite zone under the form BaseName/ZoneName """
  predicates = ['Zone_t', 'ZoneGridConnectivity_t', lambda n: I.getType(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t']]
  for base in I.getBases(tree):
    base_n = I.getName(base)
    for gc in PT.iter_children_from_predicates(base, predicates):
      I.setValue(gc, explore.getZoneDonorPath(base_n, gc))

