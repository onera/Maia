import maia.pytree.pred as PTp

from .conventions import is_intra_gc

def is_gc_of_kind(is_intra=None, is_1to1=None, is_perio=None):
  pred = PTp.is_gc_of_kind(is_1to1, is_perio)
  if is_intra is not None:
    pred &= PTp.NodePredicate(lambda n : is_intra_gc(n[0]) == is_intra)
  return pred

