import maia.pytree.pred as PTp

from .conventions import is_intra_gc

def is_gc_with(intra=None, match=None, perio=None):
  pred = PTp.is_gc_with(match, perio)
  if intra is not None:
    pred = pred & PTp.UnaryPredicate(lambda n : is_intra_gc(n[0]) == intra)
  return pred

