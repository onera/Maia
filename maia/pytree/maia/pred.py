import maia.pytree as PT
import maia.pytree.pred as PTp

from .conventions import is_intra_gc

def is_gc_of_kind(is_intra=None, is_1to1=None, is_perio=None):
  pred = PTp.is_gc_of_kind(is_1to1, is_perio)
  if is_intra is not None:
    pred &= PTp.NodePredicate(lambda n : is_intra_gc(n[0]) == is_intra)
  return pred

FULL_CTN = PTp.label_in(['FlowSolution_t', 'DiscreteData_t']) \
         & PTp.NodePredicate(lambda c: not PT.Container._is_partial(c)) \
         & PTp.has_child_of_label('DataArray_t')

FULL_CTN_VTX  = FULL_CTN & PTp.has_location('Vertex') 
FULL_CTN_CELL = FULL_CTN & PTp.has_location('CellCenter') 

BASE_THEN_ZONE = [PTp.label_is('CGNSBase_t'),  PTp.label_in(['Zone_t', 'ParticleZone_t'])]