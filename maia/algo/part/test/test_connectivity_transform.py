import pytest
import pytest_parallel
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia
from maia.factory       import dcube_generator as DCG
from maia.algo.part     import connectivity_transform as CNT

def as_int32(zone):
  predicate = lambda n : PT.get_label(n) in ['Zone_t', 'DataArray_t', 'IndexArray_t', 'IndexRange_t']
  for array in PT.iter_nodes_from_predicate(zone, predicate, explore='deep'):
    array[1] = array[1].astype(np.int32)

@pytest_parallel.mark.parallel(1)
def test_enforce_boundary_pe_left(comm):
  tree = DCG.dcube_generate(3, 1., [0., 0., 0.], comm)
  zone = PT.get_all_Zone_t(tree)[0]
  as_int32(zone)
  maia.algo.pe_to_nface(zone, comm)
  pe_node = PT.get_node_from_path(zone, 'NGonElements/ParentElements')
  pe_bck = pe_node[1].copy()
  zone_bck = PT.deep_copy(zone)
  CNT.enforce_boundary_pe_left(zone)
  assert PT.is_same_tree(zone, zone_bck)

  #Test with swapped pe
  pe_node[1][2] = pe_node[1][2][::-1]
  CNT.enforce_boundary_pe_left(zone)

  assert PT.get_node_from_path(zone, 'NFaceElements/ElementConnectivity')[1][12] == -29
  expt_ng_ec = PT.get_node_from_path(zone_bck, 'NGonElements/ElementConnectivity')[1].copy()
  expt_ng_ec[4*2 : 4*3] = [4, 7, 8, 5]
  assert (PT.get_node_from_path(zone, 'NGonElements/ElementConnectivity')[1] == expt_ng_ec).all()

  #Test with no NFace
  zone = PT.deep_copy(zone_bck)
  pe_node = PT.get_node_from_path(zone, 'NGonElements/ParentElements')
  pe_node[1][2] = pe_node[1][2][::-1]
  PT.rm_children_from_name(zone, 'NFaceElements')
  CNT.enforce_boundary_pe_left(zone)
  assert (PT.get_node_from_path(zone, 'NGonElements/ElementConnectivity')[1] == expt_ng_ec).all()
