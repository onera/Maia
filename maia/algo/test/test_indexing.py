import pytest
import pytest_parallel
import numpy as np
import maia
import maia.pytree as PT
import pytest_parallel
from maia.algo import indexing

def test_get_pe_local():
  yt = """
  NGonElements Elements_t [22, 0]:
    ElementRange IndexRange_t [1, 8]:
    ParentElements DataArray_t [[9, 0], [10, 0], [11, 12], [0, 12]]:
  """
  ngon = PT.yaml.to_node(yt)
  assert (indexing.get_pe_local(ngon) == np.array([[9-8,0], [10-8,0], [11-8,12-8], [0,12-8]])).all()

  yt = """
  NGonElements Elements_t [22, 0]:
    ElementRange IndexRange_t [1, 8]:
    ParentElements DataArray_t [[1, 0], [2, 0], [3, 4], [0, 4]]:
  """
  ngon = PT.yaml.to_node(yt)
  assert (indexing.get_pe_local(ngon) == np.array([[1,0], [2,0], [3,4], [0,4]])).all()

  yt = """
  NGonElements Elements_t [22, 0]:
    ElementRange IndexRange_t [1, 8]:
  """
  ngon = PT.yaml.to_node(yt)
  with pytest.raises(RuntimeError):
    indexing.get_pe_local(ngon)
  


@pytest_parallel.mark.parallel(1)
def test_edge_pe_to_ngon(comm):
    dist_tree= maia.factory.generate_dist_block(4, "TRI_3", comm)
    maia.algo.dist.convert_elements_to_ngon(dist_tree, comm)
    #PT.print_tree(dist_tree)
    dist_ngon_bck = PT.get_node_from_name(dist_tree,'NGonElements')
    PT.rm_nodes_from_name(dist_tree, 'NGonElements')
    indexing.edge_pe_to_ngon(dist_tree, comm)
    dist_ngon_new = PT.get_node_from_name(dist_tree,'NGonElements')
    assert PT.is_same_tree(dist_ngon_new, dist_ngon_bck)
    part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
    part_ngon_bck = PT.get_node_from_name(part_tree,'NGonElements')
    PT.rm_nodes_from_name(part_tree, 'NGonElements')
    indexing.edge_pe_to_ngon(part_tree, True)
    part_ngon_new = PT.get_node_from_name(part_tree,'NGonElements')
    #assert PT.is_same_tree(part_ngon_new, part_ngon_bck) # does not work


@pytest_parallel.mark.parallel([1])
def test_ngon_to_edge_pe(comm):
    tree = maia.factory.generate_dist_block(4, "TRI_3", comm)
    maia.algo.dist.convert_elements_to_ngon(tree, comm)
    assert PT.get_node_from_name(tree, 'NGonElements') is not None, \
        "NGonElements node should exist after conversion."
    edge_node = PT.get_node_from_name(tree, 'EdgeElements')
    if edge_node is not None:
        parent_node = PT.get_child_from_name(edge_node, 'ParentElements')
        if parent_node is not None:
            PT.rm_child(edge_node, parent_node)
    
    indexing.ngon_to_edge_pe(tree, comm, True)
    assert PT.get_node_from_name(tree, 'NGonElements') is None, \
        "NGonElements node was not removed as expected."
    edge_node = PT.get_node_from_name(tree, 'EdgeElements')
    assert edge_node is not None, "EdgeElements node should exist."
    assert PT.get_child_from_name(edge_node, 'ParentElements') is not None, \
        "ParentElements node should exist under EdgeElements."
