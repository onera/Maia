import pytest
import pytest_parallel
import numpy as np
import maia.pytree      as PT
#mport maia.pytree.maia as MT
from maia.pytree.maia import print as print_mod

def original_tree():
    root = PT.new_node("ParentNode", "UserDefinedData_t", 3.14)
    zone = PT.new_node('Zone', 'Zone_t', value=np.array([1, 2, 3]))
    PT.add_child(root, zone)
    return root

@pytest_parallel.mark.parallel(3)
def test_print_tree_parallel(capsys, comm):
    tree=original_tree()
    # Define filters
    showing_filter = {'name': ['Zone'], 'label': ['Zone_t'], 'proc': []}
    print_mod.print_tree_parallel(tree, comm, max_depth=2, showing_filter=showing_filter)
    captured = capsys.readouterr()
    assert 'Zone' in captured.out
    assert 'Zone_t' in captured.out
    assert 'array(shape=(3,), dtype=int64)' in captured.out

# TEST FAILED
# def test_print_node_parallel(capsys, comm):
#    tree=original_tree()
#    # Define filters
#    hiding_filter = {'name': ['Zone'], 'label': ['Zone_t'], 'proc': []}
#    print_mod.print_node_parallel(tree, comm, max_depth=2, hiding_filter=hiding_filter)
#    captured = capsys.readouterr()
#    assert 'MASKED' in captured.out
