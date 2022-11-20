from maia.utils.graph.algo_base import dfs_interface_report
from maia.utils.graph.io_graph import io_graph_tree_adaptor, rooted_tree_example

def test_io_graph_tree_adaptor_is_depth_first_searchable():
  t = rooted_tree_example()
  assert type(t) == io_graph_tree_adaptor
  assert dfs_interface_report(t) == ''
