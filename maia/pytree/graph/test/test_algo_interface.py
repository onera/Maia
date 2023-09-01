from maia.pytree.graph.algo_interface import dfs_interface_report
from maia.pytree.graph.utils import list_iterator_type

def test_dfs_interface_report():
  class graph_type_0:
    pass

  graph_conforms_to_interface, report = dfs_interface_report(graph_type_0())

  assert not graph_conforms_to_interface
  assert report == ('Type "graph_type_0" does not satisfy the interface of the depth-first search algorithm:\n'
                    '  Attribute "root_iterator" should be of the form\n'
                    '      `def root_iterator(self) -> graph_child_iterator`\n'
                    '    but it is not because there is no such attribute\n'
                    '  Attribute "child_iterator" should be of the form\n'
                    '      `def child_iterator(self, node) -> graph_child_iterator`\n'
                    '    but it is not because there is no such attribute\n'
                   )


  class graph_type_1:
    def root_iterator(self):
      return 0
    def child_iterator(self, node):
      return 0

  graph_conforms_to_interface, report = dfs_interface_report(graph_type_1())

  assert not graph_conforms_to_interface
  assert report == ('Type "graph_type_1" does not satisfy the interface of the depth-first search algorithm:\n'
                    '  Attribute "root_iterator" should be of the form\n'
                    '      `def root_iterator(self) -> graph_child_iterator`\n'
                    '    but it is not because it must have a return annotation so that the return type can be checked\n'
                    '  Attribute "child_iterator" should be of the form\n'
                    '      `def child_iterator(self, node) -> graph_child_iterator`\n'
                    '    but it is not because it must have a return annotation so that the return type can be checked\n'
                   )


  class graph_type_2:
    def root_iterator(self) -> int:
      return 0
    def child_iterator(self, node) -> int:
      return 0

  graph_conforms_to_interface, report = dfs_interface_report(graph_type_2())

  assert not graph_conforms_to_interface
  assert report == ('Type "graph_type_2" does not satisfy the interface of the depth-first search algorithm:\n'
                    '  Attribute "root_iterator" should be of the form\n'
                    '      `def root_iterator(self) -> graph_child_iterator`\n'
                    '    but it is not because its return type is "int", which is not an Iterator\n'
                    '  Attribute "child_iterator" should be of the form\n'
                    '      `def child_iterator(self, node) -> graph_child_iterator`\n'
                    '    but it is not because its return type is "int", which is not an Iterator\n'
                   )


  class graph_type_3:
    def root_iterator(self, extra_param) -> list_iterator_type:
      return iter([0])
    def child_iterator(self) -> list_iterator_type:
      return iter([0])

  graph_conforms_to_interface, report = dfs_interface_report(graph_type_3())

  assert not graph_conforms_to_interface
  assert report == ('Type "graph_type_3" does not satisfy the interface of the depth-first search algorithm:\n'
                    '  Attribute "root_iterator" should be of the form\n'
                    '      `def root_iterator(self) -> graph_child_iterator`\n'
                    '    but it is not because it must take 0 parameter but currently takes 1\n'
                    '  Attribute "child_iterator" should be of the form\n'
                    '      `def child_iterator(self, node) -> graph_child_iterator`\n'
                    '    but it is not because it must take 1 parameter but currently takes 0\n'
                   )


  class graph_type_4:
    def root_iterator(self) -> list_iterator_type:
      return iter([0])
    def child_iterator(self, node) -> list_iterator_type:
      return iter([0])

  graph_conforms_to_interface, report = dfs_interface_report(graph_type_4())

  # While the graph conforms to the interface,
  #   it does not mean it will produce meaningful results
  # In this case, calling `depth_first_search` on `graph_type_4` with a no-op visitor
  #   would result in an infinite loop
  # ...But we can't know that by inspecting the code!
  assert graph_conforms_to_interface
  assert report == ''
