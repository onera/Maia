import inspect
from collections.abc import Iterator


def _dfs_attr_report(g, attr_name, sig):
  report = ''

  attr = getattr(g, attr_name, None)
  if not attr:
    report += 'there is no such attribute\n'

  elif not hasattr(attr, '__call__'):
    report += 'it is not a method\n'
  else:
    attr_sig = inspect.signature(attr)
    attr_params = attr_sig.parameters
    attr_return_type = attr_sig.return_annotation
    if attr_return_type == inspect.Signature.empty:
      report += 'it must have a return annotation so that the return type can be checked\n'
    elif not issubclass(attr_return_type, Iterator):
      report += f'its return type is "{attr_return_type.__name__}", which is not an Iterator\n'
    else:
      if attr_name == 'root_iterator' and len(attr_params) != 0:
        report += f'it must take 0 parameter but currently takes {len(attr_params)}\n'
      elif attr_name == 'child_iterator' and len(attr_params) != 1:
        report += f'it must take 1 parameter but currently takes {len(attr_params)}\n'

  if report != '':
    return f'  Attribute "{attr_name}" should be of the form\n      `{sig}`\n    but it is not because ' + report
  else:
    return ''


def dfs_interface_report(g):
  """ Tells if `g` conforms to the depth-first search interface, and if not, why.

  To be conforming, `g` has to have:
    - a `root_iterator(self)` method that returns the roots of the graph.
    - a `child_iterator(self, n)` method that returns the children of node `n` in the graph.
  Both methods should return object that are iterators over nodes of the graph.
  """
  report = ''

  # check has `roots` and `children`
  expected_attrs_and_sigs = {
      'root_iterator': 'def root_iterator(self) -> graph_child_iterator',
      'child_iterator': 'def child_iterator(self, node) -> graph_child_iterator',
  }
  for attr_name, sig in expected_attrs_and_sigs.items():
    report += _dfs_attr_report(g, attr_name, sig)

  if report != '':
    return False, f'Type "{type(g).__name__}" does not satisfy the interface of the depth-first search algorithm:\n'  + report
  else:
    return True, ''


