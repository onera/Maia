import numpy as np

import maia.pytree as PT
from maia.pytree.yaml import parse_cgns_yaml

def test_generate_line():
  sol = PT.new_FlowSolution('FlowSolution', loc='CellCenter')
  array = PT.new_DataArray('Data', np.array([1,2,3,4], dtype=np.int64), parent=sol)

  lines = []
  parse_cgns_yaml.generate_line(array, lines)
  assert lines[0] == "Data DataArray_t I8 [1, 2, 3, 4]:"
  lines = []
  parse_cgns_yaml.generate_line(array, lines, ident=4)
  assert lines[0] == "    Data DataArray_t I8 [1, 2, 3, 4]:"

  lines = []
  parse_cgns_yaml.generate_line(sol, lines)
  assert lines == \
      ["FlowSolution FlowSolution_t:", "  GridLocation GridLocation_t 'CellCenter':", "  Data DataArray_t I8 [1, 2, 3, 4]:"]

  lines = []
  parse_cgns_yaml.generate_line(sol, lines, line_max=15)
  assert lines == \
        ["FlowSolution FlowSolution_t:", "  GridLocation GridLocation_t 'CellCenter':", \
        "  Data DataArray_t:\n    I8 : [1, 2,\n          3, 4]"]

  #Try non numpy values
  array[1] = 19.89
  sol[1] = ["All your base", "are belong to us"]
  lines = []
  parse_cgns_yaml.generate_line(sol, lines)
  assert lines == \
      ["FlowSolution FlowSolution_t ['All your base', 'are belong to us']:",\
      "  GridLocation GridLocation_t 'CellCenter':", "  Data DataArray_t 19.89:"]
      
def test_to_yaml():
  sol = PT.new_FlowSolution('FlowSolution', loc='CellCenter')
  array = PT.new_DataArray('Data', np.array([1,2,3,4], dtype=np.int64), parent=sol)

  lines = parse_cgns_yaml.to_yaml(sol)
  assert lines[0] == "GridLocation GridLocation_t 'CellCenter':"
  lines = parse_cgns_yaml.to_yaml(sol, write_root=True)
  assert lines[0] == "FlowSolution FlowSolution_t:"
