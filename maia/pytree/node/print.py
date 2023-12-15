import sys
import numpy       as np
from typing import TextIO

from maia.pytree.typing import *
import maia.pytree.cgns_keywords as CGK
import maia.pytree.walk as W

from .       import access as NA
from .create import shallow_copy

hierar_color = "\u001b[38;5;33m"  # Tree, Base, Zone
family_color = "\u001b[38;5;220m" # Family_t
zchild_color = "\u001b[38;5;183m" # Nodes under Zone_t
label_color  = "\u001b[38;5;246m"

bold  = "\u001b[1m"
reset = "\u001b[0m"

def _n_nodes(tree):
  n_nodes = 0
  for node in W.iter_nodes_from_predicate(tree, lambda n: True, explore='deep'):
    n_nodes += 1
  return n_nodes

def _render_value(value, line_prefix, verbose):

  if value is None:
    out = ""
  elif value.dtype.kind == 'S': # Strings
    str_value = NA._np_to_string(value)
    if value.ndim == 1:
      str_value = str_value.replace('\n', '\\n')
      str_value = '"' + str_value + '"'
      if len(str_value) < 20: # Short string
        out = str_value
      else:
        if verbose: # Long string, verbose mode
          splitted = [str_value[i:i+70] for i in range(0, len(str_value), 70)]
          out = ''.join(['\n' + line_prefix + f'{s}' for s in splitted])
        else: # Long string, non verbose mode
          out = f'{str_value[:20]}[...]{str_value[-5:]}'
    elif value.ndim == 2:
      str_value = [s.replace('\n', '\\n') for s in str_value]  
      if sum([len(s) for s in str_value]) < 20: # Short strings
        out = '['
        for word in str_value:
          out += f'"{word}" '
        out = out[:-1] + ']'
      else:
        if verbose: # Long strings, verbose mode
          out = ''
          line = "["
          while str_value:
            word = str_value.pop(0)
            line += f'"{word}" '
            if len(line) > 70:
              out += f'\n{line_prefix}{line}'
              line = ''
          line = line[:-1] + "]"
          out += f"\n{line_prefix}{line}" 
        else: # Long strings, non verbose mode
          out = '['
          if len(str_value) > 0:
            first = str_value[0]
            _first = f'"{first}"' if len(first) < 10 else f'"{first[:10]}[...]{first[-3:]}"'
            out += _first
          if len(str_value) > 1:
            last = str_value[-1]
            _last = f'"{last}"' if len(last) < 10 else f'"{last[:10]}[...]{last[-3:]}"'
            out += " ... " + _last
          out += ']'

  else: #Data arrays
    cg_dtype = CGK.dtype_to_cgns[value.dtype]
    if value.size <= 9:
      array_str = np.array2string(value).replace('\n', '')
      out = f"{cg_dtype} {array_str}"
    else:
      if verbose:
        out = np.array2string(value, prefix=line_prefix)
        out = out.replace(' '*len(line_prefix), line_prefix)
        out = f"{cg_dtype} {value.shape}\n{line_prefix}{out}"
      else:
        out = f"{cg_dtype} {value.shape}"

  return out


def print_node(node, depth, is_last_child, line_prefix, plabel, cst_props, out_lines):

  verbose    = cst_props['verbose']
  max_depth  = cst_props['max_depth']
  colors     = cst_props['colors']
  

  # Format : line_header has the "├───" stuff, line prefix has just indentation
  if is_last_child:
    line_header = line_prefix+"└───"
    line_prefix = line_prefix+"    "
  else:
    line_header = line_prefix+"├───"
    line_prefix = line_prefix+"│   "
  if depth == 0:
    line_header = ""
    line_prefix = ""

  name  = NA.get_name(node)
  label = NA.get_label(node)
  value = NA.get_value(node, raw=True)
  sons  = NA.get_children(node)

  rendered_val = _render_value(value, line_prefix, verbose)

  # Set colors
  rendered_label = label
  rendered_name  = name
  if colors:
    rendered_label = label_color + label + reset
    if label in ['CGNSTree_t', 'CGNSBase_t', 'Zone_t']:
      rendered_name = bold + hierar_color + name + reset
    elif label in ['Family_t']:
      rendered_name = bold + family_color + name + reset
    elif plabel == 'Zone_t':
      rendered_name = bold + zchild_color + name + reset

  out_lines.append(f"{line_header}{rendered_name} {rendered_label} {rendered_val}\n")

  n_sons = len(sons)
  if depth < max_depth:
    for i_child, child in enumerate(sons):
      print_node(child, depth+1, i_child==n_sons-1, line_prefix, label, cst_props, out_lines)
  elif n_sons > 0:
    sons_w = "child" if n_sons == 1 else "children"
    out_lines.append(f"{line_prefix}╵╴╴╴ ({n_sons} {sons_w} masked)\n")

def to_string(tree:CGNSTree, 
              *, 
              verbose:bool=False,
              max_depth:int=1000,
              colors:bool=False,
              print_if:Callable[[CGNSTree], bool]=lambda n: True):
  # TODO : Keeping 1 level of children with print_if is complicated, 
  # maybe update if later using graph iterators
  masked_tree = shallow_copy(tree)
  for node in W.iter_nodes_from_predicate(masked_tree, lambda n: True, explore='deep'):
    node[1] = (node[1], print_if(node))
  # Remove nodes having a False value and no child
  n_nodes_prev = _n_nodes(masked_tree)
  n_nodes      = -1 # Init loop
  while (n_nodes != n_nodes_prev):
    W.rm_nodes_from_predicate(masked_tree, lambda n: len(NA.get_children(n)) == 0 and not n[1][1])
    n_nodes_prev = n_nodes
    n_nodes      = _n_nodes(masked_tree)
  for node in W.iter_nodes_from_predicate(masked_tree, lambda n: True, explore='deep'):
    node[1] = node[1][0]
   

  print_traits = {'max_depth' : max_depth,
                  'colors'    : colors,
                  'verbose'   : verbose}

  out_lines = []
  print_node(masked_tree, 0, False, "", "", print_traits, out_lines)
  return out_lines

def print_tree(tree:CGNSTree, 
               out:TextIO =sys.stdout,
               *, 
               verbose:bool=False,
               max_depth:int=1000,
               colors:bool=True,
               print_if:Callable[[CGNSTree], bool]=lambda n: True):
  """
  print_tree(tree, out=sys.stdout, **kwargs)

  Display the arborescence of a CGNSTree

  By default, display will be done in the standard output. It is possible to print
  in a file by setting out to either a string or a python file object:

  ::

    PT.print_tree(tree)                # Print tree in stdout
    PT.print_tree(tree, 'tree.txt')    # Print tree in a tree.txt file
    with open('out.txt', 'w') as f:
        PT.print_tree(tree, f)         # Print tree in f

  In addition, the following kwargs can be used to parametrize the output:

  - ``colors`` (bool) -- If False, disable the colors in rendered text. Default to ``True``
    for stdout or stderr output, and ``False`` for file output.
  - ``verbose`` (bool) -- If True, data arrays are displayed (using numpy display settings).
    Otherwise, small arrays are printed, and large arrays only show their size.
    Defaults to ``False``.
  - ``max_depth`` (int) -- Stop printing once max_depth is reached. By default, displays all nodes.
  - ``print_if`` (callable) -- If set, it must be a function with the signature : ``f(n:CGNSTree) -> bool``.
    The branches not leading to a node that evaluates to ``True`` are removed from displayed tree.




  Args:
    tree      (CGNSTree) : Node to be printed
    out       (TextIO)   : Where to print

  See also:
    To display a tree in custom output, use function :func:`to_string`
    which takes the same parameters, except ``out``, and return the string
    representation of the tree.

  Examples:
    >>> tree = PT.yaml.parse_yaml_cgns.to_node('''
    ... Base CGNSBase_t [3,3]:
    ...   Wall Family_t:
    ...   MyZone Zone_t I4 [[16,6,0]]:
    ...     GridCoordinates GridCoordinates_t:
    ...       CoordinateX DataArray_t [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]:
    ... ''')
    >>> PT.print_tree(tree)
    Base CGNSBase_t I4 [3 3]
    ├───Wall Family_t 
    └───MyZone Zone_t I4 [[16  6  0]]
        └───GridCoordinates GridCoordinates_t 
            └───CoordinateX DataArray_t I4 (16,)
    >>> PT.print_tree(tree, max_depth=1)
    Base CGNSBase_t I4 [3 3]
    ├───Wall Family_t 
    └───MyZone Zone_t I4 [[16  6  0]]
        ╵╴╴╴ (1 child masked)
    
    On the next example, notice that the branches not leading to 'CoordinateX' are
    not printed; and, because of verbose option, all the coordinate array is displayed.

    >>> PT.print_tree(tree, verbose=True, 
    ...               print_if=lambda n : PT.get_name(n) == 'CoordinateX')
    Base CGNSBase_t I4 [3 3]
    └───MyZone Zone_t I4 [[16  6  0]]
        └───GridCoordinates GridCoordinates_t 
            └───CoordinateX DataArray_t I4 (16,)
                [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3]
  """

  if out not in [sys.stdout, sys.stderr]:
    colors = False

  out_lines = to_string(tree, verbose=verbose, max_depth=max_depth, 
                        colors=colors, print_if=print_if)
  
  if isinstance(out, str):
    with open(out, 'w') as f: # Auto open file if filename is provided
      for l in out_lines:
        f.write(l)
  else:
    for l in out_lines:
      out.write(l)

