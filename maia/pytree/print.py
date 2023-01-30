import sys
from   mpi4py      import MPI
import numpy       as np
import maia.pytree as PT

import maia.pytree.cgns_keywords as CGK
from maia.pytree.node.access import _np_to_string 

hierar_color = "\u001b[38;5;33m"  # Tree, Base, Zone
family_color = "\u001b[38;5;220m" # Family_t
zchild_color = "\u001b[38;5;183m" # Nodes under Zone_t

label_color  = "\u001b[38;5;246m"

bold    = "\u001b[1m"
reset   = "\u001b[0m"

def _n_nodes(tree):
  n_nodes = 0
  for node in PT.iter_nodes_from_predicate(tree, lambda n: True, explore='deep'):
    n_nodes += 1
  return n_nodes

def _render_value(value, line_prefix, verbose):

  if value is None:
    out = ""
  elif value.dtype.kind == 'S': # Strings
    str_value = _np_to_string(value)
    if value.ndim == 1:
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

# =======================================================================================
# ---------------------------------------------------------------------------------------
def print_node(node, depth, is_last_child, line_prefix, align, plabel, cst_props, out_lines):

  verbose    = cst_props['verbose']
  max_depth  = cst_props['max_depth']
  no_colors  = cst_props['no_colors']
  

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

  name  = PT.get_name(node)
  label = PT.get_label(node)
  value = PT.get_value(node, raw=True)
  sons  = PT.get_children(node)

  rendered_val = _render_value(value, line_prefix, verbose)

  # Set colors
  rendered_label = label
  rendered_name  = name
  if not no_colors:
    rendered_label = label_color + label + reset
    if label in ['CGNSTree_t', 'CGNSBase_t', 'Zone_t']:
      rendered_name = bold + hierar_color + name + reset
    elif label in ['Family_t']:
      rendered_name = bold + family_color + name + reset
    elif plabel == 'Zone_t':
      rendered_name = bold + zchild_color + name + reset

  align = (0,0)
  out_lines.append(f"{line_header}{rendered_name:<{align[0]}} {rendered_label:<{align[1]}} {rendered_val}\n")

  n_sons = len(sons)
  if depth < max_depth:

    # Old code for alignement
    if len(sons) == 0:
      name_lenght = (0,0)
    else:
      name_lenght = (max([len(s[0]) for s in sons]), max([len(s[3]) for s in sons]))

      has_subchilds = any([len(PT.get_children(s)) > 0 for s in sons[:-1]])
      if has_subchilds:
        # Tester si pas d'enfants (si le dernier a des enfants c'est pas grave)
        name_lenght = (0,0)

    for i_child, child in enumerate(sons):
      print_node(child, depth+1, i_child==n_sons-1, line_prefix, name_lenght, label, cst_props, out_lines)
  elif n_sons > 0:
    sons_w = "child" if n_sons == 1 else "children"
    out_lines.append(f"{line_prefix}╵╴╴╴ ({n_sons} {sons_w} masked)\n")

def print_tree(tree, out=sys.stdout, *, 
        verbose=False, max_depth=1000, no_colors=False, print_if=lambda n: True):
  """
  Print arborescence of a CGNSTree or a node.

  Args:
    tree      (CGNSTree) -- CGNSTree or node to be printed.
    max_depth (int)      -- Maximum tree depth for printing.
    filtering (dict)     -- Dictionary on node names and label to not be printed.

  """

  # TODO : Keeping 1 level of children with print_if is complicated, 
  # maybe update if later using graph iterators
  masked_tree = PT.shallow_copy(tree)
  for node in PT.iter_nodes_from_predicate(masked_tree, lambda n: True, explore='deep'):
    node[1] = (node[1], print_if(node))
  # Remove nodes having a False value and no child
  n_nodes_prev = _n_nodes(masked_tree)
  n_nodes      = -1 # Init loop
  while (n_nodes != n_nodes_prev):
    PT.rm_nodes_from_predicate(masked_tree, lambda n: len(PT.get_children(n)) == 0 and not n[1][1])
    n_nodes_prev = n_nodes
    n_nodes      = _n_nodes(masked_tree)
  for node in PT.iter_nodes_from_predicate(masked_tree, lambda n: True, explore='deep'):
    node[1] = node[1][0]
   
  if out not in [sys.stdout, sys.stderr]:
    no_colors = True

  print_traits = {'max_depth' : max_depth,
                  'no_colors' : no_colors,
                  'verbose'   : verbose}

  out_lines = []
  print_node(masked_tree, 0, False, "", (0,0), '', print_traits, out_lines)
  
  if isinstance(out, str):
    with open(out, 'w') as f: # Auto open file if filename is provided
      for l in out_lines:
        f.write(l)
  else:
    for l in out_lines:
      out.write(l)
# ---------------------------------------------------------------------------------------
# =======================================================================================



# =======================================================================================
# ---------------------------------------------------------------------------------------
def print_node_parallel(node, depth, pformat, plabel, last_child, comm, max_depth, filtering, filter_type):

  # Format
  if depth != 0:
    pipe   = '|'
    under  = '_'
    if len(pformat)>0:
      if pformat[-1]=='|': offset = '   '
      else               : offset = '    '
    else :                 offset = '  '
  else         :
    pipe   = ''
    under  = ''
    offset = ''

  # Value to print
  name  = PT.get_name(node)
  if type(PT.get_value(node))==np.ndarray:
    value = f'array(shape={PT.get_value(node).shape}, dtype={PT.get_value(node).dtype})'
  else:
    value = type(PT.get_value(node))
  nsons = len(PT.get_children(node))
  label = PT.get_label(node)
  
  if   nsons==0:fson ="son"
  elif nsons==1:fson ="son"
  else         :fson ="sons"

  if   filter_type=='hide': filt = lambda n,m : not(n) and not(m)
  elif filter_type=='show': filt = lambda n,m :     n  or     m

  # if name=='Base':
  #   print(filt( name  in filtering['name' ], label  in filtering['label' ]))

  if  filt( name in filtering['name' ],  label in filtering['label']):
    
    for i in range(comm.Get_size()) :
      if comm.Get_rank()==i and comm.Get_rank() not in filtering['proc']:
        if plabel in ["Zone_t", "ZoneBC_t", "ZoneGridConnectivity_t"] and not(last_child):
          pformat = pformat+offset+pipe
          print(f"{pformat}{under} [{i}] [{name} : {value}, {nsons} {fson}, {label}]")
        else :
          pformat = pformat+offset
          print(f"{pformat}{pipe}{under} [{i}] [{name} : {value}, {nsons} {fson}, {label}]")
      comm.barrier()
      
      depth +=1
      if depth<=max_depth:
        for i_child, child in enumerate(PT.get_children(node)):
          print_node_parallel(child, depth, pformat, label, i_child==nsons-1, comm, max_depth, filtering, filter_type)

  else :
    for i in range(comm.Get_size()) :
      if comm.Get_rank()==i and comm.Get_rank() not in filtering['proc']:
        if plabel in ["Zone_t", "ZoneBC_t", "ZoneGridConnectivity_t"] and not(last_child):
          pformat = pformat+offset+pipe
          print(f"{pformat}{under} [{i}] [MASKED]")
        else :
          pformat = pformat+offset
          print(f"{pformat}{pipe}{under} [{i}] [MASKED]")
      comm.barrier()

    if filter_type=='show' :
      depth +=1
      if depth<=max_depth:
        for i_child, child in enumerate(PT.get_children(node)):
          print_node_parallel(child, depth, pformat, label, i_child==nsons-1, comm, max_depth, filtering, filter_type)



def print_tree_parallel(tree, comm,  max_depth=10000, showing_filter={}, hiding_filter={}):
  '''
  Print arborescence of a distributed or partitioned CGNSTree or node.

  Args:
    tree      (CGNSTree) -- CGNSTree or node to be printed.
    comm      (MPIComm)  -- MPI Communicator.
    max_depth (int)      -- Maximum tree depth for printing.
    filtering (dict)     -- Dictionary on node names and label to not be printed.

  '''
  # Only one filter
  assert not(showing_filter!=dict() and hiding_filter!=dict())
  if  hiding_filter!=dict() : pfilter =  hiding_filter ; pfilter_type = "hide"
  if showing_filter!=dict() : pfilter = showing_filter ; pfilter_type = "show"

  if 'name'  not in pfilter.keys():  pfilter['name' ] = []
  if 'label' not in pfilter.keys():  pfilter['label'] = []
  if 'proc'  not in pfilter.keys():  pfilter['proc' ] = []

  depth   = 0
  pformat = ''
  plabel  = ''

  print_node_parallel(tree, depth, pformat, plabel, False, comm,
                      max_depth=max_depth, filtering=pfilter, filter_type=pfilter_type)
# ---------------------------------------------------------------------------------------
# =======================================================================================