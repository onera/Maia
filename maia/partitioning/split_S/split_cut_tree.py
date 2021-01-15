import itertools
import numpy as np
from copy import deepcopy
from .    import balancing_cut_tree as BCT

def get_part_volume(part_bounds):
  """Compute the number of cells in a structured partition described
  by its bounds"""
  np_array = np.asarray(part_bounds)
  return abs(np.diff(np_array).prod())

def apply_weights_to_cut_tree(cut_tree, weights):
  """ Apply a list of weights to a cut_tree and return a weigthed tree.
  A weigthed tree is a modified cut_tree (see balancing_cut_tree.py) where the number
  of leaves in terminal nodes is replaced by the weight(%) of each of these leaves.
  Weights are applied on leaves from left to right.
  tree  = [[[1], [1]], [[2]]] + weights = [.2, .1, .5, .2] -> [[[.2], [.1]], [[.5,.2]]]
  Only for 2d/3d trees, not generic
  """
  w_tree = deepcopy(cut_tree)
  shift = 0
  if BCT.depth(w_tree) == 2:
    for i1,child in enumerate(w_tree):
      len_node = child[0]
      w_tree[i1] = weights[shift:shift+len_node]
      shift += len_node
  elif BCT.depth(w_tree) == 3:
    for i1 in w_tree:
      for j, i2 in enumerate(i1):
        len_node = i2[0]
        i1[j] = weights[shift:shift+len_node]
        shift += len_node
  return w_tree

def bct_to_partitions_bounds(tree, dims, w_tree=None):
  """ Deduce the partitions bounds from a balancing cuttree and
  zone dimensions.
  If w_tree is None, all parts are of same volume (n_cell/n_part)
  Otherwise, uses w_tree to produce partition of requested size
  percentage.
  Return a list of partition bounds of size n_part, each bounds beeing
  semi open intervals [[istart, iend], [jstart, jend], [kstart, kend]]
  starting at 0.
  """
  parts = []
  if w_tree is not None:
    x_weights     = [BCT.sum_leaves(child) / BCT.sum_leaves(w_tree) for child in w_tree]
  else:
    x_weights     = [BCT.sum_leaves(child) / BCT.sum_leaves(tree) for child in tree]

  sumed_weights = [sum(x_weights[:i]) for i in range(len(tree)+1)]
  x_splits = [int(weight*dims[0]+0.5) for weight in sumed_weights]

  if len(dims) == 2:
    for i, child in enumerate(tree):
      n_y_splits = child[0]
      if w_tree is not None:
        w_sub_tree = w_tree[i]
        y_weights = [leave/BCT.sum_leaves(w_sub_tree) for leave in w_sub_tree]
      else:
        y_weights     = n_y_splits * [1. / n_y_splits]
      sumed_weights = [sum(y_weights[:i]) for i in range(n_y_splits+1)]
      y_splits = [int(weight*dims[1]+0.5) for weight in sumed_weights]
      for j in range(n_y_splits):
        part = [[x_splits[i], x_splits[i+1]], [y_splits[j], y_splits[j+1]]]
        parts.append(part)
  else: #Use recursion if 3d
    for i, i_child in enumerate(tree):
      w_sub_tree = w_tree[i] if w_tree is not None else None
      y_splits_this_col = bct_to_partitions_bounds(i_child, dims[1:], w_sub_tree)
      for split_2d in y_splits_this_col:
        part = [[x_splits[i], x_splits[i+1]]] + split_2d
        parts.append(part)

  return parts

def split_S_block(dims, n_parts, weights = None, max_it = 1E6):
  """ Top level function. Split a structured block of number of cells
  dims into n_parts partitions.
  If weights is None, all the produced partition are (more or less) of
  same volume.
  Otherwise, a list of weights of size n_parts and for which sum == 1
  can be passed to the function.
  To assure maximal respect of this constraint, the function will try
  different permutations of weight list until max_it is reached : the
  criterion to select the best permutatation if to minimize the quantity
  max(weight - ouput_weight).
  """
  cut_tree = BCT.init_cut_tree(len(dims))
  for k in range(n_parts-1):
    BCT.refine_cut_tree(cut_tree, dims)

  #If weights are none, just split this tree
  if weights is None:
    return bct_to_partitions_bounds(cut_tree, dims)
  #Otherwise, try different weights combinations
  else:
    assert(len(weights) == n_parts)
    assert(abs(sum(weights) - 1) < 1E-9)
    n_cells = np.prod(dims)

    # > Init for loop
    permutation_idx = list(range(n_parts))
    n_tries     = 0
    best_diff   = 1E50

    for permutation in itertools.permutations(enumerate(weights)):
      n_tries += 1
      permuted_weights = [p[1] for p in permutation]
      w_tree = apply_weights_to_cut_tree(cut_tree, permuted_weights)
      parts = bct_to_partitions_bounds(cut_tree, dims, w_tree)
      max_diff = max([get_part_volume(parts[k])/n_cells - permuted_weights[k]
        for k in range(n_parts)])

      if max_diff < best_diff:
        best_diff = max_diff
        best_weights = permuted_weights
        permutation_idx = [p[0] for p in permutation]
      if (n_tries >= max_it):
        break
    #print('Selected the best repartition after', n_tries, 'tries')
    w_tree = apply_weights_to_cut_tree(cut_tree, best_weights)
    parts  = bct_to_partitions_bounds(cut_tree, dims, w_tree)
    part_permutation = np.argsort(permutation_idx)
    return [parts[k] for k in part_permutation]

