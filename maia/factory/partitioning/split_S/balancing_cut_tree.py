r"""
This module implements a balancing cut tree structure
and some algorithm, detailed in https://doi.org/10.1016/j.jpdc.2019.12.010,
useful for structured blocks partitioning.

Trees are represented by nested list. A l levels tree is a (l-1) nested list
   and at a given level k, we find:
   - either a list of sub nodes (if k < l)
   - either an interger value   (if k == l). This value represent the number 
   of terminal leaves in this branch of the tree.
       _ o _
     /   |   \          For example, this tree is represented by
    /    |    \         T = [ [[2]], [[1], [3]], [[1], [2]] ]
   a     b     c        We have len(tree) = 3, meaning that 3 branches start
   |    /\     |\       from top level node (let call them a, b, c); 
   |   /  \    | \      since branch b leads to two node, len(b:=T[1]) = 2;
   o  o    o   o  o     unlikely, len(a:=T[0]) = 1 because branch a continues
  /|  |   /|\  |  |\    in one node. Going deeper in tree, we are now almost at 
 / |  |  / | \ |  | \   terminal level : nodes have child who are terminal leaves.
o  o  o  o o  oo  o  o  Thus, we juste store in the structure the number of this leaves,
                        that is eg a[0] = T[0][0] = [2] ; b[0] = T[1][0] = [1]

In the context of blocks partitioning, trees represent the number of cuts of the block,
each level beeing associated to a spatial dimension.
    ______________
   /   /   /     /|  Previous tree would thus represent a 3d block (because deepth(L) = 3)
  /   /___/     / |  which is split in 3 parts along the x axis :
 /   /   /_____/ /|  - first part is not splitted in y axis, but it splitted in two along
/___/___/____ /|/ |    z axis
|   |   |    | | /   - second part and third are both splitted in two along y axis, but
|___|   |    | |/ k    the two resulting chunks of the second part are respectively splitted
|   |   |    | /  ^    in 1 and 3 along z axis, whereas the two resulting chunks of the third
|___|___|____|/   |    part are respectively splitted in 1 and 2 along z axis.
                  --> i  
Note that the balancing cut tree only indicates the number of cuts. Position of cuts 
are computed using the number of cells in each direction + some weight information, and
are also taken into account when cut tree are refined (ie when new partition are insered)
"""

def init_cut_tree(n_dims):
  """Create an empty tree (1 leave) with the requested number of levels"""
  tree = 1
  for i in range(n_dims):
    tree = [tree]
  return tree

def reset_sub_tree(node):
  """ From a given node, traverse (recursively) childrens to set
  their number of leaves to 1 """
  if depth(node) > 1:
    for child in node:
      reset_sub_tree(child)
  else:
    node[0] = 1

def depth(l):
  """Return the depth of a nested list and 0 if l is not a list"""
  result = isinstance(l, list) and max(map(depth, l))+1
  return result if result else 0

def sum_leaves(node):
  """Sums recursively the leaves from a given node"""
  if isinstance(node, list):
    n_leaves = 0
    for child in node:
      n_leaves += sum_leaves(child)
  else:
    n_leaves = node
  return n_leaves

def weight(node, dims):
  """Compute the weight associated to a node, defined as the
  number of childs of this node (non recursively count)
  divided by the dimension corresponding to the level of the node
  """
  n_dims = len(dims)
  dep    = depth(node)
  if dep > 1:
    return len(node) / dims[n_dims - dep]
  elif dep == 1:
    return node[0] / dims[n_dims - dep]
  else:
    raise ValueError('Can not compute weight of last level node')

def child_with_least_leaves(node):
  """ Compare the children of a given node and return the
  one having the least leaves. If tie, return first found (leftmost)"""
  candidate = node[0]
  for child in node:
    if sum_leaves(child) < sum_leaves(candidate):
      candidate = child
  return candidate

def select_insertion(starting_node, insersion_node, dims):
  """ Select the position in tree where a new node will be insered
  ie the node for which weight is minimal
  """
  if weight(starting_node, dims) < weight(insersion_node, dims):
    insersion_node = starting_node
  if depth(starting_node) > 1:
    next_node = child_with_least_leaves(starting_node)
    insersion_node = select_insertion(next_node, insersion_node, dims)
  return insersion_node


def insert_child_at(node, dims):
  """ Insert a new child a the given position. If node is
  not base level, its subtree is reseted and recomputed nd in order
  to have a well balanced tree
  """
  n_leaves_ini = sum_leaves(node)
  if depth(node) == 0:
    raise ValueError("invalid node in insertChildAt")
  elif depth(node) == 1:
    node[0] += 1
  else:
    for child in node:
      reset_sub_tree(child)
    #Create new node with good dim
    newNode = init_cut_tree(depth(node)-1)
    node.append(newNode)

    while (sum_leaves(node) != n_leaves_ini+1):
      nextins = select_insertion(node, node, dims)
      insert_child_at(nextins, dims)

def refine_cut_tree(cut_tree, dims):
  """Refine a cut_tree ie select an insertion position
  add a new child at this position
  """
  ins = select_insertion(cut_tree, cut_tree, dims)
  insert_child_at(ins, dims)

