from collections import defaultdict
import hashlib

from maia.pytree.typing import *
from maia.pytree.meta import CGNSNodeNotFoundError
import maia.pytree as PT

#begin_api_export()
FULL_NAME_NODE_NAME = 'FullNameLongerThan32CharactersLi'

def _unambiguous_short_names(names:List[str]) -> List[str]:
  """ Find shorter names that are:
  - less than 32 chars
  - unambiguous (two different original names should have two different short names)
  - human-readable as much as possible """
  if len(set(names)) < len(names):
    raise RuntimeError(f"There are two siblings of the same name among {names}")
  short_names = [PT.node.short_name(n) for n in names]

  name_to_idx = defaultdict(list)
  for i,name in enumerate(short_names):
    name_to_idx[name].append(i)
  
  # Erase short name
  for _, idx_l in name_to_idx.items():
    if len(idx_l) != 1:
      for idx in idx_l:
        hash = hashlib.sha256(names[idx].encode('ascii')).hexdigest()[:8]
        new_name = short_names[idx][:23] + '.' + hash
        short_names[idx] = new_name
  return short_names

def short_name_with_hash(name:str) -> str:
  if len(name) < 32:
    return name
  hash = hashlib.sha1(name.encode('ascii')).hexdigest()[:8]
  return short_name(name)[:23] + '.' + hash

def create_full_name_children(tree:CGNSTree):
  def _create_full_name_child(node):
    if len(name := PT.get_name(node)) > 32:
      PT.new_child(node, FULL_NAME_NODE_NAME, 'Descriptor_t', name)
      PT.set_name(node, short_name_with_hash(name))

  PT.scan(tree, _create_full_name_child)

def replace_with_full_names(tree:CGNSTree):
  def _replace_with_full_name(node):
    try:
      full_name_node = PT.pop_node_from_path(node, FULL_NAME_NODE_NAME)
      node[0] = PT.get_value(full_name_node)
    except CGNSNodeNotFoundError:
      pass # Short name -> nothing to do
    
  PT.scan(tree, _replace_with_full_name)

def get_full_name(node:CGNSTree) -> str:
  if (child := PT.get_child_from_name(node, FULL_NAME_NODE_NAME)) is not None:
    return PT.get_str_value(child)
  return PT.get_name(node)

def short_name(old_name:str):
  if len(old_name) <= 32:
    return old_name
  else:
    new_name = ""
    cnt = 0
    for c in old_name:
      if c.isupper():
        cnt = 0
        new_name += c
      else:
        cnt += 1
        if cnt < 4:
          new_name += c
    return new_name[:32]

def shorten_names(t:CGNSTree, quiet:bool=False, labels_to_shorten:Optional[List[str]]=None):
  old_name = PT.get_name(t)
  can_shorten_label = labels_to_shorten is None or (PT.get_label(t) in labels_to_shorten)
  if can_shorten_label:
    new_name = short_name(old_name)
    if not quiet:
      print("WARNING: field "+old_name+" is too long. It will be renamed "+new_name)
    PT.set_name(t,new_name)
  for x in PT.get_children(t):
    shorten_names(x,quiet,labels_to_shorten)

def shorten_field_names(t:CGNSTree, quiet:bool=False):
  shorten_names(t,quiet,labels_to_shorten=["DataArray_t"])

def rename_zone(t:CGNSTree, name:str, new_name:str):
  """
  Rename a zone and its occurences in GCs
  """
  is_gc = lambda n: PT.get_label(n) in ['GridConnectivity1to1_t', 'GridConnectivity_t'] and \
                    PT.get_value(n) == name
  zones = PT.get_all_Zone_t(t)
  for zone in zones:
    if PT.get_name(zone) == name:
      PT.set_name(zone, new_name)
    for gc in PT.get_children_from_predicates(zone, ['ZoneGridConnectivity_t', is_gc]):
      PT.set_value(gc, new_name)

#end_api_export()
