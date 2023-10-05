import re
from maia.pytree.typing import *
#TODO : merge with maia_nodes ? 

def add_part_suffix(name:str, i_proc:int, i_part:int) -> str:
  return f"{name}.P{i_proc}.N{i_part}"

def get_part_prefix(name:str)->str:
  split = name.split('.')
  assert len(split) >= 3, \
      f"Name {name} don't seem to follow part convention"
  return '.'.join(split[:-2])

def get_part_suffix(name:str) -> Tuple[int, int]:
  split = name.split('.')
  assert len(split) >= 3, \
      f"Name {name} don't seem to follow part convention"
  assert (split[-2].startswith("P") and split[-1].startswith("N")) #TODO ? use a regex ?
  return int(split[-2][1:]), int(split[-1][1:])

def add_split_suffix(name:str, count:int) -> str:
  return f"{name}.{count}"

def get_split_prefix(name:str) -> str:
  split = name.split('.')
  assert len(split) >= 2, \
      f"Name {name} don't seem to follow split convention"
  return '.'.join(split[:-1])

def get_split_suffix(name:str) -> str:
  split = name.split('.')
  assert len(split) >= 2, \
      f"Name {name} don't seem to follow split convention"
  return split[-1]

def name_intra_gc(cur_proc:int, cur_part:int, opp_proc:int, opp_part:int) -> str:
  return f"JN.P{cur_proc}.N{cur_part}.LT.P{opp_proc}.N{opp_part}"

def is_intra_gc(gc_name:str) -> bool:
  return bool(re.match(r"JN\.P\d+\.N\d+\.LT\.P\d+\.N\d+", gc_name))
