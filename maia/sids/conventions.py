import re

def add_part_suffix(name, i_proc, i_part):
  return f"{name}.P{i_proc}.N{i_part}"

def get_part_prefix(name):
  split = name.split('.')
  assert len(split) >= 3, \
      f"Name {name} don't seem to follow part convention"
  return '.'.join(split[:-2])

def get_part_suffix(name):
  split = name.split('.')
  assert len(split) >= 3, \
      f"Name {name} don't seem to follow part convention"
  return int(split[-2][1:]), int(split[-1][1:])

def add_split_suffix(name, count):
  return f"{name}.{count}"

def get_split_prefix(name):
  split = name.split('.')
  assert len(split) >= 2, \
      f"Name {name} don't seem to follow split convention"
  return '.'.join(split[:-1])

def name_intra_gc(cur_proc, cur_part, opp_proc, opp_part):
  return f"JN.P{cur_proc}.N{cur_part}.LT.P{opp_proc}.N{opp_part}"

def is_intra_gc(gc_name):
  return bool(re.match(r"JN\.P\d+\.N\d+\.LT\.P\d+\.N\d+", gc_name))
