import maia.pytree as PT

from maia.utils import par_utils

from maia.typing        import *
from maia.pytree.typing import Predicate

def _gather_containers_name(all_nodes:List[List[CGNSTree]], kind:str, comm:MPIComm) -> List[str]:
  # Base implem of gather_containers_name
  all_loc_names = [{PT.get_name(node) for node in nodes} for nodes in all_nodes]
  if kind == 'all':
    glob_cnt = par_utils.sets_intersection(all_loc_names, comm)
  elif kind == 'any':
    glob_cnt = par_utils.sets_union(all_loc_names, comm)
  else:
    raise ValueError('Unsupported kind')

  return sorted(glob_cnt) if glob_cnt is not None else []

def gather_containers_name(zones:List[CGNSPartTree], pred:Predicate, kind:str, comm:MPIComm) -> List[str]:
  """
  Get the name of children satisfying ``pred`` predicate: 
  - on all zones if kind == 'all'
  - on any zone if kind == 'any'
  """
  return _gather_containers_name([PT.get_children_from_predicate(zone, pred) for zone in zones],
                                 kind,
                                 comm)