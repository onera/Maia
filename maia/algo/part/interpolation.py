from mpi4py import MPI
from collections import defaultdict

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.typing import *

from maia.utils                  import py_utils, par_utils
from maia.utils                  import logging as mlog
from maia.utils.ndarray.vstride  import VStrideArray
from maia.factory.dist_from_part import get_parts_per_blocks
from maia.pytree.maia.check_tree import check_cgns_part_tree

from .import point_cloud_utils as PCU
from .import multidom_gnum     as MDG
from .import localize       as LOC
from .import closest_points as CLO

from .utils import gather_containers_name

from maia.algo.interpolation_utils import Interpolator, _cell_tgt_to_vtx_tgt, _combine_geo_results

def create_src_to_tgt(src_parts_per_dom:List[List[CGNSPartTree]],
                      tgt_parts_per_dom:List[List[CGNSPartTree]],
                      comm:MPIComm,
                      src_loc:Literal['CellCenter', 'Vertex'] = 'CellCenter',
                      tgt_loc:Literal['CellCenter', 'Vertex'] = 'CellCenter',
                      strategy:str = 'Closest',
                      loc_tolerance:float = 1E-6,
                      n_closest_pt:int = 1):
  """ Create a source to target indirection depending of the choosen strategy.

  This indirection can then be used to create an interpolator object.
  """

  assert strategy in ['LocationAndClosest', 'Location', 'Closest']

  location_out_inv:List[List[Dict[str, VStrideArray]]] = [] # Init to avoid unbound error
  closest_out_inv:List[List[Dict[str, VStrideArray]]]  = []

  #Phase 1 -- localisation
  if strategy != 'Closest':

    location_out, location_out_inv = LOC._localize_points(src_parts_per_dom, tgt_parts_per_dom, \
        tgt_loc, comm, True, loc_tolerance)

    n_unlocated = sum([sum([data['unlocated_ids'].size for data in domain]) \
                       for domain in location_out])
    n_tot_unlocated = comm.allreduce(n_unlocated, op=MPI.SUM)
    if comm.Get_rank() == 0:
      mlog.stat(f"[interpolation] Number of unlocated points for Location method is {n_tot_unlocated}")

    if src_loc=="Vertex":
      # Move results of mesh location from cell to vtx
      for src_parts, domain_location_out_inv in zip(src_parts_per_dom, location_out_inv):
        for part, data in zip(src_parts, domain_location_out_inv):
          vtx_to_tgt, vtx_to_weight = _cell_tgt_to_vtx_tgt(data['cell_vtx'],
                                                           data['points_gnum_shifted'],
                                                           data['points_weights'].values,      
                                                           PT.Zone.n_vtx(part))
          data['points_gnum_shifted@VTX'] = vtx_to_tgt
          data['points_weights@VTX'] = vtx_to_weight


  #Phase 2 -- closest point
  if strategy == 'Closest' or (strategy == 'LocationAndClosest' and n_tot_unlocated > 0):

    # We hook midlevel API to filter some target points (the one already located)
    src_clouds = [[PCU.get_point_cloud(part, src_loc) for part in src_parts] \
      for src_parts in src_parts_per_dom]
    tgt_clouds = [[PCU.get_point_cloud(part, tgt_loc) for part in tgt_parts] \
      for tgt_parts in tgt_parts_per_dom]
    tgt_need_shift = False
    if strategy != 'Closest':
      tgt_need_shift = True
      tgt_clouds = [[PCU.extract_sub_cloud(*cloud, location_out[i][j]['unlocated_ids']) for j,cloud in enumerate(clouds)] \
        for i, clouds in enumerate(tgt_clouds)]

    _, closest_out_inv = CLO._mdom_closest_points(src_clouds, tgt_clouds, comm, True, n_pts=n_closest_pt, need_shift=tgt_need_shift)


  all_located_inv = py_utils.to_flat_list(location_out_inv)
  all_closest_inv = py_utils.to_flat_list(closest_out_inv)

  #Phase 3 : Combine Location & Closest results if both method were used
  _strategy = 'Location' if (strategy == 'LocationAndClosest' and n_tot_unlocated == 0) else strategy
  tgt_in_src_gnum, tgt_in_src_wght = _combine_geo_results(all_located_inv, all_closest_inv, _strategy, src_loc)

  # Finalize: add src and tgt gnum (shifted and as flat data)
  _, src_lngn_per_dom = MDG.get_shifted_ln_to_gn_from_loc(src_parts_per_dom, src_loc, comm)
  _, tgt_lngn_per_dom = MDG.get_shifted_ln_to_gn_from_loc(tgt_parts_per_dom, tgt_loc, comm)

  src_to_tgt = {
    'src_gnum' : py_utils.to_flat_list(src_lngn_per_dom),
    'tgt_gnum' : py_utils.to_flat_list(tgt_lngn_per_dom),
    'src_to_tgt' : tgt_in_src_gnum
    }

  if tgt_in_src_wght is not None:
    src_to_tgt['src_to_tgt_weight'] = tgt_in_src_wght

  return src_to_tgt




def interpolate(src_tree:CGNSPartTree,
                tgt_tree:CGNSPartTree,
                comm:MPIComm,
                containers_name:Union[List[str], Literal['ALL']],
                location:Literal['CellCenter', 'Vertex'],
                **options) -> None:
  """
  Partitioned implementation of maia.algo.interpolate
  """
  check_cgns_part_tree(src_tree)
  check_cgns_part_tree(tgt_tree)

  loc_to_containers_name = defaultdict(list)
  # Guess location of input fields using first input zone
  if containers_name == 'ALL':
    for loc, pred in zip(['Vertex', 'CellCenter'], [MT.pred.FULL_CTN_VTX, MT.pred.FULL_CTN_CELL]):
      loc_to_containers_name[loc] = gather_containers_name(PT.get_all_Zone_t(src_tree), pred, 'all', comm)
  else:
    try:
      first_part = next(PT.iter_all_Zone_t(src_tree))
      input_locs = [PT.Container.GridLocation(PT.find_child_from_name(first_part, name)) for name in containers_name]
    except StopIteration:
      input_locs = ['' for name in containers_name]
    input_locs = comm.allreduce(input_locs, op=MPI.MAX)
    for loc, name in zip(input_locs, containers_name):
      loc_to_containers_name[loc].append(name)

  if (lc:=len(loc_to_containers_name)) > 1:
    mlog.info(f"Requested containers have different GridLocation. Interpolation process will be done in {lc} steps")

  for input_loc, loc_containers_name in loc_to_containers_name.items():

    _input_loc:Literal['Vertex', 'CellCenter'] = input_loc #type:ignore[assignment]
    # Create interpolator
    interpolator = create_interpolator(src_tree, tgt_tree, comm, _input_loc, location, **options)
    # Exchange fields
    for container_name in loc_containers_name:
      interpolator.exchange_fields(container_name)



def create_interpolator(src_tree:CGNSPartTree,
                        tgt_tree:CGNSPartTree,
                        comm:MPIComm,
                        src_location:Literal['CellCenter', 'Vertex'],
                        tgt_location:Literal['CellCenter', 'Vertex'],
                        **options) -> Interpolator:
  """
  Partitioned implementation of maia.algo.interpolate
  """
  assert src_location in ['CellCenter', 'Vertex']
  check_cgns_part_tree(src_tree)
  check_cgns_part_tree(tgt_tree)
  src_parts_per_dom = list(get_parts_per_blocks(src_tree, comm).values())
  tgt_parts_per_dom = list(get_parts_per_blocks(tgt_tree, comm).values())

  src_to_tgt = create_src_to_tgt(src_parts_per_dom, tgt_parts_per_dom, comm, src_location, tgt_location, **options)
  src_parts = py_utils.to_flat_list(src_parts_per_dom)
  tgt_parts = py_utils.to_flat_list(tgt_parts_per_dom)
  return Interpolator(src_parts, tgt_parts, src_to_tgt, src_location, tgt_location, comm)
