from mpi4py import MPI
import numpy as np

import maia.pytree        as PT

from maia.utils                  import py_utils
from maia.utils                  import logging as mlog
from maia.utils                  import vstride as vs
from maia.factory.dist_from_part import get_parts_per_blocks

from .import point_cloud_utils as PCU
from .import multidom_gnum     as MDG
from .import localize       as LOC
from .import closest_points as CLO

from maia.algo.interpolation_utils import Interpolator, _cell_tgt_to_vtx_tgt


def create_src_to_tgt(src_parts_per_dom,
                      tgt_parts_per_dom,
                      comm,
                      src_loc = 'CellCenter',
                      tgt_loc = 'CellCenter',
                      strategy = 'Closest',
                      loc_tolerance = 1E-6,
                      n_closest_pt = 1):
  """ Create a source to target indirection depending of the choosen strategy.

  This indirection can then be used to create an interpolator object.
  """

  assert strategy in ['LocationAndClosest', 'Location', 'Closest']

  location_out_inv = [] # Init to avoid unbound error
  closest_out_inv  = []

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

    _, closest_out_inv = CLO._mdom_closest_points(src_clouds, tgt_clouds, comm, n_pts=n_closest_pt, reverse=True, need_shift=tgt_need_shift)


  dist2weight = lambda V : vs.from_displs(V.displs, 1. / np.maximum(V.values, 1E-20))
  all_located_inv = py_utils.to_flat_list(location_out_inv)
  all_closest_inv = py_utils.to_flat_list(closest_out_inv)
  #Phase 3 : Combine Location & Closest results if both method were used
  if strategy == 'Location' or (strategy == 'LocationAndClosest' and n_tot_unlocated == 0):
    if src_loc == "CellCenter":
      tgt_in_src_gnum = [data['points_gnum_shifted'] for data in all_located_inv]
      tgt_in_src_wght = None
    elif src_loc == "Vertex":
      tgt_in_src_gnum = [data['points_gnum_shifted@VTX'] for data in all_located_inv]
      tgt_in_src_wght = [data['points_weights@VTX'] for data in all_located_inv]
        
  elif strategy == 'Closest':
    tgt_in_src_gnum = [data['tgt_in_src_shifted'] for data in all_closest_inv]
    tgt_in_src_wght = [dist2weight(data['tgt_in_src_dist2']) for data in all_closest_inv]

  else:
    tgt_in_src_gnum = []
    tgt_in_src_wght = []
    for res_loc, res_clo in zip(all_located_inv, all_closest_inv):
      clo_tgt_in_src_gnum = res_clo['tgt_in_src_shifted']
      clo_tgt_in_scr_wght = dist2weight(res_clo['tgt_in_src_dist2'])

      if src_loc=="CellCenter":
        loc_src_to_tgt_gnum = res_loc['points_gnum_shifted']
        loc_tgt_in_src_wght = vs.from_displs(loc_src_to_tgt_gnum.displs, np.ones(loc_src_to_tgt_gnum.dsize))
      elif src_loc=="Vertex":
        loc_src_to_tgt_gnum = res_loc['points_gnum_shifted@VTX']
        loc_tgt_in_src_wght = res_loc['points_weights@VTX']
        
      tgt_in_src_gnum.append(vs.concatenate([loc_src_to_tgt_gnum, clo_tgt_in_src_gnum], vs.INNER_AXIS))
      tgt_in_src_wght.append(vs.concatenate([loc_tgt_in_src_wght, clo_tgt_in_scr_wght], vs.INNER_AXIS))


  # Finalize: add src and tgt gnum (shifted and as flat data)
  _, src_lngn_per_dom = MDG.get_shifted_ln_to_gn_from_loc(src_parts_per_dom, src_loc, comm)
  _, tgt_lngn_per_dom = MDG.get_shifted_ln_to_gn_from_loc(tgt_parts_per_dom, tgt_loc, comm)

  src_to_tgt = {
    'src_gnum' : py_utils.to_flat_list(src_lngn_per_dom),
    'tgt_gnum' : py_utils.to_flat_list(tgt_lngn_per_dom),
    'target_gnum' : tgt_in_src_gnum
    }

  if tgt_in_src_wght is not None:
    src_to_tgt['target_weight'] = tgt_in_src_wght

  return src_to_tgt




def interpolate(src_tree, tgt_tree, comm, containers_name, location, **options):
  """
  Partitioned implementation of maia.algo.interpolate
  """
  # Early return if containers_name is empty
  assert isinstance(containers_name, list)
  if len(containers_name) == 0:
    return

  # Guess location of input fields using first input zone
  try:
    first_part = next(PT.iter_all_Zone_t(src_tree))
    input_loc = PT.Subset.GridLocation(PT.get_child_from_name(first_part, containers_name[0]))
  except StopIteration:
    input_loc = ''
  input_loc = comm.allreduce(input_loc, op=MPI.MAX)

  # Create interpolator
  interpolator = create_interpolator(src_tree, tgt_tree, comm, input_loc, location, **options)

  # Exchange fields
  for container_name in containers_name:
    interpolator.exchange_fields(container_name)



def create_interpolator(src_tree, tgt_tree, comm, src_location, location, **options):
  """
  Distributed implementation of maia.algo.interpolate
  """
  src_parts_per_dom = list(get_parts_per_blocks(src_tree, comm).values())
  tgt_parts_per_dom = list(get_parts_per_blocks(tgt_tree, comm).values())

  src_to_tgt = create_src_to_tgt(src_parts_per_dom, tgt_parts_per_dom, comm, src_location, location, **options)
  src_parts = py_utils.to_flat_list(src_parts_per_dom)
  tgt_parts = py_utils.to_flat_list(tgt_parts_per_dom)
  return Interpolator(src_parts, tgt_parts, src_to_tgt, src_location, location, comm)
