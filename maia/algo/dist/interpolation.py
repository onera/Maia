from mpi4py import MPI
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils    import logging as mlog
from maia.transfer import protocols as EP

from .import localize       as LOC
from .import closest_points as CLO

from .import point_cloud_utils as PCU

from maia.algo.interpolation_utils import Interpolator, _cell_tgt_to_vtx_tgt, _combine_geo_results



def get_shifted_gnum_from_loc(zones, loc):
  assert loc in ['CellCenter', 'Vertex']
  all_gnum = []
  offset = 0
  for zone in zones:
    distri_name = 'Cell' if loc == 'CellCenter' else 'Vertex'
    distri = MT.get_distribution(zone, distri_name)[1]
    gnum = np.arange(distri[0]+1+offset, distri[1]+1+offset, dtype=distri.dtype)
    if loc == 'CellCenter':
      offset += PT.Zone.n_cell(zone)
    else:
      offset += PT.Zone.n_vtx(zone)
    all_gnum.append(gnum)
  return all_gnum


def create_src_to_tgt(src_dom,
                      tgt_dom,
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

    # Use midlevel API because we need the number of vertices on the "fake" partitions
    src_parts = LOC._collect_source(src_dom, comm)
    tgt_clouds = LOC._collect_target(tgt_dom, tgt_loc, comm)
    location_out, location_out_inv = LOC._mdom_mesh_location(src_parts, tgt_clouds, \
        comm, True, loc_tolerance)

    # output is nested by domain so we need to flatten it
    n_unlocated = sum([data['unlocated_ids'].size for data in location_out])
    n_tot_unlocated = comm.allreduce(n_unlocated, op=MPI.SUM)
    if comm.Get_rank() == 0:
      mlog.stat(f"[interpolation] Number of unlocated points for Location method is {n_tot_unlocated}")


    if src_loc=="Vertex":
      # Move results of mesh location from cell to vtx
      for src_zone, src_part, data in zip(src_dom, src_parts, location_out_inv):
        fake_part = src_part[2]
        vtx_gnum = fake_part[-1]
        vtx_to_tgt, vtx_to_weight = _cell_tgt_to_vtx_tgt(data['cell_vtx'],
                                                         data['points_gnum_shifted'],
                                                         data['points_weights'].values,      
                                                         vtx_gnum.size)
        part_data = {'points_gnum_shifted@VTX' : vtx_to_tgt,
                     'points_weights@VTX'      : vtx_to_weight}
        # Careful : vertex case, the "fake partition"  vertices are not equal to the implicit distributed vertices
        # (they are local and reordered). Thus we need to move the vtx output on the distributed vtx view
        # This exchange is done with append mode because we want to merge the located data coming from different
        # partitions, and data is initially computed from cell point of view before beeing moved to vertices
        # We dont need this in Cell mode because "fake partition" cell lngn is equal to the cell distribution
        distri_vtx = MT.getDistribution(src_zone, 'Vertex')[1]
        data.update(EP.part_to_block(part_data, distri_vtx, vtx_gnum-1, comm, append=True))


  #Phase 2 -- closest point
  if strategy == 'Closest' or (strategy == 'LocationAndClosest' and n_tot_unlocated > 0):

    # We hook midlevel API to filter some target points (the one already located)
    src_clouds = [PCU.get_point_cloud(zone, comm, src_loc) for zone in src_dom]
    tgt_clouds = [PCU.get_point_cloud(zone, comm, tgt_loc) for zone in tgt_dom]
    tgt_need_shift = False
    if strategy != 'Closest':
      tgt_need_shift = True
      tgt_clouds = [PCU.extract_sub_cloud(*cloud, location_out[j]['unlocated_ids']) for j,cloud in enumerate(tgt_clouds)]

    _, closest_out_inv = CLO._mdom_closest_points(src_clouds, tgt_clouds, comm, n_pts=n_closest_pt, reverse=True, need_shift=tgt_need_shift)

  all_located_inv = location_out_inv
  all_closest_inv = closest_out_inv

  #Phase 3 : Combine Location & Closest results if both method were used
  _strategy = 'Location' if (strategy == 'LocationAndClosest' and n_tot_unlocated == 0) else strategy
  tgt_in_src_gnum, tgt_in_src_wght = _combine_geo_results(all_located_inv, all_closest_inv, _strategy, src_loc)

  # Finalize: add src and tgt gnum (shifted and as flat data)

  all_src_lngn = get_shifted_gnum_from_loc(src_dom, src_loc)
  all_tgt_lngn = get_shifted_gnum_from_loc(tgt_dom, tgt_loc)

  src_to_tgt = {
    'src_gnum' : all_src_lngn,
    'tgt_gnum' : all_tgt_lngn,
    'src_to_tgt' : tgt_in_src_gnum
    }

  if tgt_in_src_wght is not None:
    src_to_tgt['src_to_tgt_weight'] = tgt_in_src_wght

  return src_to_tgt




def interpolate(src_tree, tgt_tree, comm, containers_name, location, **options):
  """
  Distributed implementation of maia.algo.interpolate
  """
  # Early return if containers_name is empty
  assert isinstance(containers_name, list)
  if len(containers_name) == 0:
    return

  # Guess location of input fields using first input zone
  first_part = next(PT.iter_all_Zone_t(src_tree))
  input_loc = PT.Subset.GridLocation(PT.get_child_from_name(first_part, containers_name[0]))

  # Create interpolator
  interpolator = create_interpolator(src_tree, tgt_tree, comm, input_loc, location, **options)

  # Exchange fields
  for container_name in containers_name:
    interpolator.exchange_fields(container_name)



def create_interpolator(src_tree, tgt_tree, comm, src_location, tgt_location, **options):
  """
  Distributed implementation of maia.algo.create_interpolator
  """
  src_dom = PT.get_children_from_predicates(src_tree, 'CGNSBase_t/Zone_t')
  tgt_dom = PT.get_children_from_predicates(tgt_tree, 'CGNSBase_t/Zone_t')

  src_to_tgt = create_src_to_tgt(src_dom, tgt_dom, comm, src_location, tgt_location, **options)
  return Interpolator(src_dom, tgt_dom, src_to_tgt, src_location, tgt_location, comm)
