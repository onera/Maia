from mpi4py import MPI
import numpy as np

import Pypdm.Pypdm as PDM

import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia                        import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.utils                  import py_utils, np_utils
from maia.transfer               import utils as te_utils
from maia.factory.dist_from_part import get_parts_per_blocks

from .import point_cloud_utils as PCU
from .import localize as LOC
from .import closest_points as CLO

class Interpolator:
  """
  """
  def __init__(self, src_parts_per_dom, tgt_parts_per_dom, src_to_tgt, output_loc, comm):
    self.src_parts = list()
    self.tgt_parts = list()
    all_src_lngn = []
    for i_domain, src_parts in enumerate(src_parts_per_dom):
      for i_part, src_part in enumerate(src_parts):
        lngn = I.getVal(MT.getGlobalNumbering(src_part, 'Cell')).astype(pdm_gnum_dtype, copy=False)
        all_src_lngn.append(lngn)
        self.src_parts.append(src_part)

    all_cloud_lngn = []
    for i_domain, tgt_parts in enumerate(tgt_parts_per_dom):
      for i_part, tgt_part in enumerate(tgt_parts):
        if output_loc == 'Vertex':
          lngn = I.getVal(MT.getGlobalNumbering(tgt_part, 'Vertex')).astype(pdm_gnum_dtype, copy=False)
        else:
          lngn = I.getVal(MT.getGlobalNumbering(tgt_part, 'Cell')).astype(pdm_gnum_dtype, copy=False)
        all_cloud_lngn.append(lngn)
        self.tgt_parts.append(tgt_part)

    _src_to_tgt_idx = [data['target_idx'] for data in src_to_tgt]
    _src_to_tgt     = [data['target'] for data in src_to_tgt]
    self.PTP = PDM.PartToPart(comm,
                              all_src_lngn,
                              all_cloud_lngn,
                              _src_to_tgt_idx,
                              _src_to_tgt)

    self.referenced_nums = self.PTP.get_referenced_lnum2()
    self.sending_gnums = self.PTP.get_gnum1_come_from()
    self.output_loc = output_loc

    # Send distances to targets partitions
    _dist = [data['dist2'] for data in src_to_tgt]
    request = self.PTP.iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                             PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART1_TO_PART2,
                             _dist)
    _, self.tgt_dist = self.PTP.wait(request)


  def _reduce_single_val(self, i_part, data):
    """
    A basic reduce function who take the first received value for each target
    """
    come_from_idx = self.sending_gnums[i_part]['come_from_idx']
    assert (np.diff(come_from_idx) == 1).all()
    return data

  def _reduce_mean_dist(self, i_part, data):
    """
    Compute a weighted mean of the received values. Weights are the inverse of the squared distance from each source
    """
    come_from_idx = self.sending_gnums[i_part]['come_from_idx']
    n_recv = come_from_idx[1] #We assert this one to be the same for each located gn
    assert (np.diff(come_from_idx) == n_recv).all()
    n_reduced = self.referenced_nums[i_part].size

    reduced_data = np.zeros(n_reduced, float)
    factor = np.zeros(n_reduced, float)
    dist_threshold = np.maximum(self.tgt_dist[i_part], 1E-20)
    for i in range(n_recv):
      reduced_data += (1./dist_threshold[i::n_recv]) * data[i::n_recv]
      factor += (1./dist_threshold[i::n_recv])
    reduced_data /= factor
    return reduced_data


  def exchange_fields(self, container_name, reduce_func=_reduce_single_val):

    #Check that solutions are known on each source partition
    fields_per_part = list()
    for src_part in self.src_parts:
      container = I.getNodeFromPath(src_part, container_name)
      assert PT.Subset.GridLocation(container) == 'CellCenter' #Only cell center sol supported for now
      fields_name = sorted([I.getName(array) for array in I.getNodesFromType1(container, 'DataArray_t')])
    fields_per_part.append(fields_name)
    assert fields_per_part.count(fields_per_part[0]) == len(fields_per_part)

    #Cleanup target partitions
    for tgt_part in self.tgt_parts:
      I._rmNodesByName(tgt_part, container_name)
      fs = I.createUniqueChild(tgt_part, container_name, 'FlowSolution_t')
      I.newGridLocation(self.output_loc, fs)

    #Collect src sol
    src_field_dic = dict()
    for field_name in fields_per_part[0]:
      field_path = container_name + '/' + field_name
      src_field_dic[field_name] = [I.getNodeFromPath(part, field_path)[1] for part in self.src_parts]

    #Exchange
    for field_name, src_sol in src_field_dic.items():
      request = self.PTP.iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                               PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART1,
                               src_sol)
      strides, lnp_part_data = self.PTP.wait(request)

      for i_part, tgt_part in enumerate(self.tgt_parts):
        fs = I.getNodeFromPath(tgt_part, container_name)
        data_size = PT.Zone.n_cell(tgt_part) if self.output_loc == 'CellCenter' else PT.Zone.n_vtx(tgt_part)
        data = np.nan * np.ones(data_size)
        reduced_data = reduce_func(self, i_part, lnp_part_data[i_part])
        data[self.referenced_nums[i_part]-1] = reduced_data #Use referenced ids to erase default value
        if PT.Zone.Type(tgt_part) == 'Unstructured':
          I.createUniqueChild(fs, field_name, 'DataArray_t', data)
        else:
          shape = PT.Zone.CellSize(tgt_part) if self.output_loc == 'CellCenter' else PT.Zone.VertexSize(tgt_part)
          I.createUniqueChild(fs, field_name, 'DataArray_t', data.reshape(shape, order='F'))


def create_src_to_tgt(src_parts_per_dom,
                      tgt_parts_per_dom,
                      comm,
                      location = 'CellCenter',
                      strategy = 'LocationAndClosest',
                      loc_tolerance = 1E-6,
                      order = 0):
  """
  """
  n_dom_src = len(src_parts_per_dom)
  n_dom_tgt = len(tgt_parts_per_dom)

  assert n_dom_src == n_dom_tgt == 1
  assert strategy in ['LocationAndClosest', 'Location', 'Closest']

  n_part_src = len(src_parts_per_dom[0])
  n_part_tgt = len(tgt_parts_per_dom[0])

  #Phase 1 -- localisation
  if strategy != 'Closest':
    location_out, location_out_inv = LOC._localize_points(src_parts_per_dom, tgt_parts_per_dom, \
        location, comm, True, loc_tolerance)

    all_unlocated = [data['unlocated_ids'] for data in location_out[0]]
    all_located_inv = location_out_inv[0]
    n_unlocated = sum([t.size for t in all_unlocated])
    n_tot_unlocated = comm.allreduce(n_unlocated, op=MPI.SUM)
    if(comm.Get_rank() == 0):
      print(" n_tot_unlocated = ", n_tot_unlocated )


  all_closest_inv = list()
  if strategy == 'Closest' or (strategy == 'LocationAndClosest' and n_tot_unlocated > 0):

    # > Setup source for closest point
    src_clouds = []
    for i_domain, src_part_zones in enumerate(src_parts_per_dom):
      for i_part, src_part in enumerate(src_part_zones):
        src_clouds.append(PCU.get_point_cloud(src_part, 'CellCenter'))

    # > Setup target for closest point
    tgt_clouds = []
    if strategy == 'Closest':
      for i_domain, tgt_part_zones in enumerate(tgt_parts_per_dom):
        for i_part, tgt_part in enumerate(tgt_part_zones):
          tgt_clouds.append(PCU.get_point_cloud(tgt_part, location))
    else:
      # > If we previously did a mesh location, we only treat unlocated points : create a sub global numbering
      for i_domain, tgt_part_zones in enumerate(tgt_parts_per_dom):
        for i_part, tgt_part in enumerate(tgt_part_zones):
          indices = all_unlocated[i_part] #One domain so OK
          tgt_cloud = PCU.get_point_cloud(tgt_part, location)
          sub_cloud = PCU.extract_sub_cloud(*tgt_cloud, indices)
          tgt_clouds.append(sub_cloud)
      all_extracted_lngn = [sub_cloud[1] for sub_cloud in tgt_clouds]
      all_sub_lngn = PCU.create_sub_numbering(all_extracted_lngn, comm) #This one is collective
      tgt_clouds = [(tgt_cloud[0], sub_lngn) for tgt_cloud, sub_lngn in zip(tgt_clouds, all_sub_lngn)]

    all_closest, all_closest_inv = CLO._closest_points(src_clouds, tgt_clouds, comm, 1, reverse=True)

    #If we worked on sub gnum, we must go back to original numbering
    if strategy != 'Closest':
      gnum_to_transform = [results["tgt_in_src"] for results in all_closest_inv]
      PDM.transform_to_parent_gnum(gnum_to_transform, all_sub_lngn, all_extracted_lngn, comm)

  #Combine Location & Closest results if both method were used
  if strategy == 'Location' or (strategy == 'LocationAndClosest' and n_tot_unlocated == 0):
    src_to_tgt = [{'target_idx' : data['elt_pts_inside_idx'],
                   'target'     : data['points_gnum'],
                   'dist2'      : data['points_dist2']} for data in all_located_inv]
  elif strategy == 'Closest':
    src_to_tgt = [{'target_idx' : data['tgt_in_src_idx'],
                   'target'     : data['tgt_in_src'],
                   'dist2'      : data['tgt_in_src_dist2']} for data in all_closest_inv]
  else:
    src_to_tgt = []
    for res_loc, res_clo in zip(all_located_inv, all_closest_inv):
      tgt_in_src_idx, tgt_in_src = np_utils.jagged_merge(res_loc['elt_pts_inside_idx'], res_loc['points_gnum'], \
                                                         res_clo['tgt_in_src_idx'], res_clo['tgt_in_src'])
      tgt_in_src_idx, tgt_to_dis = np_utils.jagged_merge(res_loc['elt_pts_inside_idx'], res_loc['points_dist2'], \
                                                         res_clo['tgt_in_src_idx'], res_clo['tgt_in_src_dist2'])
      src_to_tgt.append({'target_idx' :tgt_in_src_idx, 'target' :tgt_in_src, 'dist2' :tgt_to_dis})
  
  return src_to_tgt



def interpolate_from_parts_per_dom(src_parts_per_dom, tgt_parts_per_dom, comm, containers_name, location, **options):
  """
  Low level interface for interpolation
  Input are a list of partitioned zones for each src domain, and a list of partitioned zone for each tgt
  domain. Lists mush be cohérent across procs, ie we must have an empty entry if a proc does not know a domain.

  containers_name is the list of FlowSolution containers to be interpolated
  location is the output location (CellCenter or Vertex); input location must be CellCenter
  **options are passed to interpolator creationg function, see create_src_to_tgt
  """
  src_to_tgt = create_src_to_tgt(src_parts_per_dom, tgt_parts_per_dom, comm, location, **options)

  interpolator = Interpolator(src_parts_per_dom, tgt_parts_per_dom, src_to_tgt, location, comm)
  for container_name in containers_name:
    interpolator.exchange_fields(container_name)

def interpolate_from_part_trees(src_tree, tgt_tree, comm, containers_name, location, **options):
  """Interpolate fields between two partitionned trees.

  For now, interpolation is limited to lowest order: target points take the value of the
  closest point (or their englobing cell, depending of choosed options) in the source mesh.
  Interpolation strategy can be controled thought the options kwargs:

  - ``strategy`` (default = 'LocationAndClosest') -- control interpolation method

    - 'Closest' : Target points take the value of the closest source cell center.
    - 'Location' : Target points take the value of the cell in which they are located.
      Unlocated points have take a ``NaN`` value.
    - 'LocationAndClosest' : Use 'Location' method and then 'ClosestPoint' method
      for the unlocated points.

  - ``loc_tolerance`` (default = 1E-6) -- Geometric tolerance for Location method.

  Important:
    - Source fields must be located at CellCenter.
    - Source tree must be unstructured and have a ngon connectivity.
    - Partitions must come from a single initial domain on both source and target tree.

  Args:
    src_tree (CGNSTree): Source tree, partitionned. Only U-NGon connectivities are managed.
    tgt_tree (CGNSTree): Target tree, partitionned. Structured or U-NGon connectivities are managed.
    comm       (MPIComm): MPI communicator
    containers_name (list of str) : List of the names of the source FlowSolution_t nodes to transfer.
    location ({'CellCenter', 'Vertex'}) : Expected target location of the fields.
    **options: Options related to interpolation strategy

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #interpolate_from_part_trees@start
        :end-before: #interpolate_from_part_trees@end
        :dedent: 2
  """
  src_parts_per_dom = list(get_parts_per_blocks(src_tree, comm).values())
  tgt_parts_per_dom = list(get_parts_per_blocks(tgt_tree, comm).values())

  interpolate_from_parts_per_dom(src_parts_per_dom, tgt_parts_per_dom, comm, containers_name, location, **options)

