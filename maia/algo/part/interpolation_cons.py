from   mpi4py import MPI
import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

import maia
import maia.algo.part.point_cloud_utils as PCU
import maia.algo.part.closest_points    as CLO
import maia.transfer.protocols          as EP
from   maia.utils                       import par_utils
from   maia.utils import vstride as vs

from .cgns_to_pdm_pmesh import part_zones_to_pdm_pmesh_nodal

from maia.algo import interpolation_utils as itp_utils


import Pypdm.Pypdm as PDM

from maia.typing import *

def _get_native_measure(zone):
  dim = PT.Zone.CellDimension(zone)
  return PT.get_np_value(PT.find_node_from_path(zone, f'Geometry_{dim}d/Measure'))


def compute_mesh_intersection(src_parts:List[CGNSPartTree],
                              tgt_parts:List[CGNSPartTree],
                              comm:MPIComm,
                              dim:int) -> Tuple[PDM.PartToPart, List[Dict[str, NDArray]]]:

  pmn1 = part_zones_to_pdm_pmesh_nodal(src_parts, comm)
  pmn2 = part_zones_to_pdm_pmesh_nodal(tgt_parts, comm)

  # NB : the two last args are unused by class
  mi = PDM.MeshIntersection(comm,  PDM._PDM_MESH_INTERSECTION_KIND_WEIGHT, dim, dim, 1, 1)

  # /!\ Register src as part_2 and tgt as part_1 /!\
  mi.part_nodal_set(0, pmn2)
  mi.part_nodal_set(1, pmn1)

  mi.compute()
  ptp = mi.part_to_part_get()
  res = [mi.a_to_b_get(ipart) for ipart in range(len(tgt_parts))]

  return ptp, res


def cell_data_transfer(ptp:PDM.PartToPart,
                       src_weights_l:List[vs.VStrideArray],
                       src_fields_l:Dict[str, List[NDArray]]) -> Dict[str, List[NDArray]]:

  # Remainder : in ptp, part1 is target mesh, part2 is src mesh
  rq_dict = dict()
  for name, src_fields in src_fields_l.items():
    
    rq_dict[name] = ptp.reverse_iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                                      PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART2,
                                      src_fields)
  tgt_fields_l = {}
  for name, rq in rq_dict.items():
    _, recv_datas = ptp.reverse_wait(rq)
 
    # Ponderate by weights
    tgt_fields_l[name] = list()
    for data, src_weights in zip(recv_datas, src_weights_l):
      extended_data = vs.from_displs(src_weights.displs, data*src_weights.values)
      reduced_data = extended_data.reduce(vs.ReduceOp.SUM)
      
      tgt_fields_l[name].append(reduced_data)


  return tgt_fields_l



def tree_dim(tree:CGNSPartTree, comm:MPIComm) -> int:
  base_dims = set(PT.Base.CellDimension(b) for b in PT.iter_all_CGNSBase_t(tree))
  dims = comm.allreduce(base_dims, lambda s1,s2 : s1 & s2)
  if len(dims) != 1:
    raise RuntimeError("Inconsistent mesh dimension")
  return dims.pop()

class VertexToCell:

  def __init__(self, tree:CGNSPartTree, comm:MPIComm):
    zones = PT.get_all_Zone_t(tree)

    self.cell_vtx_l = []
    for zone in zones:
      dim = PT.Zone.CellDimension(zone)
      elt = PT.find_child_from_predicate(zone, PT.pred.is_element_of_type('TRI_3' if dim == 2 else 'TETRA_4'))
      self.cell_vtx_l.append(MT.Element.connectivity(elt))

    dim = dim if len(zones) > 0 else 0

    # Weight must be already extended (size = cell_vtx.dsize)

    # Method 1: Basic arithmetic mean
    #   Within a cell, each vertex contributes with a constant weight
    #   w = 1 / n_vtx_of_cell  
    #   ---> Suitable for conservative fields (eg density)
    self.cons_weight_l = [1./(dim+1)] * len(zones) # Trick : use cste, numpy will broadcast

    # Method 2: Dual volumes
    #  Within a cell, each vertex contributes up to the fraction of dual volume provided by this cell:
    #  w = dual_volume_contribution_from_cell / dual_volume_of_vertex
    #  ---> Suitable for integrated fields (eg mass)
    self.inte_weight_l = []
    dual_vol_l = PDM.part_mesh_nodal_dual_volume(part_zones_to_pdm_pmesh_nodal(zones, comm))
    for i, zone in enumerate(zones):
      vol = _get_native_measure(zone)
      vol_dispatch = np.repeat(vol / (dim+1), dim+1)
      dual_vol = dual_vol_l[i]
      self.inte_weight_l.append(vol_dispatch / dual_vol[self.cell_vtx_l[i].values-1])

  def _exchange_fields(self, vtx_fields:Dict[str, List[NDArray]], is_conservative:bool) -> Dict[str, List[NDArray]]:

    weight_l = self.cons_weight_l if is_conservative else self.inte_weight_l
    cell_fields = {key: [] for key in vtx_fields}

    for i, cell_vtx in enumerate(self.cell_vtx_l):
      for fname, vtx_vals_l in vtx_fields.items():
        rep_field = vs.from_displs(cell_vtx.displs, vtx_vals_l[i][cell_vtx.values-1]*weight_l[i])
        cell_field = rep_field.reduce(vs.ReduceOp.SUM)
        cell_fields[fname].append(cell_field)

    return cell_fields

class CellToVertex:

  def __init__(self, tree:CGNSPartTree, comm:MPIComm):
    zones = PT.get_all_Zone_t(tree)

    nb_tot_vertex = MT.Zone.n_vtx(zones, comm)
    vtx_distri  = par_utils.uniform_distribution(nb_tot_vertex,  comm)

    cell_vtx_gnum_l = list()
    vtx_gnum_l = list()
    for zone in zones:
      dim = PT.Zone.CellDimension(zone)
      elt = PT.find_child_from_predicate(zone, PT.pred.is_element_of_type('TRI_3' if dim == 2 else 'TETRA_4'))
      elt_vtx = PT.get_np_value(PT.find_child_from_name(elt, 'ElementConnectivity'))
      vtx_gnum = MT.Zone.vtx_globalnumbering(zone)

      cell_vtx_gnum_l.append(vtx_gnum[elt_vtx - 1])
      vtx_gnum_l.append(vtx_gnum)
  
    self.dim = dim if len(zones) > 0 else 0

    # First indexer for cell_vtx connectivity, second one only for vertices
    self.cnt_gi = EP.GlobalIndexer(vtx_distri, cell_vtx_gnum_l, comm, gnum_offset=1)
    self.vtx_gi = EP.GlobalIndexer(vtx_distri, vtx_gnum_l,      comm, gnum_offset=1)

    # Weight must be already extended (size = cell_vtx.dsize)

    # Method 1: Basic arithmetic mean
    #   From each cell, use constant weight for every vertices
    #   w = 1 / n_vtx_of_cell  
    #  ---> Suitable for integrated fields (eg mass)
    self.inte_weight_l = [1./(self.dim+1)] * len(zones) # Trick : use cste, numpy will broadcast

    # Method 2: Dual volumes
    #  From each cell, contribute to each vtx using to the fraction of vtx dual volume provided by the cell:
    #  w = dual_volume_contribution_from_cell / dual_volume_of_vertex
    #   ---> Suitable for conservative fields (eg density)
    volume_l = [_get_native_measure(zone) for zone in zones]
    vol_dispatch_l = [np.repeat(vol / (self.dim+1), self.dim+1) for vol in volume_l]
    # We already have indexer so compute dual volume manually
    dual_vol_rep_l = self.cnt_gi.Take(self.cnt_gi.Put(vol_dispatch_l, reduce=EP.ReduceOp.SUM))
    self.cons_weight_l = [vol_dispatch / dual_vol_rep for vol_dispatch, dual_vol_rep in zip(vol_dispatch_l, dual_vol_rep_l)]

  def _exchange_fields(self, cell_fields:Dict[str, List[NDArray]], is_conservative:bool) -> Dict[str, List[NDArray]]:

    weight_l = self.cons_weight_l if is_conservative else self.inte_weight_l
    vtx_fields = dict()

    for fname, vals in cell_fields.items():
      data = [np.repeat(val, self.dim+1)*weight for val,weight in zip(vals, weight_l)]
      vtx_fields[fname] = self.vtx_gi.Take(self.cnt_gi.Put(data, reduce=EP.ReduceOp.SUM))

    return vtx_fields
  
    
class ConservativeInterpolator:

  def __init__(self, src_tree, tgt_tree, src_loc, tgt_loc, comm):
    
    #  Restrictions
    # monodomain ? 
    # elements simpliciaux ?
    # src_loc == tgt_loc
    src_dim = tree_dim(src_tree, comm)
    tgt_dim = tree_dim(tgt_tree, comm)
    assert src_dim == tgt_dim

    maia.algo.compute_elements_measure(src_tree, 'CellCenter', comm)
    maia.algo.compute_elements_measure(tgt_tree, 'CellCenter', comm)

    src_parts = PT.get_all_Zone_t(src_tree)
    tgt_parts = PT.get_all_Zone_t(tgt_tree)
    vol_src = [_get_native_measure(zone) for zone in src_parts]
    vol_tgt = [_get_native_measure(zone) for zone in tgt_parts]


    # Compute intersection between src (part 2) and tgt (part 1)
    ptp, tgt_to_src = compute_mesh_intersection(src_parts, tgt_parts, comm, src_dim)

    # Convert PDM weight (which are actually measures) to ratio,
    # dividing by src volumes (exchange needed to bring this to tgt)
    rq = ptp.reverse_iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                              PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART2,
                              vol_src)
    _, recv_vols = ptp.reverse_wait(rq)

    src_weights = [vs.from_displs(tgt_to_src['a_to_b_idx'], tgt_to_src['a_to_b_weight'] / vol)
                    for tgt_to_src, vol in zip(tgt_to_src, recv_vols)]

    # Detect tgt cell not completly covered by src cells:
    #   - outside_mask is True if tgt cell is totally   outside src mesh
    #   - partial_mask is True       "        partially        "
    # vol_ratio is the fraction of tgt cell covered by src mesh
    vol_from_src = [vs.from_displs(tgt_to_src['a_to_b_idx'], tgt_to_src['a_to_b_weight'])
                   .reduce(vs.ReduceOp.SUM) for tgt_to_src in tgt_to_src]
    vol_ratio = [from_src / tgt for from_src, tgt in zip(vol_from_src, vol_tgt)]
    partial_mask = [r*(1-r) > 1E-15 for r in vol_ratio]
    outside_mask = [r < 1E-15       for r in vol_ratio]

    # Incorporate cut-cell correction direcly in src_weights
    for i, weight in enumerate(src_weights):
      vol_ratio_rep    = np.repeat(vol_ratio[i], weight.counts)
      partial_mask_rep = np.repeat(partial_mask[i], weight.counts)
      np.divide(weight.values, vol_ratio_rep, where=partial_mask_rep, out=weight._values)

    # For outside cells, detect the closest cell in src mesh.
    # This time we have to incorporate it in PartToPart (update it) and in weight
    # as well (using a weight of 1.)
    if comm.allreduce(any([outside.any() for outside in outside_mask]), MPI.LOR):
      # Perform closest point on outside cells only
      src_clouds = [PCU.get_point_cloud(part, 'CellCenter') for part in src_parts]
      tgt_clouds = [PCU.get_point_cloud(part, 'CellCenter') for part in tgt_parts]
      tgt_clouds = [PCU.extract_sub_cloud_from_flag(cloud, flag) for cloud,flag in zip(tgt_clouds, outside_mask)]

      closest_out = CLO._mdom_closest_points([src_clouds], [tgt_clouds], comm, False, n_pts=1, need_shift=True)[0]

      a_to_b_cat = []
      weights_cat = []
      for i in range(len(tgt_clouds)):
        a_to_b_mi = vs.from_displs(tgt_to_src[i]['a_to_b_idx'], tgt_to_src[i]['a_to_b'])
        weight_mi = src_weights[i]

        if closest_out[i]['closest_src_gnum'].size > 0:
          a_to_b_clo = vs.from_counts(outside_mask[i].astype(np.int32), closest_out[i]['closest_src_gnum'])
          weight_clo = vs.from_counts(a_to_b_clo.counts, np.ones(a_to_b_clo.dsize))
          
          a_to_b_cat.append(vs.concatenate([a_to_b_mi, a_to_b_clo], vs.INNER_AXIS))
          weights_cat.append(vs.concatenate([weight_mi, weight_clo], vs.INNER_AXIS))
        else:
          a_to_b_cat.append(a_to_b_mi)
          weights_cat.append(weight_mi)

      # Override PartToPart & src_weights
      ptp = PDM.PartToPart(comm,
                           [MT.Zone.cell_globalnumbering(z) for z in tgt_parts], # Part 1 is tgt
                           [MT.Zone.cell_globalnumbering(z) for z in src_parts], # Part 2 is src
                           [a.displs for a in a_to_b_cat],
                           [a.values for a in a_to_b_cat])
      src_weights = weights_cat

    self.ptp = ptp
    self.src_weights = src_weights
    self.src_tree = src_tree
    self.tgt_tree = tgt_tree
    self.comm = comm

    # Caching
    self._vtx_to_cell_src = None
    self._cell_to_vtx_tgt = None

    # If some rank have no partitions, store a rank used as root to share FS names
    self.root = None
    if comm.allreduce(len(src_parts) == 0, MPI.LOR):
      self.root = self.comm.allreduce(-1 if len(src_parts) == 0 else comm.rank, MPI.MAX)

  @property
  def vtx_to_cell_src(self):
    if self._vtx_to_cell_src is None:
      self._vtx_to_cell_src = VertexToCell(self.src_tree, self.comm)
    return self._vtx_to_cell_src
  @property
  def cell_to_vtx_tgt(self):
    if self._cell_to_vtx_tgt is None:
      self._cell_to_vtx_tgt = CellToVertex(self.tgt_tree, self.comm)
    return self._cell_to_vtx_tgt

  def exchange_fields(self, container_name:str, tgt_loc:str):

    src_parts = PT.get_all_Zone_t(self.src_tree)
    tgt_parts = PT.get_all_Zone_t(self.tgt_tree)
    field_names, cnt_label = itp_utils.discover_fields_name(src_parts, container_name, self.root, self.comm)

    src_fields_l = {key: [] for key in field_names}
    for src_zone in src_parts:
      container = PT.find_node_from_path(src_zone, container_name)
      loc = PT.Container.GridLocation(container)
      for key, val in PT.Container.fields(container).items():
        src_fields_l[key].append(val)

    if loc == 'Vertex':
      src_fields_l = self.vtx_to_cell_src._exchange_fields(src_fields_l)
    elif loc != 'CellCenter':
      raise ValueError(f"Unsupported location for input container: {loc}")
      
    tgt_fields_l = cell_data_transfer(self.ptp, self.src_weights, src_fields_l)

    # Back to vertex
    if tgt_loc == 'Vertex':
      tgt_fields_l = self.cell_to_vtx_tgt._exchange_fields(tgt_fields_l)
    elif tgt_loc != 'CellCenter':
      raise ValueError(f"Unsupported location for output container: {tgt_loc}")


    # Update target partitions
    for i,tgt_part in enumerate(tgt_parts):
      PT.rm_children_from_name(tgt_part, container_name)
      fields = {key: vals[i] for key,vals in tgt_fields_l.items()}
      fs = PT.new_FlowSolution(container_name, loc=tgt_loc, fields=fields, parent=tgt_part)
      PT.set_label(fs, cnt_label)

