import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

import maia.transfer.protocols          as EP
from   maia.utils                       import par_utils
from   maia.utils import vstride as vs

from .geometry import compute_elements_measure

from maia.typing import *


def _get_native_measure(zone:CGNSTree, comm:MPIComm) -> NDArray:
  path = f'Geometry_{PT.Zone.CellDimension(zone)}d/Measure'
  if (mes := PT.get_node_from_path(zone, path)) is not None:
    return PT.get_np_value(mes)
  else:
    compute_elements_measure(zone, 'CellCenter', comm)
    return PT.get_np_value(PT.find_node_from_path(zone, path))

class VertexToCell:

  def __init__(self, zones:List[CGNSDistTree], comm:MPIComm):

    self.cell_vtx_l = []
    self.indexer_l = []
    for zone in zones:
      dim = PT.Zone.CellDimension(zone)
      elt = PT.find_child_from_predicate(zone, PT.pred.is_element_of_type('TRI_3' if dim == 2 else 'TETRA_4'))
      cell_vtx = MT.Element.connectivity(elt)
      indexer = EP.GlobalIndexer(MT.Zone.vtx_distribution(zone), cell_vtx.values-1, comm)
      self.cell_vtx_l.append(cell_vtx)
      self.indexer_l.append(indexer)

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
    for i, zone in enumerate(zones):
      vol = _get_native_measure(zone, comm)
      vol_dispatch = np.repeat(vol / (dim+1), dim+1)
      dual_vol = self.indexer_l[i].Put(vol_dispatch, reduce=EP.ReduceOp.SUM)
      self.inte_weight_l.append(vol_dispatch / self.indexer_l[i].Take(dual_vol))

  def _exchange_fields(self, vtx_fields:Dict[str, List[NDArray]], is_conservative:bool) -> Dict[str, List[NDArray]]:

    weight_l = self.cons_weight_l if is_conservative else self.inte_weight_l
    cell_fields:Dict[str, List[NDArray]] = {key: [] for key in vtx_fields}

    for i, cell_vtx in enumerate(self.cell_vtx_l):
      for fname, vtx_vals_l in vtx_fields.items():
        rep_field = vs.from_displs(cell_vtx.displs, self.indexer_l[i].Take(vtx_vals_l[i])*weight_l[i])
        cell_field = rep_field.reduce(vs.ReduceOp.SUM)
        cell_fields[fname].append(cell_field)

    return cell_fields

class CellToVertex:

  def __init__(self, zones:List[CGNSDistTree], comm:MPIComm):

    nb_tot_vertex = 0
    cell_vtx_gnum_l = list()
    vtx_gnum_l = list()
    for zone in zones:
      dim = PT.Zone.CellDimension(zone)
      elt = PT.find_child_from_predicate(zone, PT.pred.is_element_of_type('TRI_3' if dim == 2 else 'TETRA_4'))
      elt_vtx = PT.get_np_value(PT.find_child_from_name(elt, 'ElementConnectivity'))
      vtx_distri = MT.Zone.vtx_distribution(zone)

      cell_vtx_gnum_l.append(elt_vtx - 1 + nb_tot_vertex)
      vtx_gnum_l.append(np.arange(vtx_distri[0]+nb_tot_vertex, vtx_distri[1]+nb_tot_vertex))

      nb_tot_vertex += MT.Zone.n_vtx(zone)
  
    self.dim = dim if len(zones) > 0 else 0

    vtx_distri  = par_utils.uniform_distribution(nb_tot_vertex,  comm)
    # First indexer for cell_vtx connectivity, second one only for vertices
    self.cnt_gi = EP.GlobalIndexer(vtx_distri, cell_vtx_gnum_l, comm)
    self.vtx_gi = EP.GlobalIndexer(vtx_distri, vtx_gnum_l,      comm)

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
    volume_l = [_get_native_measure(zone, comm) for zone in zones]
    vol_dispatch_l = [np.repeat(vol / (self.dim+1), self.dim+1) for vol in volume_l]
    # We already have indexer so compute dual volume manually
    dual_vol_rep_l = self.cnt_gi.Take(self.cnt_gi.Put(vol_dispatch_l, reduce=EP.ReduceOp.SUM))
    self.cons_weight_l = [vol_dispatch / dual_vol_rep for vol_dispatch, dual_vol_rep in zip(vol_dispatch_l, dual_vol_rep_l)]

  def _exchange_fields(self, cell_fields:Dict[str, List[NDArray]], is_conservative:bool) -> Dict[str, List[NDArray]]:
    # NB : data must be already flattened in parts_per_dom order

    weight_l = self.cons_weight_l if is_conservative else self.inte_weight_l
    vtx_fields = dict()

    for fname, vals in cell_fields.items():
      data = [np.repeat(val, self.dim+1)*weight for val,weight in zip(vals, weight_l)]
      vtx_fields[fname] = self.vtx_gi.Take(self.cnt_gi.Put(data, reduce=EP.ReduceOp.SUM))

    return vtx_fields
  
    