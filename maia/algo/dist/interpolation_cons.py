from mpi4py import MPI
import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.pytree.maia import pdm_elts

import maia.transfer.protocols          as EP
from   maia.utils                       import par_utils
from   maia.utils import vstride as vs

from .         import point_cloud_utils as PCU
from .geometry import compute_elements_measure
from .localize    import minimal_partitioning
from .closest_elt import minimal_partitioning_poly2D

from maia.algo.interpolation_utils import ConservativeInterpolator

import Pypdm.Pypdm as PDM

from maia.typing import *

def tree_dim(parts:List[CGNSDistTree]) -> int:
  dims = set(PT.Zone.CellDimension(z) for z in parts)
  if len(dims) != 1:
    raise RuntimeError("Inconsistent mesh dimension")
  return dims.pop()

def _get_native_measure(zone:CGNSTree, comm:MPIComm) -> NDArray:
  path = f'Geometry_{PT.Zone.CellDimension(zone)}d/Measure'
  if (mes := PT.get_node_from_path(zone, path)) is not None:
    return PT.get_np_value(mes)
  else:
    compute_elements_measure(zone, 'CellCenter', comm)
    return PT.get_np_value(PT.find_node_from_path(zone, path))

def compute_mesh_intersection(src_doms:List[CGNSDistTree],
                              tgt_doms:List[CGNSDistTree],
                              comm:MPIComm) -> Tuple[PDM.PartToPart, List[Dict[str, NDArray]]]:

  from maia.algo.part.interpolation_cons import _init_tetraisation_pt_type

  keep_alive = list()

  src_dim = tree_dim(src_doms)
  tgt_dim = tree_dim(tgt_doms)
  assert src_dim == tgt_dim
  dim = src_dim

  mi = PDM.MeshIntersection(comm,  PDM._PDM_MESH_INTERSECTION_KIND_WEIGHT, dim, dim, len(tgt_doms), len(src_doms))
  _init_tetraisation_pt_type(mi)

  for i_mesh, zones in enumerate([tgt_doms, src_doms]):

    vtx_mdom = face_mdom = cell_mdom = 0 # Multidomain management
    if all(PT.pred.is_zone_of_kind('Poly')(z) for z in zones): # Poly elements

      mi.n_part_set(i_mesh, len(zones))
      for i_zone,zone in enumerate(zones):
        if PT.Zone.CellDimension(zone) == 2:
          face_vtx_idx, face_vtx, coords, face_gnum, vtx_gnum = minimal_partitioning_poly2D(zone, comm)
          mi.part_set(i_mesh, i_zone, 0, face_gnum.size, 0, vtx_gnum.size, None, None, None, None, None,
                      face_vtx_idx, face_vtx, None, face_gnum+cell_mdom, None, vtx_gnum+vtx_mdom, coords)
        else:
          cell_face_idx, cell_face, face_vtx_idx, face_vtx, coords, \
          cell_gnum, face_gnum, vtx_gnum = minimal_partitioning(zone, comm)

          mi.part_set(i_mesh, i_zone, cell_gnum.size, face_gnum.size, 0, vtx_gnum.size, cell_face_idx, cell_face, None, None, None,
                      face_vtx_idx, face_vtx, cell_gnum+cell_mdom, face_gnum+face_mdom, None, vtx_gnum+vtx_mdom, coords)

          face_mdom += PT.Zone.n_face(zone)
        vtx_mdom  += MT.Zone.n_vtx(zone)
        cell_mdom += MT.Zone.n_cell(zone)

    else:

      pmesh_nodal = PDM.PartMeshNodal(comm, len(zones), dim)
      if all(PT.pred.is_zone_of_kind('S')(z) for z in zones): # Struct
        # S meshes : use a single section for all domains
        pdm_elt_type = pdm_elts.cgns_elt_name_to_pdm_element_type('QUAD_4' if dim == 2 else 'HEXA_8')
        id_section   = pmesh_nodal.add_section(pdm_elt_type)

        for i_zone, zone in enumerate(zones):
          cell_vtx_idx, cell_vtx, coords, cell_gnum, vtx_gnum = minimal_partitioning(zone, comm)

          pmesh_nodal.set_coordinates(i_zone, coords, vtx_gnum+vtx_mdom)
          pmesh_nodal.set_section(id_section, i_zone, cell_vtx, cell_gnum+cell_mdom, None, None, cell_gnum.size)

          vtx_mdom  += PT.Zone.n_vtx(zone)
          cell_mdom += PT.Zone.n_cell(zone)

      else:
        id_sections = list()
        # Elt meshes : each domain store distinct sections # TODO : part offset on cell gnum
        for zone in zones:
          for elt in PT.Zone.get_ordered_elements_per_dim(zone)[dim]:
            pdm_elt_type = pdm_elts.cgns_elt_name_to_pdm_element_type(PT.Element.Type(elt))
            id_sections.append(pmesh_nodal.add_section(pdm_elt_type))
        id_sections_it = iter(id_sections)
        
        for i_zone, zone in enumerate(zones):
          cell_distri = MT.Zone.cell_distribution(zone)
          cell_vtx_idx, cell_vtx, coords, cell_gnum, vtx_gnum = minimal_partitioning(zone, comm)
          
          pmesh_nodal.set_coordinates(i_zone, coords, vtx_gnum+vtx_mdom)
          elt_offset = 0 # Global start of current Element section, in dim-elements numbering
          loc_offset = 0 # Local start of generated cells
          for elt in PT.Zone.get_ordered_elements_per_dim(zone)[dim]:
            # Compute number of elements intersecting global distri
            n_elt = MT.Element.n_elt(elt)
            n_inter = max(0, min(cell_distri[1], elt_offset+n_elt) - max(cell_distri[0], elt_offset))
            
            start, stop = cell_vtx_idx[loc_offset], cell_vtx_idx[loc_offset + n_inter]
            pmesh_nodal.set_section(next(id_sections_it), i_zone, cell_vtx[start:stop], cell_gnum[loc_offset:loc_offset+n_inter]+cell_mdom, None, None, n_inter)
            loc_offset += n_inter
            elt_offset += n_elt

          cell_mdom += PT.Zone.n_cell(zone)
          vtx_mdom  += PT.Zone.n_vtx(zone)

      mi.part_nodal_set(i_mesh, pmesh_nodal)
      keep_alive.append(pmesh_nodal)

  mi.compute()
  ptp = mi.part_to_part_get()
  res = [mi.a_to_b_get(idom) for idom in range(len(tgt_doms))]

  return ptp, res

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
  
    
class ConservativeDistInterpolator(ConservativeInterpolator):

  """ Distributed implementation of ConservativeInterpolator
  Multidomain is managed with local offset in get_cell_clouds and mesh_intersection """

  @staticmethod
  def get_native_measure(zone, comm):
    return _get_native_measure(zone, comm)
  @staticmethod
  def get_cell_clouds(zones, comm):
    clouds = list()
    offset = 0
    for zone in zones:
      cloud = PCU.get_point_cloud(zone, comm, 'CellCenter')
      cloud[1].__iadd__(offset)
      clouds.append(cloud)
      offset += PT.Zone.n_cell(zone)
    return clouds
  @staticmethod
  def compute_mesh_intersection(src_parts, tgt_parts, comm):
    return compute_mesh_intersection(src_parts, tgt_parts, comm)

  def VertexToCell(self):
    return VertexToCell(self.src_parts, self.comm)
  def CellToVertex(self):
    return CellToVertex(self.tgt_parts, self.comm)

