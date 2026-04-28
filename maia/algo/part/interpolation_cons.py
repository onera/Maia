from   mpi4py import MPI
import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

import maia
import maia.algo.part.point_cloud_utils as PCU
import maia.algo.part.closest_points    as CLO
import maia.transfer.protocols          as EP
from   maia.factory.partitioning        import part_bound_orient as PBO
from   maia.utils                       import py_utils, par_utils, np_utils
from   maia.utils import logging as mlog
from   maia.utils import vstride as vs

from .cgns_to_pdm_pmesh import part_zones_to_pdm_pmesh_nodal
from .connectivity_utils import cell_vtx_connectivity_S
from .ngon_tools import pe_to_nface, edge_pe_to_ngon
from .geometry import compute_elements_measure

from maia.algo import interpolation_utils as itp_utils

import Pypdm.Pypdm as PDM

from maia.typing import *

def _get_native_measure(zone:CGNSTree) -> NDArray:
  path = f'Geometry_{PT.Zone.CellDimension(zone)}d/Measure'
  if (mes := PT.get_node_from_path(zone, path)) is not None:
    return PT.get_np_value(mes)
  else:
    compute_elements_measure(zone, 'CellCenter')
    return PT.get_np_value(PT.find_node_from_path(zone, path))

def offset_mdom(parts_per_dom:List[List[CGNSPartTree]], comm:MPIComm, revert:bool=False):
  """ Offset gnum inplace to deal with multidomain cases
  If revert=True, do the opposite switch.
  
  For now, this is "poor multidomain" since shared vtx / faces between domaines does not have
  the same gnum (because get_mdom_gnum_vtx does not support S meshes)
  """

  if len(parts_per_dom) == 1:
    return

  cell_offset = face_offset = vtx_offset = 0

  # Precondition : mix of poly domains and std domains not allowed
  for parts in parts_per_dom:
    is_poly3d = comm.allreduce(all(PT.pred.is_zone_of_kind('Poly', 3)(z) for z in parts), MPI.LAND)
    n_cell_t = MT.Zone.n_cell(parts, comm)
    n_vtx_t  = MT.Zone.n_vtx(parts, comm)
    n_face_t = MT.Element.n_elt([PT.Zone.NGonNode(z) for z in parts], comm) if is_poly3d else 0

    for part in parts:

      vtx_gnum = MT.Zone.vtx_globalnumbering(part)
      cell_gnum = MT.Zone.cell_globalnumbering(part)
      vtx_gnum += vtx_offset
      cell_gnum += cell_offset

      # Elts
      if is_poly3d:
        face_gnum = MT.Element.globalnumbering(PT.Zone.NGonNode(part))
        face_gnum += face_offset
      else:
        celldim = PT.Zone.CellDimension(part)
        pred = PT.pred.label_is('Elements_t') & (lambda n : PT.Element.Dimension(n) == celldim and PT.Element.Type(n) != 'NGON_n')
        for elt in PT.get_children_from_predicate(part, pred):
          elt_gnum = PT.get_np_value(MT.find_GlobalNumbering(elt, 'Sections'))
          elt_gnum += cell_offset

    sign = -1 if revert else 1
    vtx_offset  += sign*n_vtx_t
    face_offset += sign*n_face_t
    cell_offset += sign*n_cell_t

def compute_mesh_intersection(src_parts:List[CGNSPartTree],
                              tgt_parts:List[CGNSPartTree],
                              comm:MPIComm,
                              dim:int) -> Tuple[PDM.PartToPart, List[Dict[str, NDArray]]]:

  keep_alive = list()

  mi = PDM.MeshIntersection(comm,  PDM._PDM_MESH_INTERSECTION_KIND_WEIGHT, dim, dim, len(tgt_parts), len(src_parts))

  # /!\ Register src as part_2 and tgt as part_1 /!\
  # For now we treat elt meshes / S meshes as part_mesh_nodal and Poly meshes with raw API
  # It may be better to use always partmeshnodal but this require some additions in part mesh nodal
  # cython class
  for i_mesh, parts in enumerate([tgt_parts, src_parts]):

    if comm.allreduce(all(PT.pred.is_zone_of_kind('Poly')(z) for z in parts), MPI.LAND): # Poly elements
      
      # NB : 2D poly meshes does not need to be reoriented
      if dim == 3:
        if not PBO.orientation_preserved(parts, comm):
          msg = "Poly3D meshes need to be partitioned with preserve_orientation=True to compute intersections." \
                " Orientations have been recomputed, but consider using this option for better performances."
          parts = PBO.shallow_preserve_orientation(parts, comm)
          mlog.warning(msg)

      mi.n_part_set(i_mesh, len(parts))
      for i_part,part in enumerate(parts):
        n_vtx = PT.Zone.n_vtx(part)
        cx, cy, cz = PT.Zone.coordinates(part)
        if cz is None:
          cz = np.zeros_like(cx)
        coords = np_utils.interweave_arrays([cx,cy,cz])
        vtx_lngn = MT.Zone.vtx_globalnumbering(part)
        cell_lngn = MT.Zone.cell_globalnumbering(part)

        if PT.Zone.CellDimension(part) == 2:
          if not PT.Zone.has_ngon_elements(part):
            edge_pe_to_ngon(part)

          ngon_n = PT.Zone.NGonNode(part)
          face_vtx = MT.Element.connectivity(ngon_n)
          n_face = cell_lngn.size

          mi.part_set(i_mesh, i_part, 0, n_face, 0, n_vtx, None, None, None, None, None,
                      face_vtx.displs, face_vtx.values, None, cell_lngn, None, vtx_lngn, coords)

        elif PT.Zone.CellDimension(part) == 3:
          if not PT.Zone.has_nface_elements(part):
            pe_to_nface(part)

          ngon_n = PT.Zone.NGonNode(part)
          nface_n = PT.Zone.NFaceNode(part)
          face_vtx = MT.Element.connectivity(ngon_n)
          face_lngn = MT.Element.globalnumbering(ngon_n)
          cell_face = MT.Element.connectivity(nface_n)
          n_cell = cell_lngn.size
          n_face = face_lngn.size

          mi.part_set(i_mesh, i_part, n_cell, n_face, 0, n_vtx, cell_face.displs, cell_face.values, None, None, None,
                      face_vtx.displs, face_vtx.values, cell_lngn, face_lngn, None, vtx_lngn, coords)

    else:

      if comm.allreduce(all(PT.pred.is_zone_of_kind('S')(z) for z in parts), MPI.LAND): # Struc
        _parts = []
        for i_part, part in enumerate(parts):
          _part = PT.new_Zone(PT.get_name(part), type='Unstructured', size=[[PT.Zone.n_vtx(part), PT.Zone.n_cell(part), 0]])
          cx, cy, cz = PT.Zone.coordinates(part)
          if cz is None:
            cz = np.zeros_like(cx)
          PT.new_GridCoordinates(fields={f'Coordinate{d}' : c.reshape(-1, order='F') for d,c in zip('XYZ', [cx,cy,cz])}, parent=_part)
          cell_vtx = cell_vtx_connectivity_S(part, PT.Zone.CellDimension(part))
          kind = 'QUAD_4' if PT.Zone.CellDimension(part) == 2 else 'HEXA_8'
          elt = PT.new_Elements('ELTS', kind, erange=[1, PT.Zone.n_cell(part)], econn=cell_vtx.values, parent=_part)
          MT.new_GlobalNumbering({'Element' : MT.Zone.cell_globalnumbering(part),
                                  'Sections': MT.Zone.cell_globalnumbering(part)}, elt)
          PT.add_child(_part, MT.find_GlobalNumbering(part))
          _parts.append(_part)

      else:
        _parts = parts

      pmn = part_zones_to_pdm_pmesh_nodal(_parts, comm)
      mi.part_nodal_set(i_mesh, pmn)
      keep_alive.append(pmn)


  mi.compute()
  ptp = mi.part_to_part_get()
  res = [mi.a_to_b_get(ipart) for ipart in range(len(tgt_parts))]

  return ptp, res

def tree_dim(parts:List[CGNSPartTree], comm:MPIComm) -> int:
  zone_dims = set(PT.Zone.CellDimension(z) for z in parts)
  dims = comm.allreduce(zone_dims, lambda s1,s2 : s1 | s2)
  if len(dims) != 1:
    raise RuntimeError("Inconsistent mesh dimension")
  return dims.pop()

class VertexToCell:

  def __init__(self, parts_per_dom:List[List[CGNSPartTree]], comm:MPIComm):

    zones = py_utils.to_flat_list(parts_per_dom)
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
    dual_vol_per_dom = [PDM.part_mesh_nodal_dual_volume(part_zones_to_pdm_pmesh_nodal(parts, comm)) for parts in parts_per_dom]
    dual_vol_l = py_utils.to_flat_list(dual_vol_per_dom)
    for i, zone in enumerate(zones):
      vol = _get_native_measure(zone)
      vol_dispatch = np.repeat(vol / (dim+1), dim+1)
      dual_vol = dual_vol_l[i]
      self.inte_weight_l.append(vol_dispatch / dual_vol[self.cell_vtx_l[i].values-1])

  def _exchange_fields(self, vtx_fields:Dict[str, List[NDArray]], is_conservative:bool) -> Dict[str, List[NDArray]]:
    # NB : data must be already flattened in parts_per_dom order

    weight_l = self.cons_weight_l if is_conservative else self.inte_weight_l
    cell_fields = {key: [] for key in vtx_fields}

    for i, cell_vtx in enumerate(self.cell_vtx_l):
      for fname, vtx_vals_l in vtx_fields.items():
        rep_field = vs.from_displs(cell_vtx.displs, vtx_vals_l[i][cell_vtx.values-1]*weight_l[i])
        cell_field = rep_field.reduce(vs.ReduceOp.SUM)
        cell_fields[fname].append(cell_field)

    return cell_fields

class CellToVertex:

  def __init__(self, parts_per_dom:List[List[CGNSPartTree]], comm:MPIComm):

    nb_tot_vertex = 0
    cell_vtx_gnum_l = list()
    vtx_gnum_l = list()
    for parts in parts_per_dom:
      for zone in parts:
        dim = PT.Zone.CellDimension(zone)
        elt = PT.find_child_from_predicate(zone, PT.pred.is_element_of_type('TRI_3' if dim == 2 else 'TETRA_4'))
        elt_vtx = PT.get_np_value(PT.find_child_from_name(elt, 'ElementConnectivity'))
        vtx_gnum = MT.Zone.vtx_globalnumbering(zone) - 1 + nb_tot_vertex

        cell_vtx_gnum_l.append(vtx_gnum[elt_vtx - 1])
        vtx_gnum_l.append(vtx_gnum)

      nb_tot_vertex += MT.Zone.n_vtx(parts, comm)
  
    zones = py_utils.to_flat_list(parts_per_dom)
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
    volume_l = [_get_native_measure(zone) for zone in zones]
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
  
    
class ConservativeInterpolator:

  def __init__(self,
               src_parts_per_dom:List[List[CGNSPartTree]],
               tgt_parts_per_dom:List[List[CGNSPartTree]],
               comm:MPIComm,
               **kwargs):
    

    # In the init part of the interpolator we build the part to part and weights
    # used to exchange data from source to target cells.
    # On each partition, weights are a strided array of size len(tgt_cells),
    # containing for each tgt cell a weight w_i for each of its related src cells
    #
    # Assuming that the exchanged fields will be in integrated from (eg mass),
    # the weigth from a source cell I to a tgt cell J is
    # 
    #   Volume_(I∩J) / Volume_I    for standard cells
    #   Volume_J / Volume_I        for tgt cells outside src mesh, where I is the closest cell
    #
    # Target cells that are partially outside src mesh are corrected with the
    # coefficient (1/r) where r = \sum_I Volume_(I∩J)  / Volume_J

    # Since the exchange fields will be usually in conservative form (eg density), we can
    # report the conservative <-> integrated factor (Volume_I / Volume_J) directly in weights
    # which become
    #   Volume_(I∩J) / Volume_J    for standard cells
    #   1                          for tgt cells outside src mesh, where I is the closest cell
    # and correction by (1/r) is unchanged

    src_parts = py_utils.to_flat_list(src_parts_per_dom)
    tgt_parts = py_utils.to_flat_list(tgt_parts_per_dom)
    src_dim = tree_dim(src_parts, comm)
    tgt_dim = tree_dim(tgt_parts, comm)
    assert src_dim == tgt_dim

    vol_src = [_get_native_measure(zone).reshape(-1, order='F') for zone in src_parts]
    vol_tgt = [_get_native_measure(zone).reshape(-1, order='F') for zone in tgt_parts]

    offset_mdom(src_parts_per_dom, comm)
    offset_mdom(tgt_parts_per_dom, comm)

    # Compute intersection between src (part 2) and tgt (part 1)
    ptp, tgt_to_src = compute_mesh_intersection(src_parts, tgt_parts, comm, src_dim)

    src_weights_l = [vs.from_displs(r['a_to_b_idx'], r['a_to_b_weight']) / vol \
                     for r,vol in zip(tgt_to_src, vol_tgt)]
    # Detect tgt cell not completly covered by src cells:
    #   - outside_mask is True if tgt cell is totally   outside src mesh
    #   - partial_mask is True       "        partially        "
    # vol_ratio is the fraction of tgt cell covered by src mesh, since a_to_b_weight already
    # include src vol and we divided by tgt_vol, we just have to sum
    tol = kwargs.get('measure_ratio_tol', 1E-12)
    vol_ratio = [src_weights.reduce(vs.ReduceOp.SUM) for src_weights in src_weights_l]
    partial_mask = [r*(1-r) > tol for r in vol_ratio]
    outside_mask = [r < tol       for r in vol_ratio]

    # Incorporate cut-cell correction for partial cells
    for i, weight in enumerate(src_weights_l):
      vol_ratio_rep    = np.repeat(vol_ratio[i], weight.counts)
      partial_mask_rep = np.repeat(partial_mask[i], weight.counts)
      np.divide(weight.values, vol_ratio_rep, where=partial_mask_rep, out=weight._values)

    # For outside cells, detect the closest cell in src mesh.
    # Incorporate it in PartToPart (update it) with a weight equal to tgt cell volume
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
        weight_mi = src_weights_l[i]

        if closest_out[i]['closest_src_gnum'].size > 0:
          a_to_b_clo = vs.from_counts(outside_mask[i].astype(np.int32), closest_out[i]['closest_src_gnum'])
          weight_clo = vs.from_counts(a_to_b_clo.counts, np.ones(a_to_b_clo.dsize))
          
          a_to_b_cat.append(vs.concatenate([a_to_b_mi, a_to_b_clo], vs.INNER_AXIS))
          weights_cat.append(vs.concatenate([weight_mi, weight_clo], vs.INNER_AXIS))
        else:
          a_to_b_cat.append(a_to_b_mi)
          weights_cat.append(weight_mi)

      # Override PartToPart and weights
      ptp = PDM.PartToPart(comm,
                           [MT.Zone.cell_globalnumbering(z) for z in tgt_parts], # Part 1 is tgt
                           [MT.Zone.cell_globalnumbering(z) for z in src_parts], # Part 2 is src
                           [a.displs for a in a_to_b_cat],
                           [a.values for a in a_to_b_cat])
      src_weights_l = weights_cat

    offset_mdom(src_parts_per_dom, comm, revert=True)
    offset_mdom(tgt_parts_per_dom, comm, revert=True)

    self.ptp = ptp
    self.src_weights_l = src_weights_l
    self.src_parts_per_dom = src_parts_per_dom
    self.tgt_parts_per_dom = tgt_parts_per_dom
    self.src_parts = src_parts
    self.tgt_parts = tgt_parts
    self.src_vol = vol_src
    self.tgt_vol = vol_tgt
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
      self._vtx_to_cell_src = VertexToCell(self.src_parts_per_dom, self.comm)
    return self._vtx_to_cell_src
  @property
  def cell_to_vtx_tgt(self):
    if self._cell_to_vtx_tgt is None:
      self._cell_to_vtx_tgt = CellToVertex(self.tgt_parts_per_dom, self.comm)
    return self._cell_to_vtx_tgt

  def cell_data_transfer(self, 
                         src_fields_l:Dict[str, List[NDArray]],
                         is_conservative:bool) -> Dict[str, List[NDArray]]:

    # This function is relevant for integrated fields (such as mass) :
    # if data is in conservative form, we must multiply it by Density
    # Reminder : in ptp, part1 is target mesh, part2 is src mesh
    rq_dict = dict()
    for name, src_fields in src_fields_l.items():
      
      # Integrated to conservative, if needed
      if not is_conservative:
        src_fields = [f / vol for f,vol in zip(src_fields, self.src_vol)]

      rq_dict[name] = self.ptp.reverse_iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                                             PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART2,
                                             src_fields)
    tgt_fields_l = {}
    for name, rq in rq_dict.items():
      _, recv_datas = self.ptp.reverse_wait(rq)
  
      # Ponderate by weights
      tgt_fields = [
        vs.from_displs(src_weights.displs, data*src_weights.values).reduce(vs.ReduceOp.SUM)
        for data, src_weights in zip(recv_datas, self.src_weights_l)]

      # Conservative to integrated, if needed
      if not is_conservative:
        for f, vol in zip(tgt_fields, self.tgt_vol):
          f *= vol
      
      tgt_fields_l[name] = tgt_fields

    return tgt_fields_l


  def exchange_fields(self, container_name:str, tgt_loc:str, is_conservative=True):

    field_names, cnt_label, src_loc = itp_utils.discover_fields_name(self.src_parts, container_name, self.root, self.comm)

    src_fields_l = {key: [] for key in field_names}
    for src_zone in self.src_parts:
      container = PT.find_node_from_path(src_zone, container_name)
      for key, val in PT.Container.fields(container).items():
        src_fields_l[key].append(val.reshape(-1, order='F')) # Flatten if src zone is S

    if src_loc == 'Vertex':
      src_fields_l = self.vtx_to_cell_src._exchange_fields(src_fields_l, is_conservative)
    elif src_loc != 'CellCenter':
      raise ValueError(f"Unsupported location for input container: {src_loc}")
      
    tgt_fields_l = self.cell_data_transfer(src_fields_l, is_conservative)

    # Back to vertex
    if tgt_loc == 'Vertex':
      tgt_fields_l = self.cell_to_vtx_tgt._exchange_fields(tgt_fields_l, is_conservative)
    elif tgt_loc != 'CellCenter':
      raise ValueError(f"Unsupported location for output container: {tgt_loc}")

    # Update target partitions
    for i,tgt_part in enumerate(self.tgt_parts):
      shape = PT.Zone.CellSize(tgt_part) if tgt_loc == 'CellCenter' else PT.Zone.VertexSize(tgt_part)
      PT.rm_children_from_name(tgt_part, container_name)
      fields = {key: vals[i].reshape(shape, order='F') for key,vals in tgt_fields_l.items()}
      fs = PT.new_FlowSolution(container_name, loc=tgt_loc, fields=fields, parent=tgt_part)
      PT.set_label(fs, cnt_label)

