from packaging.version import Version
from   mpi4py import MPI
import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

import maia
import maia.algo.part.point_cloud_utils as PCU
import maia.transfer.protocols          as EP
from   maia.factory.partitioning        import part_bound_orient as PBO
from   maia.utils                       import py_utils, par_utils, np_utils
from   maia.utils import logging as mlog
from   maia.utils import vstride as vs

from .cgns_to_pdm_pmesh import part_zones_to_pdm_pmesh_nodal
from .connectivity_utils import cell_vtx_connectivity_S
from .ngon_tools import pe_to_nface, edge_pe_to_ngon
from .geometry import _compute_elements_measure

from maia.algo.interpolation_impl import ConservativeInterpolator

import Pypdm.Pypdm as PDM

from maia.typing import *

PDM_VERSION = Version(PDM.__version__)

def _init_tetraisation_pt_type(pdm_intersection):
  """ A wrapper to set mi->tetraisation_pt_type, which is uninitialized if
  PDM < 2.8 (see paradigm!273) """
  if PDM_VERSION < Version('2.8'): # Do nothing if Version >= 2.8 (patch in PDM). 
    # Do dark magic to access directly C API
    import ctypes
    addr = id(pdm_intersection)

    # Offset for PyObject_HEAD
    offset = ctypes.sizeof(ctypes.c_ssize_t) + ctypes.sizeof(ctypes.c_void_p)
    mi_ptr = ctypes.cast(addr + offset, ctypes.POINTER(ctypes.c_void_p)).contents

    lib = ctypes.CDLL("libpdm.so")
    lib.PDM_mesh_intersection_tetraisation_pt_set(mi_ptr, ctypes.c_int(0), None)

def _get_native_measure(zone:CGNSTree) -> NDArray:
  path = f'Geometry_{PT.Zone.CellDimension(zone)}d/Measure'
  if (mes := PT.get_node_from_path(zone, path)) is not None:
    return PT.get_np_value(mes)
  else:
    return _compute_elements_measure(zone, 'CellCenter')

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
        pred = PT.pred.label_is('Elements_t') & PT.pred.NodePredicate(lambda n : PT.Element.Dimension(n) == celldim and PT.Element.Type(n) != 'NGON_n')
        for elt in PT.get_children_from_predicate(part, pred):
          elt_gnum = PT.get_np_value(MT.find_GlobalNumbering(elt, 'Sections'))
          elt_gnum += cell_offset

    sign = -1 if revert else 1
    vtx_offset  += sign*n_vtx_t
    face_offset += sign*n_face_t
    cell_offset += sign*n_cell_t

def compute_mesh_intersection(src_parts:List[CGNSPartTree],
                              tgt_parts:List[CGNSPartTree],
                              comm:MPIComm) -> Tuple[PDM.PartToPart, List[Dict[str, NDArray]]]:

  keep_alive = list()

  src_dim = tree_dim(src_parts, comm)
  tgt_dim = tree_dim(tgt_parts, comm)
  assert src_dim == tgt_dim
  dim = src_dim


  mi = PDM.MeshIntersection(comm,  PDM._PDM_MESH_INTERSECTION_KIND_WEIGHT, dim, dim, len(tgt_parts), len(src_parts))
  _init_tetraisation_pt_type(mi)

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
        assert (cx is not None) and (cy is not None) and (cz is not None)
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
          assert (cx is not None) and (cy is not None) and (cz is not None)
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
    self.volume_l = [_get_native_measure(zone) for zone in py_utils.to_flat_list(parts_per_dom)]
    self._create(parts_per_dom, comm)

  def _create(self, parts_per_dom:List[List[CGNSPartTree]], comm:MPIComm):

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
      vol = self.volume_l[i]
      vol_dispatch = np.repeat(vol / (dim+1), dim+1)
      dual_vol = dual_vol_l[i]
      self.inte_weight_l.append(vol_dispatch / dual_vol[self.cell_vtx_l[i].values-1])

  def _exchange_fields(self, vtx_fields:Dict[str, List[NDArray]], is_conservative:bool) -> Dict[str, List[NDArray]]:
    # NB : data must be already flattened in parts_per_dom order

    weight_l = self.cons_weight_l if is_conservative else self.inte_weight_l
    cell_fields:Dict[str, List[NDArray]] = {key: [] for key in vtx_fields}

    for i, cell_vtx in enumerate(self.cell_vtx_l):
      for fname, vtx_vals_l in vtx_fields.items():
        rep_field = vs.from_displs(cell_vtx.displs, vtx_vals_l[i][cell_vtx.values-1]*weight_l[i])
        cell_field = rep_field.reduce(vs.ReduceOp.SUM)
        cell_fields[fname].append(cell_field)

    return cell_fields

class CellToVertex:

  def __init__(self, parts_per_dom:List[List[CGNSPartTree]], comm:MPIComm):
    self.volume_l = [_get_native_measure(zone) for zone in py_utils.to_flat_list(parts_per_dom)]
    self._create(parts_per_dom, comm)
  def _create(self, parts_per_dom:List[List[CGNSPartTree]], comm:MPIComm):

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
    vol_dispatch_l = [np.repeat(vol / (self.dim+1), self.dim+1) for vol in self.volume_l]
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
  
    
class ConservativePartInterpolator(ConservativeInterpolator):

  """ Partitioned implementation of ConservativeInterpolator
  Multidomain is managed with global offset on gnum arrays """

  # Implementation of specific methods (see ConservativeInterpolator doc)
  @staticmethod
  def get_native_measure(zone, comm):
    return _get_native_measure(zone)
  @staticmethod
  def get_cell_clouds(zones, comm):
    return [PCU.get_point_cloud(zone, 'CellCenter') for zone in zones]
  @staticmethod
  def compute_mesh_intersection(src_parts, tgt_parts, comm):
    return compute_mesh_intersection(src_parts, tgt_parts, comm)

  def VertexToCell(self):
    obj = VertexToCell.__new__(VertexToCell)
    obj.volume_l = self.src_vol
    obj._create(self.src_parts_per_dom, self.comm)
    return obj
  def CellToVertex(self):
    obj = CellToVertex.__new__(CellToVertex)
    obj.volume_l = self.tgt_vol
    obj._create(self.tgt_parts_per_dom, self.comm)
    return obj

  # Specific __init__ to account for multidomain
  def __init__(self,
               src_parts_per_dom:List[List[CGNSPartTree]],
               tgt_parts_per_dom:List[List[CGNSPartTree]],
               comm:MPIComm,
               **kwargs):

    offset_mdom(src_parts_per_dom, comm)
    offset_mdom(tgt_parts_per_dom, comm)

    ConservativeInterpolator.__init__(self,
                                      py_utils.to_flat_list(src_parts_per_dom),
                                      py_utils.to_flat_list(tgt_parts_per_dom),
                                      comm,
                                      **kwargs)

    offset_mdom(src_parts_per_dom, comm, revert=True)
    offset_mdom(tgt_parts_per_dom, comm, revert=True)

    self.src_parts_per_dom = src_parts_per_dom
    self.tgt_parts_per_dom = tgt_parts_per_dom

