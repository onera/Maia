from mpi4py import MPI
import numpy as np

import Pypdm.Pypdm as PDM

import Converter.Internal as I

import maia.sids.Internal_ext   as IE

from maia                            import npy_pdm_gnum_dtype  as pdm_dtype
from maia.sids                       import sids as sids
from maia.sids                       import conventions as conv
from maia.utils                      import py_utils
from maia.tree_exchange.part_to_dist import discover    as disc
from maia                            import tree_exchange as TE

from maia.interpolation.interpolate  import get_point_cloud
from maia.geometry.extract_boundary2 import extract_surf_from_bc_new
from maia.geometry.geometry          import compute_cell_center

__doc__ = """
CGNS python module which interface the ParaDiGM library for // distance to wall computation .
"""

def get_entities_numbering(part_zone, as_pdm=True):
  """
  """
  vtx_ln_to_gn   = I.getVal(IE.getGlobalNumbering(part_zone, 'Vertex'))
  cell_ln_to_gn  = I.getVal(IE.getGlobalNumbering(part_zone, 'Cell'))
  if sids.Zone.Type(part_zone) == "Structured":
    face_ln_to_gn = I.getVal(IE.getGlobalNumbering(part_zone, 'Face'))
  else:
    ngons  = [e for e in I.getNodesFromType1(part_zone, 'Elements_t') if sids.ElementCGNSName(e) == 'NGON_n']
    assert len(ngons) == 1, "For unstructured zones, only NGon connectivity is supported"
    face_ln_to_gn = I.getVal(IE.getGlobalNumbering(ngons[0], 'Element'))
  if as_pdm:
    vtx_ln_to_gn  = vtx_ln_to_gn.astype(pdm_dtype)
    face_ln_to_gn = face_ln_to_gn.astype(pdm_dtype)
    cell_ln_to_gn = cell_ln_to_gn.astype(pdm_dtype)
  return vtx_ln_to_gn, face_ln_to_gn, cell_ln_to_gn

def detect_wall_families(tree, bcwalls=['BCWall', 'BCWallViscous', 'BCWallViscousHeatFlux', 'BCWallViscousIsothermal']):
  """
  """
  fam_query = lambda n : I.getType(n) == 'Family_t' and \
      I.getNodeFromType1(n, 'FamilyBC_t') is not None and I.getValue(I.getNodeFromType1(n, 'FamilyBC_t')) in bcwalls
  return [I.getName(family) for family in IE.iterNodesByMatching(tree, ['CGNSBase_t', fam_query])]


# ------------------------------------------------------------------------
class WallDistance:

  def __init__(self, part_tree, mpi_comm, method="cloud", families=[], out_fs_name='WallDistance'):
    self.part_tree = part_tree
    self.families  = families
    self.mpi_comm  = mpi_comm
    assert method in ["cloud", "propagation"]
    self.method    = method
    self.out_fs_n  = out_fs_name

    self._walldist = None
    self._keep_alive = []
    self._n_vtx_bnd_tot_idx  = [0]
    self._n_face_bnd_tot_idx = [0]


  def _setup_surf_mesh(self, parts_per_dom, families, comm):

    #This will concatenate part data of all initial domains
    face_vtx_bnd_l = []
    face_vtx_bnd_idx_l = []
    vtx_bnd_l = []
    face_ln_to_gn_l = []
    vtx_ln_to_gn_l = []

    for i_dom, part_zones in enumerate(parts_per_dom):

      face_vtx_bnd_z, face_vtx_bnd_idx_z, face_ln_to_gn_z, \
        vtx_bnd_z, vtx_ln_to_gn_z = extract_surf_from_bc_new(part_zones, families, comm)

      #Find the maximal vtx/face id for this initial domain
      n_face_bnd_t = 0
      for face_ln_to_gn in face_ln_to_gn_z:
        n_face_bnd_t = max(n_face_bnd_t, np.max(face_ln_to_gn, initial=0))
      n_face_bnd_t = comm.allreduce(n_face_bnd_t, op=MPI.MAX)
      self._n_face_bnd_tot_idx.append(self._n_face_bnd_tot_idx[-1] + n_face_bnd_t)

      n_vtx_bnd_t = 0
      for vtx_ln_to_gn in vtx_ln_to_gn_z:
        n_vtx_bnd_t = max(n_vtx_bnd_t, np.max(vtx_ln_to_gn, initial=0))
      n_vtx_bnd_t = comm.allreduce(n_vtx_bnd_t, op=MPI.MAX)
      self._n_vtx_bnd_tot_idx.append(self._n_vtx_bnd_tot_idx[-1] + n_vtx_bnd_t)

      #Shift the face and vertex lngn because PDM does not manage multiple domain. This will avoid
      # overlapping face / vtx coming from different domain but having same id
      for face_ln_to_gn in face_ln_to_gn_z:
        face_ln_to_gn += self._n_face_bnd_tot_idx[i_dom]
      for vtx_ln_to_gn in vtx_ln_to_gn_z:
        vtx_ln_to_gn += self._n_vtx_bnd_tot_idx[i_dom]

      #Extended global lists
      face_vtx_bnd_l.extend(face_vtx_bnd_z)
      face_vtx_bnd_idx_l.extend(face_vtx_bnd_idx_z)
      vtx_bnd_l.extend(vtx_bnd_z)
      face_ln_to_gn_l.extend(face_ln_to_gn_z)
      vtx_ln_to_gn_l.extend(vtx_ln_to_gn_z)


    n_part = len(vtx_bnd_l)
    for i in range(n_part):
    # Keep numpy alive
      for array in (face_vtx_bnd_l[i], face_vtx_bnd_idx_l[i], face_ln_to_gn_l[i], vtx_bnd_l[i], vtx_ln_to_gn_l[i],):
        self._keep_alive.append(array)

    #Get global data (total number of faces / vertices)
    #This create the surf_mesh objects in PDM, thus it must be done before surf_mesh_part_set
    self._walldist.surf_mesh_global_data_set(self._n_face_bnd_tot_idx[-1], self._n_vtx_bnd_tot_idx[-1])
    
    #Setup partitions
    for i_part in range(n_part):
      n_face_bnd = face_vtx_bnd_idx_l[i_part].shape[0]-1
      n_vtx_bnd  = vtx_ln_to_gn_l[i_part].shape[0]
      self._walldist.surf_mesh_part_set(i_part, n_face_bnd,
                                        face_vtx_bnd_idx_l[i_part],
                                        face_vtx_bnd_l[i_part],
                                        face_ln_to_gn_l[i_part],
                                        n_vtx_bnd,
                                        vtx_bnd_l[i_part],
                                        vtx_ln_to_gn_l[i_part])

  def _setup_vol_mesh(self, i_domain, part_zones, comm):
    """
    """
    #First pass to compute the total number of vtx, face, cell
    n_vtx_t, n_face_t, n_cell_t = 0, 0, 0
    for part_zone in part_zones:
      vtx_ln_to_gn, face_ln_to_gn, cell_ln_to_gn = get_entities_numbering(part_zone, as_pdm=False)
      n_vtx_t  = max(n_vtx_t,  np.max(vtx_ln_to_gn, initial=0))
      n_cell_t = max(n_face_t, np.max(cell_ln_to_gn, initial=0))
      n_face_t = max(n_cell_t, np.max(face_ln_to_gn, initial=0))

    #We could retrieve data from dist zone
    n_vtx_t  = comm.allreduce(n_vtx_t , op=MPI.MAX)
    n_cell_t = comm.allreduce(n_cell_t, op=MPI.MAX)
    n_face_t = comm.allreduce(n_face_t, op=MPI.MAX)

    #Setup global data
    self._walldist.vol_mesh_global_data_set(n_cell_t, n_face_t, n_vtx_t)


    for i_part, part_zone in enumerate(part_zones):

      vtx_coords = py_utils.interweave_arrays(sids.coordinates(part_zone))
      face_vtx, face_vtx_idx, _ = sids.ngon_connectivity(part_zone)

      nfaces  = [e for e in I.getNodesFromType1(part_zone, 'Elements_t') if sids.ElementCGNSName(e) == 'NFACE_n']
      assert len(nfaces) == 1, "NFace connectivity is needed for wall distance computing"
      cell_face_idx = I.getVal(I.getNodeFromName(nfaces[0], 'ElementStartOffset'))
      cell_face     = I.getVal(I.getNodeFromName(nfaces[0], 'ElementConnectivity'))

      vtx_ln_to_gn, face_ln_to_gn, cell_ln_to_gn = get_entities_numbering(part_zone)

      n_vtx  = vtx_ln_to_gn .shape[0]
      n_cell = cell_ln_to_gn.shape[0]
      n_face = face_ln_to_gn.shape[0]

      center_cell = compute_cell_center(part_zone)
      assert(center_cell.size == 3*n_cell)

      # Keep numpy alive
      for array in (cell_face_idx, cell_face, cell_ln_to_gn, face_vtx_idx, face_vtx, face_ln_to_gn, \
          vtx_coords, vtx_ln_to_gn, center_cell):
        self._keep_alive.append(array)

      self._walldist.vol_mesh_part_set(i_part,
                                       n_cell, cell_face_idx, cell_face, center_cell, cell_ln_to_gn,
                                       n_face, face_vtx_idx, face_vtx, face_ln_to_gn,
                                       n_vtx, vtx_coords, vtx_ln_to_gn)

  def _get(self, i_domain, part_zones):
    for i_part, part_zone in enumerate(part_zones):

      fields = self._walldist.get(i_domain, i_part) if self.method == "cloud" else self._walldist.get(i_part)

      # Test if FlowSolution already exists or create it
      fs_node = I.getNodeFromName1(part_zone, self.out_fs_n)
      if fs_node is None:
        fs_node = I.newFlowSolution(name=self.out_fs_n, gridLocation='CellCenter', parent=part_zone)
      assert sids.GridLocation(fs_node) == 'CellCenter'
      shape = sids.Zone.CellSize(part_zone)

      # Wall distance
      wall_dist = np.sqrt(fields['ClosestEltDistance'])
      I.newDataArray('TurbulentDistance', value=wall_dist.reshape(shape,order='F'), parent=fs_node)

      # Closest projected element
      closest_elt_proj = np.copy(fields['ClosestEltProjected'])
      I.newDataArray('ClosestEltProjectedX', closest_elt_proj[0::3].reshape(shape,order='F'), parent=fs_node)
      I.newDataArray('ClosestEltProjectedY', closest_elt_proj[1::3].reshape(shape,order='F'), parent=fs_node)
      I.newDataArray('ClosestEltProjectedZ', closest_elt_proj[2::3].reshape(shape,order='F'), parent=fs_node)

      # Closest gnum element (face)
      closest_elt_gnum = np.copy(fields['ClosestEltGnum'])
      I.newDataArray('ClosestEltGnum', closest_elt_gnum.reshape(shape,order='F'), parent=fs_node)

      # Find domain to which the face belongs (mainly for debug)
      n_face_bnd_tot_idx = np.array(self._n_face_bnd_tot_idx, dtype=closest_elt_gnum.dtype)
      closest_surf_domain = np.searchsorted(n_face_bnd_tot_idx, closest_elt_gnum-1, side='right') -1
      closest_surf_domain = closest_surf_domain.astype(closest_elt_gnum.dtype)
      closest_elt_gnuml = closest_elt_gnum - n_face_bnd_tot_idx[closest_surf_domain]
      I.newDataArray("ClosestEltDomId", value=closest_surf_domain.reshape(shape,order='F'), parent=fs_node)
      I.newDataArray("ClosestEltLocGnum", value=closest_elt_gnuml.reshape(shape,order='F'), parent=fs_node)



  def compute(self):
    """
    """

    #Get a skeleton tree including only Base, Zones and Families
    skeleton_tree = I.newCGNSTree()
    disc.discover_nodes_from_matching(skeleton_tree, [self.part_tree], 'CGNSBase_t', self.mpi_comm, child_list=['Family_t'])
    disc.discover_nodes_from_matching(skeleton_tree, [self.part_tree], 'CGNSBase_t/Zone_t', self.mpi_comm,
        merge_rule = lambda path: conv.get_part_prefix(path))

    # Search families if its are not given
    if not self.families:
      self.families = detect_wall_families(skeleton_tree)

    skeleton_families = [I.getName(f) for f in I.getNodesFromType(skeleton_tree, "Family_t")]
    found_families = any([fn in self.families for fn in skeleton_families])

    if found_families:

      # Group partitions by original dist domain
      parts_per_dom = list()
      for dbase, dzone in IE.iterNodesWithParentsByMatching(skeleton_tree, 'CGNSBase_t/Zone_t'):
        dzone_path = I.getName(dbase) + '/' + I.getName(dzone)
        parts_per_dom.append(TE.utils.get_partitioned_zones(self.part_tree, dzone_path))

      assert len(parts_per_dom) >= 1

      # Create walldist structure
      # Multidomain is not managed for n_part_surf, n_part_surf is the total of partitions
      n_part_surf = sum([len(part_zones) for part_zones in parts_per_dom])
      if self.method == "propagation":
        if len(parts_per_dom) > 1:
          raise NotImplementedError("Wall_distance computation with method 'propagation' does not support multiple domains")
        self._walldist = PDM.DistCellCenterSurf(self.mpi_comm, n_part_surf, n_part_vol=1)
      elif self.method == "cloud":
        n_part_per_cloud = [len(part_zones) for part_zones in parts_per_dom]
        self._walldist = PDM.DistCloudSurf(self.mpi_comm, 1, n_part_surf, point_clouds=n_part_per_cloud)


      self._setup_surf_mesh(parts_per_dom, self.families, self.mpi_comm)

      # Prepare mesh depending on method
      if self.method == "cloud":
        for i_domain, part_zones in enumerate(parts_per_dom):
          for i_part, part_zone in enumerate(part_zones):
            center_cell, cell_ln_to_gn = get_point_cloud(part_zone)
            self._keep_alive.extend([center_cell, cell_ln_to_gn])
            self._walldist.cloud_set(i_domain, i_part, cell_ln_to_gn.shape[0], center_cell, cell_ln_to_gn)

      elif self.method == "propagation":
        for i_domain, part_zones in enumerate(parts_per_dom):
          self._walldist.n_part_vol = len(part_zones)
          self._setup_vol_mesh(i_domain, part_zones, self.mpi_comm)

      #Compute
      args = ['rank1'] if self.method == 'propagation' else []
      self._walldist.compute(*args)

      # Get results -- OK because name of method is the same for 2 PDM objects
      for i_domain, part_zones in enumerate(parts_per_dom):
        self._get(i_domain, part_zones)

    else:
      raise ValueError(f"Unable to find BC family(ies) : {self.families} in {skeleton_families}.")

    # Free unnecessary numpy
    del self._keep_alive


  def dump_times(self):
    self._walldist.dump_times()


# ------------------------------------------------------------------------
def wall_distance(*args, **kwargs):
  walldist = WallDistance(*args, **kwargs)
  walldist.compute()
  walldist.dump_times()

