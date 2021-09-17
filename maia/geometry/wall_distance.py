from typing import List, Tuple
import numpy as np
from functools import reduce
from mpi4py import MPI
import logging as LOG

import Converter.Internal as I
import Converter.PyTree   as C

import Pypdm.Pypdm as PDM

import maia.sids.cgns_keywords  as CGK
import maia.sids.sids           as SIDS
import maia.sids.Internal_ext   as IE
import maia.sids.pytree as PT

from maia                            import npy_pdm_gnum_dtype  as pdm_dtype
from maia.sids.cgns_keywords         import Label as CGL
from maia.sids                       import conventions as conv
from maia.utils                      import py_utils
from maia.tree_exchange.part_to_dist import discover    as disc
from maia                            import tree_exchange as TE

from maia.geometry.extract_boundary import extract_surf_from_bc
from maia.geometry.extract_boundary2 import extract_surf_from_bc_new
from maia.geometry.geometry         import compute_cell_center

__doc__ = """
CGNS python module which interface the ParaDiGM library for // distance to wall computation .
"""

# ------------------------------------------------------------------------
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

fmt = f'%(levelname)s[{mpi_rank}/{mpi_size}]:%(message)s '
LOG.basicConfig(filename = f"maia_workflow_log.{mpi_rank}.log",
                level    = 10,
                format   = fmt,
                filemode = 'w')

# ------------------------------------------------------------------------
def get_entities_numbering(part_zone, as_pdm=True):
  """
  """
  vtx_ln_to_gn   = I.getVal(IE.getGlobalNumbering(part_zone, 'Vertex'))
  cell_ln_to_gn  = I.getVal(IE.getGlobalNumbering(part_zone, 'Cell'))
  if SIDS.Zone.Type(part_zone) == "Structured":
    face_ln_to_gn = I.getVal(IE.getGlobalNumbering(part_zone, 'Face'))
  else:
    ngons  = [e for e in I.getNodesFromType1(part_zone, 'Elements_t') if SIDS.ElementCGNSName(e) == 'NGON_n']
    assert len(ngons) == 1, "For unstructured zones, only NGon connectivity is supported"
    face_ln_to_gn = I.getVal(IE.getGlobalNumbering(ngons[0], 'Element'))
  if as_pdm:
    vtx_ln_to_gn  = vtx_ln_to_gn.astype(pdm_dtype)
    face_ln_to_gn = face_ln_to_gn.astype(pdm_dtype)
    cell_ln_to_gn = cell_ln_to_gn.astype(pdm_dtype)
  return vtx_ln_to_gn, face_ln_to_gn, cell_ln_to_gn


# ------------------------------------------------------------------------
class WallDistance:

  def __init__(self, part_tree, families=[], mpi_comm=MPI.COMM_WORLD, method="propagation"):
    self.part_tree = part_tree
    self.families  = families
    self.mpi_comm  = mpi_comm
    self.method    = method

    self._register = []
    self.n_vtx_bnd_tot_idx  = [0]
    self.n_face_bnd_tot_idx = [0]

  @property
  def part_tree(self):
    return self._part_tree

  @part_tree.setter
  def part_tree(self, value):
      self._part_tree = value

  @property
  def families(self):
    return self._families

  @families.setter
  def families(self, value):
      self._families = value

  @property
  def mpi_comm(self):
    return self._mpi_comm

  @mpi_comm.setter
  def mpi_comm(self, value):
      self._mpi_comm = value

  @property
  def method(self):
    return self._method

  @method.setter
  def method(self, value):
    if value == "propagation":
      self._method = value
      self._walldist = PDM.DistCellCenterSurf(self.mpi_comm, n_part_surf=1, n_part_vol=1)
    elif value == "cloud":
      self._method = value
      self._walldist = PDM.DistCloudSurf(self.mpi_comm, 1, n_part_surf=1, point_clouds=[])
    else:
      raise ValueError(f"Only 'propagation' or 'cloud' are available for method, {value} given here.")

  @property
  def walldist(self):
    return self._walldist

  def _setup_surf_mesh(self, part_tree, families, comm):
    import time
    debut = time.time()
    face_vtx_bnd, face_vtx_bnd_idx, face_ln_to_gn, \
      vtx_bnd, vtx_ln_to_gn = extract_surf_from_bc(part_tree, families, comm)
    end = time.time()
    print ("Elapsed for bc extraction", end - debut)

    # Keep numpy alive
    for array in (face_vtx_bnd, face_vtx_bnd_idx, face_ln_to_gn, vtx_bnd, vtx_ln_to_gn,):
      self._register.append(array)

    n_face_bnd = face_vtx_bnd_idx.shape[0]-1
    n_vtx_bnd  = vtx_ln_to_gn.shape[0]

    # Get the number of face
    n_face_bnd_t = 0 if face_ln_to_gn.shape[0] == 0 else np.max(face_ln_to_gn)
    n_face_bnd_t = comm.allreduce(n_face_bnd_t, op=MPI.MAX)
    LOG.info(f"setup_surf_mesh [propagation]: n_face_bnd_t = {n_face_bnd_t}")
    # Get the number of vertex
    n_vtx_bnd_t = 0 if vtx_ln_to_gn.shape[0] == 0  else np.max(vtx_ln_to_gn)
    n_vtx_bnd_t = comm.allreduce(n_vtx_bnd_t, op=MPI.MAX)
    LOG.info(f"setup_surf_mesh [propagation]: n_vtx_bnd_t = {n_vtx_bnd_t}")

    self.walldist.n_part_surf = 1
    self.walldist.surf_mesh_global_data_set(n_face_bnd_t, n_vtx_bnd_t)
    self.walldist.surf_mesh_part_set(0, n_face_bnd,
                                     face_vtx_bnd_idx,
                                     face_vtx_bnd,
                                     face_ln_to_gn,
                                     n_vtx_bnd,
                                     vtx_bnd,
                                     vtx_ln_to_gn)

  def _setup_surf_mesh2(self, parts_per_dom, families, comm):
    import time

    all_beg = time.time()
  
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
      self.n_face_bnd_tot_idx.append(self.n_face_bnd_tot_idx[-1] + n_face_bnd_t)

      n_vtx_bnd_t = 0
      for vtx_ln_to_gn in vtx_ln_to_gn_z:
        n_vtx_bnd_t = max(n_vtx_bnd_t, np.max(vtx_ln_to_gn, initial=0))
      n_vtx_bnd_t = comm.allreduce(n_vtx_bnd_t, op=MPI.MAX)
      self.n_vtx_bnd_tot_idx.append(self.n_vtx_bnd_tot_idx[-1] + n_vtx_bnd_t)

      #Shift the face and vertex lngn because PDM does not manage multiple domain. This will avoid
      # overlapping face / vtx coming from different domain but having same id
      for face_ln_to_gn in face_ln_to_gn_z:
        face_ln_to_gn += self.n_face_bnd_tot_idx[i_dom]
      for vtx_ln_to_gn in vtx_ln_to_gn_z:
        vtx_ln_to_gn += self.n_vtx_bnd_tot_idx[i_dom]

      #Extended global lists
      face_vtx_bnd_l.extend(face_vtx_bnd_z)
      face_vtx_bnd_idx_l.extend(face_vtx_bnd_idx_z)
      vtx_bnd_l.extend(vtx_bnd_z)
      face_ln_to_gn_l.extend(face_ln_to_gn_z)
      vtx_ln_to_gn_l.extend(vtx_ln_to_gn_z)


    all_end = time.time()
    print("Whole process" , all_end - all_beg)

    n_part = len(vtx_bnd_l)
    for i in range(n_part):
    # Keep numpy alive
      for array in (face_vtx_bnd_l[i], face_vtx_bnd_idx_l[i], face_ln_to_gn_l[i], vtx_bnd_l[i], vtx_ln_to_gn_l[i],):
        self._register.append(array)

    #Get global data (total number of faces / vertices)

    LOG.info(f"setup_surf_mesh [propagation]: n_face_bnd_t = {n_face_bnd_t}")
    LOG.info(f"setup_surf_mesh [propagation]: n_vtx_bnd_t = {n_vtx_bnd_t}")

    self.walldist.n_part_surf = n_part
    self.walldist.surf_mesh_global_data_set(self.n_face_bnd_tot_idx[-1], self.n_vtx_bnd_tot_idx[-1])
    
    #Setup partitions
    for i_part in range(n_part):
      n_face_bnd = face_vtx_bnd_idx_l[i_part].shape[0]-1
      n_vtx_bnd  = vtx_ln_to_gn_l[i_part].shape[0]
      self.walldist.surf_mesh_part_set(i_part, n_face_bnd,
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
    self.walldist.vol_mesh_global_data_set(n_cell_t, n_face_t, n_vtx_t)


    for i_part, part_zone in enumerate(part_zones):

      vtx_coords = py_utils.interweave_arrays(SIDS.coordinates(part_zone))
      face_vtx, face_vtx_idx, _ = SIDS.ngon_connectivity(part_zone)

      nfaces  = [e for e in I.getNodesFromType1(part_zone, 'Elements_t') if SIDS.ElementCGNSName(e) == 'NFACE_n']
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
        self._register.append(array)

      self.walldist.vol_mesh_part_set(i_part,
                                      n_cell, cell_face_idx, cell_face, center_cell, cell_ln_to_gn,
                                      n_face, face_vtx_idx, face_vtx, face_ln_to_gn,
                                      n_vtx, vtx_coords, vtx_ln_to_gn)


  def compute(self):
    # 1. Search wall family(ies)
    # ==========================

    #Get a skeleton tree including only Base, Zones and Families
    skeleton_tree = I.newCGNSTree()
    disc.discover_nodes_from_matching(skeleton_tree, [self.part_tree], 'CGNSBase_t', comm, child_list=['Family_t'])
    disc.discover_nodes_from_matching(skeleton_tree, [self.part_tree], 'CGNSBase_t/Zone_t', comm,
        merge_rule = lambda path: conv.get_part_prefix(path))

    # Search families if its are not given
    if not self.families:
      bcwalls = ['BCWall', 'BCWallViscous', 'BCWallViscousHeatFlux', 'BCWallViscousIsothermal']
      for family in IE.getNodesByMatching(skeleton_tree, ['CGNSBase_t', 'Family_t']):
        family_bc = I.getNodeFromType1(family, 'FamilyBC_t')
        if family_bc is not None and I.getValue(family_bc) in bcwalls:
          self.families.append(I.getName(family))

    skeleton_families = [I.getName(f) for f in I.getNodesFromType(skeleton_tree, "Family_t")]
    found_families = any([fn in self.families for fn in skeleton_families])
    print(f"found_families = {found_families}")

    if found_families:

      # Group partitions by original dist domain
      parts_per_dom = list()
      for dbase, dzone in IE.iterNodesWithParentsByMatching(skeleton_tree, 'CGNSBase_t/Zone_t'):
        dzone_path = I.getName(dbase) + '/' + I.getName(dzone)
        parts_per_dom.append(TE.utils.get_partitioned_zones(self.part_tree, dzone_path))

      n_domain = len(parts_per_dom)
      assert(n_domain >= 1)


      if self.method == "propagation" and len(parts_per_dom) > 1:
        raise NotImplementedError("Wall_distance computation with method 'propagation' does not support multiple domains")

      if self.method == "cloud":
        # NB: Paradigm structure is created here -> must be done before setup_surf_mesh
        self.walldist.n_point_cloud = n_domain

      self._setup_surf_mesh2(parts_per_dom, self.families, self.mpi_comm)

      # Prepare mesh depending on method
      if self.method == "cloud":
        from maia.interpolation.interpolate import get_point_cloud
        for i_domain, part_zones in enumerate(parts_per_dom):
          self.walldist.set_n_part_cloud(i_domain, len(part_zones))
          for i_part, part_zone in enumerate(part_zones):
            center_cell, cell_ln_to_gn = get_point_cloud(part_zone)
            self._register.append(center_cell)
            self._register.append(cell_ln_to_gn)
            self.walldist.cloud_set(i_domain, i_part, cell_ln_to_gn.shape[0], center_cell, cell_ln_to_gn)

      elif self.method == "propagation":
        for i_domain, part_zones in enumerate(parts_per_dom):
          self.walldist.n_part_vol = len(part_zones)
          self._setup_vol_mesh(i_domain, part_zones, self.mpi_comm)

      #Compute
      args = ['rank1'] if self.method == 'propagation' else []
      self.walldist.compute(*args)

      # Get results -- OK because name of method is the same for 2 PDM objects
      for i_domain, part_zones in enumerate(parts_per_dom):
        self.get(i_domain, part_zones)

    else:
      raise ValueError(f"Unable to find BC family(ies) : {self.families} in {skeleton_families}.")

    # Free unnecessary numpy
    del self._register

  def get(self, i_domain, part_zones):
    for i_part, part_zone in enumerate(part_zones):

      fields = self.walldist.get(i_domain, i_part) if self.method == "cloud" else self.walldist.get(i_part)

      # Test if FlowSolution already exists or create it
      fs_node = I.getNodeFromType1(part_zone, fs_name)
      if not fs_node or not SIDS.GridLocation(fs_node) == 'CellCenter':
        fs_node = I.newFlowSolution(name=fs_name, gridLocation='CellCenter', parent=part_zone)

      # Wall distance
      wall_dist = np.sqrt(fields['ClosestEltDistance'])
      I.newDataArray('TurbulentDistance', value=wall_dist, parent=fs_node)

      # Closest projected element
      closest_elt_proj = np.copy(fields['ClosestEltProjected'])
      I.newDataArray('ClosestEltProjectedX', closest_elt_proj[0::3], parent=fs_node)
      I.newDataArray('ClosestEltProjectedY', closest_elt_proj[1::3], parent=fs_node)
      I.newDataArray('ClosestEltProjectedZ', closest_elt_proj[2::3], parent=fs_node)

      # Closest gnum element (face)
      closest_elt_gnum = np.copy(fields['ClosestEltGnum'])
      I.newDataArray('ClosestEltGnum', closest_elt_gnum, parent=fs_node)

      # Find domain to which the face belongs
      n_face_bnd_tot_idx = np.array(self.n_face_bnd_tot_idx, dtype=closest_elt_gnum.dtype)
      closest_surf_domain = np.searchsorted(n_face_bnd_tot_idx, closest_elt_gnum-1, side='right') -1
      closest_surf_domain = closest_surf_domain.astype(closest_elt_gnum.dtype)
      I.newDataArray("ClosestEltDomId", value=closest_surf_domain, parent=fs_node)
      I.newDataArray("ClosestEltLocGnum", value=closest_elt_gnum - n_face_bnd_tot_idx[closest_surf_domain], parent=fs_node)

  def dump_times(self):
    self.walldist.dump_times()


# ------------------------------------------------------------------------
def wall_distance(*args, **kwargs):
  walldist = WallDistance(*args, **kwargs)
  walldist.compute()
  walldist.dump_times()
  return walldist

# ------------------------------------------------------------------------
import os, subprocess

def shell_command(cmd):
    doit = subprocess.Popen(cmd, universal_newlines=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (done, fail) = doit.communicate()
    code = doit.wait()
    return done, fail

def compareCGNS(f1, f2):
    return shell_command(['cgnsdiff', '-f', '-d', '-t', '1.e-12', f1, f2])

class DifferentFileException(Exception):

    def __init__(self, run_filename, ref_filename, stdout, stderr):
      self.run_filename = run_filename
      self.ref_filename = ref_filename
      self.stdout = stdout
      self.stderr = stderr

    def __str__(self):
      strout = """DifferentFileException :
Generated CGNS file = {0}
Reference CGNS file = {1}
stdout :
{2}
stderr :
{3}
"""
      return strout.format(self.run_filename, self.ref_filename, self.stdout, self.stderr)

def assertCGNSFileEq(run_filename, ref_filename):
  print("Generated CGNS file = ",run_filename)
  print("Reference CGNS file = ",ref_filename)

  if not os.path.exists(run_filename):
      raise IOError(f"Generated CGNS file '{run_filename}' does not exists.")
  if not os.path.exists(ref_filename):
      raise IOError(f"Reference CGNS file '{ref_filename}' does not exists.")

  # Compare running and reference BC case
  stdout, stderr = compareCGNS(run_filename,
                               ref_filename)
  if stdout == '' and stderr == '':
      pass
  else:
      print("assertCGNSFileEq: raise DifferentFileException from CGNS files.")
      raise DifferentFileException(run_filename,
                                   ref_filename,
                                   stdout, stderr)

def compare_npart(t1, t2):
  zones_t2 = I.getZones(t2)
  I.printTree(zones_t2)
  for zone_t1 in I.getZones(t1):
    print(I.getName(zone_t1))
    zone_t2 = I.getNodeFromName2(t2, zone_t1[0])

    # fs_t1 = I.getNodeFromName1(zone_t1, 'FlowSolution#Centers')
    fs_t1 = I.getNodeFromName1(zone_t1, 'FlowSolution#Init')
    fs_t2 = I.getNodeFromName1(zone_t2, 'FlowSolution#Init')

    wd_t1 = I.getNodeFromName1(fs_t1, 'TurbulentDistance')
    wd_t2 = I.getNodeFromName1(fs_t2, 'TurbulentDistance')
    I.newDataArray("DiffTurbDist", 2*np.abs(wd_t1[1]-wd_t2[1])/np.abs(wd_t1[1]+wd_t2[1]), parent=fs_t1)

    wd_t1 = I.getNodeFromName1(fs_t1, 'ClosestEltGnum')
    wd_t2 = I.getNodeFromName1(fs_t2, 'ClosestEltGnum')
    tmp = np.abs(wd_t1[1]-wd_t2[1])
    I.newDataArray("DiffEltGnum", np.abs(wd_t1[1]-wd_t2[1]), parent=fs_t1)
    C.convertPyTree2File(t1, "diff.cgns")
    # print(f"tmp = {tmp}")

    if any([tmp[i] > 0 for i in range(tmp.shape[0])]):
      count = 0
      for i in range(tmp.shape[0]):
        if tmp[i] > 0:
          print(f"Found tmp > 0 for i : {i} -> tmp = {tmp[i]}")
          count += 1
      raise TypeError(f"Found tmp > 0, count = {count}.")


if __name__ == "__main__":
  import os
  from maia.parallel_tree import parallel_tree, merge_and_save
  from maia.cgns_io      import cgns_io_tree as IOT
  from maia.partitioning import part         as PPA
  from maia.partitioning.load_balancing import setup_partition_weights as DBA
  # filename = "cubeS_join_bnd1.hdf"
  # filename = "cubeU_join_bnd.hdf"
  # filename = "cubeH.hdf"

  # t = C.convertFile2PyTree(filename)
  # I._adaptNGon12NGon2(t)
  # C.convertPyTree2File(t, "cubeS_join_bnd1-new.hdf")
  # sys.exit(1)

  # filename = "cubeS_join_bnd1.hdf"
  # filename = "cubeU_join_bnd-new.hdf"
  # families = ['Wall']
  # filename = "AxiT2-new.hdf"
  # families = ['WALL']

  dirname = "."
  method = "cloud"
  # method = "propagation"
  # rootname = "AxiT0"
  rootname = "AxiT2"

  families = ['WALL']
  # t = C.convertFile2PyTree(filename)
  # I._adaptNGon12NGon2(t)
  # C.convertPyTree2File(t, "AxiT0-new.hdf")
  # sys.exit(1)

  # filename = "AxiT2-tetra-new2.hdf"
  # families = ['WALL']

  rootname = "Rotor37_U_MM2"
  families = ['AUBE', 'MOYEU', 'CARTER', 'BLADE']
  # t = C.convertFile2PyTree(filename)
  # I._adaptNGon12NGon2(t)
  # C.convertPyTree2File(t, "AxiT2-new.hdf")
  # sys.exit(1)
  # filename = "cubeH-new.hdf"
  # families = []
  filename = f"{rootname}-new.hdf"

  dirname = "data"
  filename = "LS89_8K_FromPrevious_NewNGon.hdf"
  # filename = "LS89_16K_NewNGon.hdf"
  families = ['BLADE']
  # filename = "mascot_2_input4elsA_newNGon.cgns"
  # filename = "tatef_next_nogc.hdf"
  filename = "tatef_previous_nogc.hdf"

  families = ['extrados', 'intrados',
  'holes', 'internal_cylinders',
  'plenum', 'plenum_2', 'plenum_3',
  'shaped_holes', 'top', 'bottom']

  fs_name = 'FlowSolution#Init'
  dist_tree = IOT.file_to_dist_tree(os.path.join(dirname, filename), comm)
  PT.rm_children_from_label(dist_tree, "FlowSolution_t")
  # I.printTree(dist_tree)
  # sys.exit(1)
  # methodf = f"{method}-rank1"
  # methodf = f"{method}"

  method = "cloud"
  part_tree = PPA.partitioning(dist_tree, comm, graph_part_tool='ptscotch')
  # C.convertPyTree2File(part_tree, f"part_tree-rank{mpi_rank}-{method}.cgns", 'bin_hdf')
  wall_distance(part_tree, mpi_comm=comm, method=method, families=families)
  # I.printTree(part_tree)
  ptree = parallel_tree(comm, dist_tree, part_tree)
  merge_and_save(ptree, f"{filename}-new-{mpi_size}procs-{method}.cgns")
  # merge_and_save(ptree, f"{rootname}-new-{mpi_size}procs-v1-{method}.cgns")
  # assertCGNSFileEq(f"{rootname}-new-{mpi_size}procs-v1-{method}.cgns", f"{rootname}-new-{mpi_size}procs-v1-{method}-ref.cgns")

  method = "propagation"
  part_tree = PPA.partitioning(dist_tree, comm, graph_part_tool='ptscotch')
  # C.convertPyTree2File(part_tree, f"part_tree-rank{mpi_rank}-{method}.cgns", 'bin_hdf')
  wall_distance(part_tree, mpi_comm=comm, method=method)
  # C.convertPyTree2File(part_tree, f"{rootname}-new-{mpi_rank}rank-{methodf}.hdf")
  I.printTree(part_tree)
  ptree = parallel_tree(comm, dist_tree, part_tree)
  merge_and_save(ptree, f"{filename}-new-{mpi_size}procs-{method}.cgns")
  # merge_and_save(ptree, f"{rootname}-new-{mpi_size}procs-v1-{method}.cgns")
  # assertCGNSFileEq(f"{rootname}-new-{mpi_size}procs-v1-{method}.cgns", f"{rootname}-new-{mpi_size}procs-v1-{method}-ref.cgns")

  # n_part = 2
  # zone_to_parts = DBA.npart_per_zone(dist_tree, comm, n_part)
  # part_tree = PPA.partitioning(dist_tree, comm, graph_part_tool='ptscotch', zone_to_parts=zone_to_parts)
  # # C.convertPyTree2File(part_tree, f"part_tree-rank{mpi_rank}-{method}.cgns", 'bin_hdf')
  # wall_distance(part_tree, mpi_comm=comm, method=method)
  # # C.convertPyTree2File(part_tree, f"{rootname}-new-{mpi_rank}rank-{methodf}.cgns")
  # ptree = parallel_tree(comm, dist_tree, part_tree)
  # merge_and_save(ptree, f"{rootname}-new-{mpi_size}procs-v2-{methodf}.cgns")
  # assertCGNSFileEq(f"{rootname}-new-{mpi_size}procs-v2-{method}.cgns", f"{rootname}-new-{mpi_size}procs-v2-{method}-ref.cgns")

  # if mpi_rank == 0:
  #   t1 = C.convertFile2PyTree(f"{rootname}-new-{mpi_size}procs-v1-{methodf}.cgns")
  #   t2 = C.convertFile2PyTree(f"{rootname}-new-{mpi_size}procs-v2-{methodf}.cgns")
  #   compare_npart(t1, t2)