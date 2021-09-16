from typing import List, Tuple
import copy
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
def create_skeleton_from_part_tree(part_base: List, comm) -> List:
  """
  Args:
      part_base (List): CGNS Base, partitioned tree
      comm (MPI communicator): MPI communicator

  Returns:
      List: A skeleton tree with all Family_t node(s)
  """
  skeleton_tree = I.newCGNSTree()
  disc.discover_nodes_from_matching(skeleton_tree, [part_base], 'CGNSBase_t', comm, child_list=['Family_t'])
  return skeleton_tree

def fill_skeleton_from_part_tree(skeleton_tree: List, part_base: List, families: List, comm) -> List:
  """Summary

  Args:
      skeleton_tree (List): CGNS Base, skeleton tree
      part_base (List): CGNS Base, partitioned tree
      families (List): List of family names
      comm (MPI communicator): MPI communicator

  Returns:
      List: A skeleton tree with all Zone(s) owns BC with FamilyName_t node(s) as wall.
  """
  def is_bc_wall(bc_node):
    if I.getType(bc_node) == 'BC_t' and I.getValue(bc_node) == "FamilySpecified":
      family_name_node = I.getNodeFromType1(bc_node, "FamilyName_t")
      if I.getValue(family_name_node) in families:
        return True
    return False

  def has_zone_bc_wall(zone_node):
    for bc_node in IE.iterNodesByMatching(zone_node, 'ZoneBC_t/BC_t'):
      if is_bc_wall(bc_node):
        return True
    return False

  # Search Zone with concerned BC
  disc.discover_nodes_from_matching(skeleton_tree, [part_base], ['CGNSBase_t', has_zone_bc_wall], comm,
                                    get_value=[False, True],
                                    merge_rule=lambda zpath : '.'.join(zpath.split('.')[:-2]))

  # Search BC
  for skeleton_base, skeleton_zone in IE.iterNodesWithParentsByMatching(skeleton_tree, 'CGNSBase_t/Zone_t'):
    disc.discover_nodes_from_matching(skeleton_zone, I.getZones(part_base), ['ZoneBC_t', is_bc_wall], comm,
                                      child_list=['FamilyName_t', 'GridLocation_t'],
                                      get_value='none')
  return skeleton_tree

# ------------------------------------------------------------------------
def find_bcwall(skeleton_tree: List,
                part_tree: List,
                comm=MPI.COMM_WORLD,
                bcwalls=['BCWall', 'BCWallViscous', 'BCWallViscousHeatFlux', 'BCWallViscousIsothermal']) -> List[str]:
  """Summary

  Args:
      skeleton_tree (List): skeleton tree
      part_tree (List): partitioned tree
      comm (MPI communicator): MPI communicator
      bcwalls (list, optional): Description

  Returns:
      List[str]: list of wall family
  """
  all_families = I.getNodesFromType2(skeleton_tree, 'Family_t')
  for part_zone in I.getNodesFromType(part_tree, 'Zone_t'):
    zone_bcs = I.getNodeFromType1(part_zone, 'ZoneBC_t')
    for bc_node in I.getNodesFromType1(zone_bcs, 'BC_t'):
      if I.getValue(bc_node) == 'FamilySpecified':
        family_name_node = I.getNodeFromType1(bc_node, 'FamilyName_t')
        family_name = I.getValue(family_name_node)
        found_family_name = I.getNodeFromName(all_families, family_name)
        found_family_bc_name = I.getNodeFromType1(found_family_name, 'FamilyBC_t')
        family_type = I.getValue(found_family_bc_name)
        if family_type in bcwalls:
          families.append(family_name)
  return families

# ------------------------------------------------------------------------
class WallDistance:

  def __init__(self, part_tree, families=[], mpi_comm=MPI.COMM_WORLD, method="propagation"):
    self.part_tree = part_tree
    self.families  = families
    self.mpi_comm  = mpi_comm
    self.method    = method

    self._register = []

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

  def _setup_surf_mesh2(self, part_tree, families, comm):
    import time
    debut = time.time()
    face_vtx_bnd_l, face_vtx_bnd_idx_l, face_ln_to_gn_l, \
      vtx_bnd_l, vtx_ln_to_gn_l = extract_surf_from_bc_new(part_tree, families, comm)
    end = time.time()
    print ("Elapsed for bc extraction", end - debut)
    
    n_part = len(vtx_bnd_l)
    for i in range(n_part):
    # Keep numpy alive
      for array in (face_vtx_bnd_l[i], face_vtx_bnd_idx_l[i], face_ln_to_gn_l[i], vtx_bnd_l[i], vtx_ln_to_gn_l[i],):
        self._register.append(array)

    #Get global data (total number of faces / vertices)
    n_face_bnd_t = 0
    for face_ln_to_gn in face_ln_to_gn_l:
      n_face_bnd_t = max(n_face_bnd_t, np.max(face_ln_to_gn, initial=0))
    n_face_bnd_t = comm.allreduce(n_face_bnd_t, op=MPI.MAX)

    n_vtx_bnd_t = 0
    for vtx_ln_to_gn in vtx_ln_to_gn_l:
      n_vtx_bnd_t = max(n_vtx_bnd_t, np.max(vtx_ln_to_gn, initial=0))
    n_vtx_bnd_t = comm.allreduce(n_vtx_bnd_t, op=MPI.MAX)

    LOG.info(f"setup_surf_mesh [propagation]: n_face_bnd_t = {n_face_bnd_t}")
    LOG.info(f"setup_surf_mesh [propagation]: n_vtx_bnd_t = {n_vtx_bnd_t}")

    self.walldist.n_part_surf = n_part
    self.walldist.surf_mesh_global_data_set(n_face_bnd_t, n_vtx_bnd_t)
    
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

  def _setup_vol_mesh(self, i_domain, dist_zone, part_tree, comm):
    # print(f"dist_zone = {dist_zone}")
    n_vtx_t  = SIDS.Zone.n_vtx(dist_zone)
    n_cell_t = SIDS.Zone.n_cell(dist_zone)

    # Get the list of all partition in this domain
    is_same_zone = lambda n:I.getType(n) == CGL.Zone_t.name and conv.get_part_prefix(I.getName(n)) == I.getName(dist_zone)
    part_zones = list(IE.iterNodesByMatching(part_tree, ['CGNSBase_t', is_same_zone]))

    # Get the number of local partition(s)
    n_part = len(part_zones)
    LOG.info(f"setup_vol_mesh: n_part   = {n_part}")
    LOG.info(f"setup_vol_mesh: n_vtx_t  [toto] = {n_vtx_t}")
    LOG.info(f"setup_vol_mesh: n_cell_t [toto] = {n_cell_t}")

    # Get the total number of face and vertex of the configuration
    n_vtx_t  = 0
    n_cell_t = 0
    n_face_t = 0
    for i_part, part_zone in enumerate(part_zones):
      LOG.info(f"setup_vol_mesh: parse Zone [1] : {I.getName(part_zone)}")
      vtx_ln_to_gn   = I.getVal(IE.getGlobalNumbering(part_zone, 'Vertex'))
      cell_ln_to_gn  = I.getVal(IE.getGlobalNumbering(part_zone, 'Cell'))
      if SIDS.Zone.Type(part_zone) == "Structured":
        face_ln_to_gn = I.getVal(IE.getGlobalNumbering(part_zone, 'Face'))
      else:
        ngons  = [e for e in I.getNodesFromType1(part_zone, 'Elements_t') if SIDS.ElementCGNSName(e) == 'NGON_n']
        assert len(ngons) == 1, "For unstructured zones, only NGon connectivity is supported"
        face_ln_to_gn = I.getVal(IE.getGlobalNumbering(ngons[0], 'Element'))
      n_vtx_t  = np.max(vtx_ln_to_gn)
      n_cell_t = np.max(cell_ln_to_gn)
      n_face_t = np.max(face_ln_to_gn)

    #We could retrieve data from dist zone
    n_vtx_t  = comm.allreduce(n_vtx_t , op=MPI.MAX)
    n_cell_t = comm.allreduce(n_cell_t, op=MPI.MAX)
    n_face_t = comm.allreduce(n_face_t, op=MPI.MAX)

    LOG.info(f"setup_vol_mesh: n_vtx_t  = {n_vtx_t}")
    LOG.info(f"setup_vol_mesh: n_cell_t = {n_cell_t}")
    LOG.info(f"setup_vol_mesh: n_face_t = {n_face_t}")

    if self.method == "cloud":
      self.walldist.set_n_part_cloud(i_domain, n_part)
    else:
      self.walldist.n_part_vol = n_part
      self.walldist.vol_mesh_global_data_set(n_cell_t, n_face_t, n_vtx_t)

    for i_part, part_zone in enumerate(part_zones):
      LOG.info(f"setup_vol_mesh: parse Zone [2] : {I.getName(part_zone)}")

      vtx_coords = py_utils.interweave_arrays(SIDS.coordinates(part_zone))
      face_vtx, face_vtx_idx, _ = SIDS.ngon_connectivity(part_zone)

      nfaces  = [e for e in I.getNodesFromType1(part_zone, 'Elements_t') if SIDS.ElementCGNSName(e) == 'NFACE_n']
      assert len(nfaces) == 1, "NFace connectivity is needed for wall distance computing"
      cell_face_idx = I.getVal(I.getNodeFromName(nfaces[0], 'ElementStartOffset'))
      cell_face     = I.getVal(I.getNodeFromName(nfaces[0], 'ElementConnectivity'))

      vtx_ln_to_gn   = I.getVal(IE.getGlobalNumbering(part_zone, 'Vertex')).astype(pdm_dtype)
      cell_ln_to_gn  = I.getVal(IE.getGlobalNumbering(part_zone, 'Cell')).astype(pdm_dtype)
      if SIDS.Zone.Type(part_zone) == "Structured":
        face_ln_to_gn = I.getVal(IE.getGlobalNumbering(part_zone, 'Face')).astype(pdm_dtype)
      else:
        ngons  = [e for e in I.getNodesFromType1(part_zone, 'Elements_t') if SIDS.ElementCGNSName(e) == 'NGON_n']
        assert len(ngons) == 1, "For unstructured zones, only NGon connectivity is supported"
        face_ln_to_gn = I.getVal(IE.getGlobalNumbering(ngons[0], 'Element')).astype(pdm_dtype)

      n_vtx  = vtx_ln_to_gn .shape[0]
      n_cell = cell_ln_to_gn.shape[0]
      n_face = face_ln_to_gn.shape[0]

      assert(SIDS.Zone.n_vtx(part_zone)  == n_vtx)
      assert(SIDS.Zone.n_cell(part_zone) == n_cell)
      assert(SIDS.Zone.n_face(part_zone) == n_face)
      LOG.info(f"setup_vol_mesh: n_vtx  = {n_vtx}")
      LOG.info(f"setup_vol_mesh: n_cell = {n_cell}")
      LOG.info(f"setup_vol_mesh: n_face = {n_face}")

      center_cell = compute_cell_center(part_zone)
      self._register.append(center_cell)
      assert(center_cell.size == 3*n_cell)

      # Keep numpy alive
      for array in (cell_face_idx, cell_face, cell_ln_to_gn,):
        self._register.append(array)
      for array in (face_vtx_idx, face_vtx, face_ln_to_gn,):
        self._register.append(array)
      for array in (vtx_coords, vtx_ln_to_gn,):
        self._register.append(array)

      if self.method == "propagation":
        self.walldist.vol_mesh_part_set(i_part,
                                        n_cell, cell_face_idx, cell_face, center_cell, cell_ln_to_gn,
                                        n_face, face_vtx_idx, face_vtx, face_ln_to_gn,
                                        n_vtx, vtx_coords, vtx_ln_to_gn)
      elif self.method == "cloud":
        self.walldist.cloud_set(i_domain, i_part, n_cell, center_cell, cell_ln_to_gn)
      else:
        raise ValueError(f"Only 'propagation' or 'cloud' are available for method, {value} given here.")

  def compute(self):
    # 1. Search wall family(ies)
    # ==========================
    # Create dist tree with families from CGNS base
    skeleton_tree = create_skeleton_from_part_tree(self.part_tree, self.mpi_comm)
    # I.printTree(skeleton_tree)

    # Search families if its are not given
    if not bool(self.families):
      self.families = find_bcwall(skeleton_tree, self.part_tree, self.mpi_comm)
    print(f"self.families = {self.families}")
    LOG.info(f"self.families = {self.families}")

    skeleton_families = [I.getName(f) for f in I.getNodesFromType(skeleton_tree, "Family_t")]
    found_families = any([fn in self.families for fn in skeleton_families])
    print(f"found_families = {found_families}")

    if found_families:
      # Fill dist tree with zone(s) where family(ies) exist(s)
      skeleton_tree = fill_skeleton_from_part_tree(skeleton_tree, self.part_tree, self.families, self.mpi_comm)
      # I.printTree(skeleton_tree)
      # I.printTree(self.part_tree)
      # C.convertPyTree2File(skeleton_tree, "skeleton_tree-cubeU_join_bnd-new-{}.hdf".format(self.mpi_comm.Get_rank()))

      n_domain = len(I.getZones(skeleton_tree))
      assert(n_domain >= 1)

      if self.method == "cloud":
        # NB: Paradigm structure is created here
        self.walldist.n_point_cloud = n_domain

      # 2. Prepare Surface (Global)
      # ===========================
      self._setup_surf_mesh2(self.part_tree, self.families, self.mpi_comm)

      if self.method == "cloud":
        # 2.1. Prepare Volume
        # -------------------
        dist_zones = list(IE.iterNodesByMatching(skeleton_tree, 'CGNSBase_t/Zone_t'))
        for i_domain, dist_zone in enumerate(dist_zones):
          assert(i_domain == 0)
          self._setup_vol_mesh(i_domain, dist_zone, self.part_tree, self.mpi_comm)

        # 2.2. Compute wall distance
        # --------------------------
        self.walldist.compute()

        # 2.3. Fill the partitioned tree with result(s)
        # ---------------------------------------------
        for i_domain, dist_zone in enumerate(dist_zones):
          self.get(self.part_tree, i_domain)
      else:
        for i_domain, dist_zone in enumerate(IE.iterNodesByMatching(skeleton_tree, 'CGNSBase_t/Zone_t')):
          assert(i_domain == 0)
          # 2.1. Prepare Volume
          # -------------------
          self._setup_vol_mesh(i_domain, dist_zone, self.part_tree, self.mpi_comm)

          # 2.2. Compute wall distance
          # --------------------------
          self.walldist.compute('rank1')

          # 2.3. Fill the partitioned tree with result(s)
          # ---------------------------------------------
          self.get(self.part_tree)
    else:
      raise ValueError(f"Unable to find BC family(ies) : {self.families} in {skeleton_families}.")
    # I.printTree(self.part_tree)

    # Free unnecessary numpy
    del self._register

  def get(self, part_tree, i_domain=0):
    for i_part, part_zone in enumerate(I.getZones(part_tree)):
      fields = self.walldist.get(i_domain, i_part) if self.method == "cloud" else self.walldist.get(i_part)
      LOG.info(f"fields = {fields}")

      # Test if FlowSolution already exists or create it
      fs_node = I.getNodeFromType1(part_zone, fs_name)
      if not fs_node or not SIDS.GridLocation(fs_node) == 'CellCenter':
        fs_node = I.newFlowSolution(name=fs_name, gridLocation='CellCenter', parent=part_zone)

      cell_size = SIDS.Zone.CellSize(part_zone)
      # Wall distance
      wall_dist = np.sqrt(copy.deepcopy(fields['ClosestEltDistance']))
      wall_dist = wall_dist.reshape(cell_size, order='F')
      wall_dist_node = SIDS.newDataArrayFromName(fs_node, 'TurbulentDistance')
      I.setValue(wall_dist_node, wall_dist)
      print(f"vtx_size = {3*cell_size}")

      # Closest projected element
      closest_elt_proj = copy.deepcopy(fields['ClosestEltProjected'])
      closest_elt_proj = closest_elt_proj.reshape(3*cell_size, order='F')
      closest_elt_projx_node = SIDS.newDataArrayFromName(fs_node, 'ClosestEltProjectedX')
      closest_elt_projy_node = SIDS.newDataArrayFromName(fs_node, 'ClosestEltProjectedY')
      closest_elt_projz_node = SIDS.newDataArrayFromName(fs_node, 'ClosestEltProjectedZ')
      I.setValue(closest_elt_projx_node, closest_elt_proj[0::3])
      I.setValue(closest_elt_projy_node, closest_elt_proj[1::3])
      I.setValue(closest_elt_projz_node, closest_elt_proj[2::3])

      # Closest gnum element (face)
      closest_elt_gnum = copy.deepcopy(fields['ClosestEltGnum'])
      closest_elt_gnum = closest_elt_gnum.reshape(cell_size, order='F')
      closest_elt_gnum_node = SIDS.newDataArrayFromName(fs_node, 'ClosestEltGnum')
      I.setValue(closest_elt_gnum_node, closest_elt_gnum)

      # # Wall index
      # wall_index = np.empty(cell_size, dtype='float64', order='F')
      # wall_index.fill(-1);
      # I.newDataArray('TurbulentDistanceIndex', wall_index, parent=fs_node)

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
  wall_distance(part_tree, mpi_comm=comm, method=method)
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