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

from maia.sids.cgns_keywords         import Label as CGL
from maia.sids                       import conventions as conv
from maia.tree_exchange.part_to_dist import discover    as disc

# from maia.geometry.extract_boundary  import extract_surf_from_bc
from maia.geometry.extract_boundary2 import extract_surf_from_bc
from maia.geometry.geometry          import get_center_cell

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
    for bc_node in IE.getNodesByMatching(zone_node, 'ZoneBC_t/BC_t'):
      if is_bc_wall(bc_node):
        return True
    return False

  # Search Zone with concerned BC
  disc.discover_nodes_from_matching(skeleton_tree, [part_base], ['CGNSBase_t', has_zone_bc_wall], comm,
                                    get_value=[False, True],
                                    merge_rule=lambda zpath : '.'.join(zpath.split('.')[:-2]))

  # Search BC
  for skeleton_base, skeleton_zone in IE.getNodesWithParentsByMatching(skeleton_tree, 'CGNSBase_t/Zone_t'):
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
    face_vtx_bnd, face_vtx_bnd_idx, face_ln_to_gn, \
      vtx_bnd, vtx_ln_to_gn = extract_surf_from_bc(part_tree, families, comm)

    # Keep numpy alive
    # part_base = I.getBases(part_tree)[0]
    # walldist_nodes = I.getNodeFromNameAndType(part_base, ':CGNS#WallDist', 'UserDefined_t')
    # if walldist_nodes is None:
    #   walldist_nodes = I.createUniqueChild(part_base, ':CGNS#WallDist', 'UserDefined_t')
    # I.newDataArray("face_vtx_bnd",     face_vtx_bnd,     parent=walldist_nodes)
    # I.newDataArray("face_vtx_bnd_idx", face_vtx_bnd_idx, parent=walldist_nodes)
    # I.newDataArray("face_ln_to_gn",    face_ln_to_gn,    parent=walldist_nodes)
    # I.newDataArray("vtx_bnd",          vtx_bnd,          parent=walldist_nodes)
    # I.newDataArray("vtx_ln_to_gn",     vtx_ln_to_gn,     parent=walldist_nodes)
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

  def _setup_vol_mesh(self, i_domain, dist_zone, part_tree, comm):
    # print(f"dist_zone = {dist_zone}")
    n_vtx_t  = SIDS.Zone.n_vtx(dist_zone)
    n_cell_t = SIDS.Zone.n_cell(dist_zone)

    # Get the list of all partition in this domain
    is_same_zone = lambda n:I.getType(n) == CGL.Zone_t.name and conv.get_part_prefix(I.getName(n)) == I.getName(dist_zone)
    part_zones = list(IE.getNodesByMatching(part_tree, ['CGNSBase_t', is_same_zone]))

    # Get the number of local partition(s)
    n_part = len(part_zones)
    LOG.info(f"setup_vol_mesh: n_part   = {n_part}")
    LOG.info(f"setup_vol_mesh: n_vtx_t  [toto] = {n_vtx_t}")
    LOG.info(f"setup_vol_mesh: n_cell_t [toto] = {n_cell_t}")

    # lsum = lambda x, y: x + y
    # n_vtx_t  = reduce(lsum, [SIDS.Zone.n_vtx(z)  for z in part_zones], 0)
    # n_cell_t = reduce(lsum, [SIDS.Zone.n_cell(z) for z in part_zones], 0)
    # n_face_t = reduce(lsum, [SIDS.Zone.n_face(z) for z in part_zones], 0)
    # print(f"n_vtx_t  = {n_vtx_t}")
    # print(f"n_cell_t = {n_cell_t}")
    # print(f"n_face_t = {n_face_t}")

    # Get the total number of face and vertex of the configuration
    n_vtx_t  = 0
    n_cell_t = 0
    n_face_t = 0
    for i_part, part_zone in enumerate(part_zones):
      LOG.info(f"setup_vol_mesh: parse Zone [1] : {I.getName(part_zone)}")
      vtx_ln_to_gn, cell_ln_to_gn, face_ln_to_gn = SIDS.Zone.get_ln_to_gn(part_zone)
      n_vtx_t  = np.max(vtx_ln_to_gn)
      n_cell_t = np.max(cell_ln_to_gn)
      n_face_t = np.max(face_ln_to_gn)
    n_vtx_t  = comm.allreduce(n_vtx_t , op=MPI.MAX)
    n_cell_t = comm.allreduce(n_cell_t, op=MPI.MAX)
    n_face_t = comm.allreduce(n_face_t, op=MPI.MAX)

    # LOG.info(f"n_vtx_t  = {SIDS.Zone.n_vtx(dist_zone)}")
    # LOG.info(f"n_cell_t = {SIDS.Zone.n_cell(dist_zone)}")
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
      vtx_coords, vtx_ln_to_gn, \
      cell_face_idx, cell_face, cell_ln_to_gn, \
      face_vtx_idx, face_vtx, face_ln_to_gn = SIDS.Zone.get_infos(part_zone)

      n_vtx  = vtx_ln_to_gn .shape[0]
      n_cell = cell_ln_to_gn.shape[0]
      n_face = face_ln_to_gn.shape[0]

      assert(SIDS.Zone.n_vtx(part_zone)  == n_vtx)
      assert(SIDS.Zone.n_cell(part_zone) == n_cell)
      assert(SIDS.Zone.n_face(part_zone) == n_face)
      LOG.info(f"setup_vol_mesh: n_vtx  = {n_vtx}")
      LOG.info(f"setup_vol_mesh: n_cell = {n_cell}")
      LOG.info(f"setup_vol_mesh: n_face = {n_face}")

      center_cell, _ = get_center_cell(part_zone)
      self._register.append(center_cell)
      assert(center_cell.size == 3*n_cell)

      # Keep numpy alive
      # walldist_nodes = I.getNodeFromNameAndType(part_zone, ':CGNS#WallDist', 'UserDefined_t')
      # if walldist_nodes is None:
      #   walldist_nodes = I.createUniqueChild(part_zone, ':CGNS#WallDist', 'UserDefined_t')
      # I.newDataArray("cell_face_idx",    cell_face_idx,    parent=walldist_nodes)
      # I.newDataArray("cell_face",        cell_face,        parent=walldist_nodes)
      # I.newDataArray("cell_ln_to_gn",    cell_ln_to_gn,    parent=walldist_nodes)
      # I.newDataArray("face_vtx_idx",     face_vtx_idx,     parent=walldist_nodes)
      # I.newDataArray("face_vtx",         face_vtx,         parent=walldist_nodes)
      # I.newDataArray("face_ln_to_gn",    face_ln_to_gn,    parent=walldist_nodes)
      # I.newDataArray("vtx_coords",       vtx_coords,       parent=walldist_nodes)
      # I.newDataArray("vtx_ln_to_gn",     vtx_ln_to_gn,     parent=walldist_nodes)
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
      self._setup_surf_mesh(self.part_tree, self.families, self.mpi_comm)

      if self.method == "cloud":
        dist_zones = list(IE.getNodesByMatching(skeleton_tree, 'CGNSBase_t/Zone_t'))
        for i_domain, dist_zone in enumerate(dist_zones):
          assert(i_domain == 0)
          # 3. Prepare Volume
          # =================
          self._setup_vol_mesh(i_domain, dist_zone, self.part_tree, self.mpi_comm)

        # 4. Compute wall distance
        # ========================
        self.walldist.compute()

        for i_domain, dist_zone in enumerate(dist_zones):
          # 5. Fill the partitioned tree with result(s)
          # ===========================================
          self.get(self.part_tree, i_domain)
      else:
        for i_domain, dist_zone in enumerate(IE.getNodesByMatching(skeleton_tree, 'CGNSBase_t/Zone_t')):
          assert(i_domain == 0)
          # 3. Prepare Volume
          # =================
          self._setup_vol_mesh(i_domain, dist_zone, self.part_tree, self.mpi_comm)

          # 4. Compute wall distance
          # ========================
          self.walldist.compute()

          # 5. Fill the partitioned tree with result(s)
          # ===========================================
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



if __name__ == "__main__":
  import maia.parallel_tree as PT
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

  # filename = "AxiT0-new.hdf"
  # families = ['WALL']
  filename = "AxiT2-new.hdf"
  families = ['WALL']
  # t = C.convertFile2PyTree(filename)
  # I._adaptNGon12NGon2(t)
  # C.convertPyTree2File(t, "AxiT0-new.hdf")
  # sys.exit(1)

  # filename = "AxiT2-tetra-new2.hdf"
  # families = ['WALL']

  # filename = "Rotor37_U_MM2-new.hdf"
  # families = ['AUBE', 'MOYEU', 'CARTER', 'BLADE']
  # t = C.convertFile2PyTree(filename)
  # I._adaptNGon12NGon2(t)
  # C.convertPyTree2File(t, "AxiT2-new.hdf")
  # sys.exit(1)
  # filename = "cubeH-new.hdf"
  # families = []

  fs_name = 'FlowSolution#Init'
  dist_tree = IOT.file_to_dist_tree(filename, comm)
  I.printTree(dist_tree)
  # sys.exit(1)

  # method = "propagation"
  method = "cloud"

  part_tree = PPA.partitioning(dist_tree, comm, graph_part_tool='ptscotch')
  C.convertPyTree2File(part_tree, f"part_tree-v10-rank{mpi_rank}.cgns", 'bin_hdf')
  wall_distance(part_tree, mpi_comm=comm, method=method)
  C.convertPyTree2File(part_tree, f"AxiT0-new-v10-{mpi_rank}rank.hdf")
  ptree = PT.parallel_tree(comm, dist_tree, part_tree)
  PT.merge_and_save(ptree, f"AxiT0-new-{mpi_size}procs-v10-cloud.cgns")

  n_part = 2
  zone_to_parts = DBA.npart_per_zone(dist_tree, comm, n_part)
  part_tree = PPA.partitioning(dist_tree, comm, graph_part_tool='ptscotch', zone_to_parts=zone_to_parts)
  C.convertPyTree2File(part_tree, f"part_tree-v20-rank{mpi_rank}.cgns", 'bin_hdf')
  wall_distance(part_tree, mpi_comm=comm, method=method)
  C.convertPyTree2File(part_tree, f"AxiT0-new-v20-{mpi_rank}rank.cgns")
  ptree = PT.parallel_tree(comm, dist_tree, part_tree)
  PT.merge_and_save(ptree, f"AxiT0-new-{mpi_size}procs-v20-cloud.cgns")
