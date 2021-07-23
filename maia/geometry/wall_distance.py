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

from maia.geometry.extract_boundary import extract_surf_from_bc
from maia.geometry.geometry         import compute_center_cell_u


__doc__ = """
CGNS python module which interface the ParaDiGM library for // distance to wall computation .
"""

# ------------------------------------------------------------------------
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

fmt = f"{mpi_rank}:{mpi_size}"
LOG.basicConfig(filename = f"maia_workflow_log.{mpi_rank}.log",
                level    = 10,
                format   = fmt,
                filemode = 'w')

# ------------------------------------------------------------------------
def create_dist_from_part_tree(part_base, comm):
  dist_tree = I.newCGNSTree()
  disc.discover_nodes_from_matching(dist_tree, [part_base], 'CGNSBase_t', comm, child_list=['Family_t'])
  return dist_tree

def fill_dist_from_part_tree(dist_tree, part_base, families, comm):
  def is_bc_wall(bc_node):
    if I.getType(bc_node) == 'BC_t':
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
  disc.discover_nodes_from_matching(dist_tree, [part_base], ['CGNSBase_t', has_zone_bc_wall], comm,
                                    get_value=[False, True],
                                    merge_rule=lambda zpath : '.'.join(zpath.split('.')[:-2]))

  # Search BC
  for dist_base, dist_zone in IE.getNodesWithParentsByMatching(dist_tree, 'CGNSBase_t/Zone_t'):
    disc.discover_nodes_from_matching(dist_zone, I.getZones(part_base), ['ZoneBC_t', is_bc_wall], comm,
                                      child_list=['FamilyName_t', 'GridLocation_t'],
                                      get_value='none')
  return dist_tree

# ------------------------------------------------------------------------
@SIDS.check_is_zone
def get_center_cell(zone):
  """
  """
  n_cell = SIDS.Zone.n_cell(zone)
  LOG.info(f"n_cell = {n_cell}")

  # Get coordinates
  cx, cy, cz = SIDS.coordinates(zone)
  LOG.info(f"cx = {cx}")

  pdm_nodes     = I.getNodeFromName1(zone, ":CGNS#Ppart")
  vtx_coords    = I.getVal(I.getNodeFromName1(pdm_nodes, "np_vtx_coord"))
  cell_ln_to_gn = I.getVal(I.getNodeFromName1(pdm_nodes, "np_cell_ln_to_gn"))
  vtx_ln_to_gn  = I.getVal(I.getNodeFromName1(pdm_nodes, "np_vtx_ln_to_gn"))
  LOG.info(f"vtx_coords = {vtx_coords}")

  if SIDS.Zone.Type(zone) == "Unstructured":
    element_node = I.getNodeFromType1(zone, CGL.Elements_t.name)
    # NGon elements
    if SIDS.ElementType(element_node) == CGK.ElementType.NGON_n.value:
      face_vtx, face_vtx_idx, ngon_pe = SIDS.face_connectivity(zone)
      center_cell = compute_center_cell_u(n_cell,
                                          cx, cy, cz,
                                          face_vtx,
                                          face_vtx_idx,
                                          ngon_pe)
      # print("cell_center", center_cell)
    else:
      raise NotImplementedError(f"Unstructured Zone {I.getName(zone)} with {SIDS.ElementCGNSName(element_node)} not yet implemented.")
  else:
    raise TypeError(f"Unable to determine the ZoneType for Zone {I.getName(zone)}")

  # > Keep alive
  I.newDataArray("cell_center", center_cell, parent=pdm_nodes)

  return center_cell, cell_ln_to_gn

# ------------------------------------------------------------------------
@SIDS.check_is_zone
def get_zone_ln_to_gn(zone_node):
  """
  """
  pdm_nodes = I.getNodeFromName1(zone_node, ":CGNS#Ppart")
  if pdm_nodes is not None:
    cell_ln_to_gn = I.getVal(I.getNodeFromName1(pdm_nodes, "np_cell_ln_to_gn"))
    face_ln_to_gn = I.getVal(I.getNodeFromName1(pdm_nodes, "np_face_ln_to_gn"))
    vtx_ln_to_gn  = I.getVal(I.getNodeFromName1(pdm_nodes, "np_vtx_ln_to_gn"))
    return cell_ln_to_gn,face_ln_to_gn, vtx_ln_to_gn
  else:
    # I.printTree(zone_node)
    raise ValueError(f"Unable ta access to the node named ':CGNS#Ppart' in Zone '{I.getName(zone_node)}'.")

# ------------------------------------------------------------------------
@SIDS.check_is_zone
def get_zone_info(zone_node):
  """
  """
  pdm_nodes = I.getNodeFromName1(zone_node, ":CGNS#Ppart")
  if pdm_nodes is not None:
    vtx_coords    = I.getVal(I.getNodeFromName1(pdm_nodes, "np_vtx_coord"))
    cell_face_idx = I.getVal(I.getNodeFromName1(pdm_nodes, "np_cell_face_idx"))
    cell_face     = I.getVal(I.getNodeFromName1(pdm_nodes, "np_cell_face"))
    cell_ln_to_gn = I.getVal(I.getNodeFromName1(pdm_nodes, "np_cell_ln_to_gn"))
    face_vtx_idx  = I.getVal(I.getNodeFromName1(pdm_nodes, "np_face_vtx_idx"))
    face_vtx      = I.getVal(I.getNodeFromName1(pdm_nodes, "np_face_vtx"))
    face_ln_to_gn = I.getVal(I.getNodeFromName1(pdm_nodes, "np_face_ln_to_gn"))
    vtx_ln_to_gn  = I.getVal(I.getNodeFromName1(pdm_nodes, "np_vtx_ln_to_gn"))
    return cell_face_idx, cell_face, cell_ln_to_gn, face_vtx_idx, face_vtx, face_ln_to_gn, vtx_coords, vtx_ln_to_gn
  else:
    # I.printTree(zone_node)
    raise ValueError(f"Unable ta access to the node named ':CGNS#Ppart' in Zone '{I.getName(zone_node)}'.")

# ------------------------------------------------------------------------
def setup_surf_mesh(walldist, dist_tree, part_tree, families, comm):
  face_vtx_bnd, face_vtx_bnd_idx, face_ln_to_gn, \
    vtx_bnd, vtx_ln_to_gn = extract_surf_from_bc(part_tree, families, comm)

  n_face_bnd = face_vtx_bnd_idx.shape[0]-1
  n_vtx_bnd  = vtx_ln_to_gn.shape[0]

  # Get the number of face
  n_face_bnd_t = 0 if face_ln_to_gn.shape[0] == 0 else np.max(face_ln_to_gn)
  n_face_bnd_t = comm.allreduce(n_face_bnd_t, op=MPI.MAX)
  LOG.info(f"setup_surf_mesh: n_face_bnd_t = {n_face_bnd_t}")
  # Get the number of vertex
  n_vtx_bnd_t  = 0 if vtx_ln_to_gn.shape[0] == 0  else np.max(vtx_ln_to_gn)
  n_vtx_bnd_t  = comm.allreduce(n_vtx_bnd_t , op=MPI.MAX)
  LOG.info(f"setup_surf_mesh: n_vtx_bnd_t = {n_vtx_bnd_t}")

  part_base = I.getBases(part_tree)[0]
  walldist_nodes = I.getNodeFromNameAndType(part_base, ':CGNS#WallDist', 'UserDefined_t')
  if walldist_nodes is None:
    walldist_nodes = I.createUniqueChild(part_base, ':CGNS#WallDist', 'UserDefined_t')
  I.newDataArray("face_vtx_bnd",     face_vtx_bnd,     parent=walldist_nodes)
  I.newDataArray("face_vtx_bnd_idx", face_vtx_bnd_idx, parent=walldist_nodes)
  I.newDataArray("face_ln_to_gn",    face_ln_to_gn,    parent=walldist_nodes)
  I.newDataArray("vtx_bnd",          vtx_bnd,          parent=walldist_nodes)
  I.newDataArray("vtx_ln_to_gn",     vtx_ln_to_gn,     parent=walldist_nodes)
  # comm.Barrier()
  # if(comm.Get_rank() == 0):
  #   print "surf_mesh_global_data_set ..."
  # comm.Barrier()
  walldist.n_part_surf = 1
  walldist.surf_mesh_global_data_set(n_face_bnd_t, n_vtx_bnd_t)

  # comm.Barrier()
  # if(comm.Get_rank() == 0):
  #   print "surf_mesh_part_set ..."
  # comm.Barrier()
  walldist.surf_mesh_part_set(0, n_face_bnd,
                                 face_vtx_bnd_idx,
                                 face_vtx_bnd,
                                 face_ln_to_gn,
                                 n_vtx_bnd,
                                 vtx_bnd,
                                 vtx_ln_to_gn)



# ------------------------------------------------------------------------
def setup_vol_mesh(walldist, dist_tree, part_tree):
  n_domain = len(I.getZones(dist_tree))
  assert(n_domain >= 1)

  n_part_per_domain = np.zeros(n_domain, dtype=np.int32)
  for i_domain, dist_zone in enumerate(IE.getNodesByMatching(dist_tree, 'CGNSBase_t/Zone_t')):
    assert(i_domain == 0)
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
      cell_ln_to_gn, face_ln_to_gn, vtx_ln_to_gn = get_zone_ln_to_gn(part_zone)
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

    # comm.Barrier()
    # if(comm.Get_rank() == 0):
    #   print "vol_mesh_global_data_set ..."
    # comm.Barrier()
    walldist.n_part_vol = n_part
    walldist.vol_mesh_global_data_set(n_cell_t, n_face_t, n_vtx_t)
    n_part_per_domain[i_domain] = n_part

    for i_part, part_zone in enumerate(part_zones):
      LOG.info(f"setup_vol_mesh: parse Zone [2] : {I.getName(part_zone)}")
      cell_face_idx, cell_face, cell_ln_to_gn, \
      face_vtx_idx, face_vtx, face_ln_to_gn,   \
      vtx_coords, vtx_ln_to_gn = get_zone_info(part_zone)

      n_cell = cell_ln_to_gn.shape[0]
      n_face = face_ln_to_gn.shape[0]
      n_vtx  = vtx_ln_to_gn .shape[0]

      assert(SIDS.Zone.n_vtx(part_zone)  == n_vtx)
      assert(SIDS.Zone.n_cell(part_zone) == n_cell)
      assert(SIDS.Zone.n_face(part_zone) == n_face)
      LOG.info(f"setup_vol_mesh: n_vtx  = {n_vtx}")
      LOG.info(f"setup_vol_mesh: n_cell = {n_cell}")
      LOG.info(f"setup_vol_mesh: n_face = {n_face}")

      center_cell, _ = get_center_cell(part_zone)
      assert(center_cell.size == 3*n_cell)

      walldist_nodes = I.getNodeFromNameAndType(part_zone, ':CGNS#WallDist', 'UserDefined_t')
      if walldist_nodes is None:
        walldist_nodes = I.createUniqueChild(part_zone, ':CGNS#WallDist', 'UserDefined_t')
      I.newDataArray("cell_face_idx",    cell_face_idx,    parent=walldist_nodes)
      I.newDataArray("cell_face",        cell_face,        parent=walldist_nodes)
      I.newDataArray("cell_ln_to_gn",    cell_ln_to_gn,    parent=walldist_nodes)
      I.newDataArray("face_vtx_idx",     face_vtx_idx,     parent=walldist_nodes)
      I.newDataArray("face_vtx",         face_vtx,         parent=walldist_nodes)
      I.newDataArray("face_ln_to_gn",    face_ln_to_gn,    parent=walldist_nodes)
      I.newDataArray("vtx_coords",       vtx_coords,       parent=walldist_nodes)
      I.newDataArray("vtx_ln_to_gn",     vtx_ln_to_gn,     parent=walldist_nodes)

      # comm.Barrier()
      # if(comm.Get_rank() == 0):
      #   print "vol_mesh_part_set ..."
      # comm.Barrier()
      walldist.vol_mesh_part_set(i_part,
                                 n_cell, cell_face_idx, cell_face, center_cell, cell_ln_to_gn,
                                 n_face, face_vtx_idx, face_vtx, face_ln_to_gn,
                                 n_vtx, vtx_coords, vtx_ln_to_gn)

# ------------------------------------------------------------------------
def find_bcwall(dist_tree, part_tree, comm, bcwalls=['BCWall', 'BCWallViscous', 'BCWallViscousHeatFlux', 'BCWallViscousIsothermal']):
  all_families = I.getNodesFromType2(dist_tree, 'Family_t')
  for part_zone in I.getNodesFromType(part_tree, 'Zone_t'):
    zone_bcs = I.getNodeFromType1(part_zone, 'ZoneBC_t')
    for bc_node in I.getNodesFromType1(zone_bcs, 'BC_t'):
      bc_type = I.getValue(bc_node)
      if bc_type == 'FamilySpecified':
        family_name_node = I.getNodeFromType1(bc_node, 'FamilyName_t')
        family_name = I.getValue(family_name_node)
        # print(f"family_name = {family_name}")
        found_family_name = I.getNodeFromName(all_families, family_name)
        found_family_bc_name = I.getNodeFromType1(found_family_name, 'FamilyBC_t')
        family_type = I.getValue(found_family_bc_name)
        # print(f"family_type = {family_type}")
        if family_type in bcwalls:
          families.append(family_name)
  return families


class WallDistance:

  def __init__(self, part_tree, families=[], mpi_comm=MPI.COMM_WORLD):
    self.part_tree = part_tree
    self.families  = families
    self.mpi_comm  = mpi_comm

    self.walldist = PDM.DistCellCenterSurf(mpi_comm)

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

  def compute(self):
    # Create dist tree with families from CGNS base
    skeleton_tree = create_dist_from_part_tree(self.part_tree, self.mpi_comm)
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
      skeleton_tree = fill_dist_from_part_tree(skeleton_tree, self.part_tree, self.families, self.mpi_comm)
      # I.printTree(skeleton_tree)
      # I.printTree(self.part_tree)
      # C.convertPyTree2File(skeleton_tree, "skeleton_tree-cubeU_join_bnd-new-{}.hdf".format(self.mpi_comm.Get_rank()))

      setup_surf_mesh(self.walldist, skeleton_tree, self.part_tree, self.families, self.mpi_comm)
      setup_vol_mesh(self.walldist, skeleton_tree, self.part_tree)

      self.walldist.compute()

      for i_part, part_zone in enumerate(I.getZones(self.part_tree)):
        fields = self.walldist.get(i_part)
        LOG.info(f"fields = {fields}")

        fs_node = I.newFlowSolution(name=fs_name, gridLocation='CellCenter', parent=part_zone)

        cell_size = SIDS.Zone.CellSize(part_zone)
        # Wall distance
        wall_dist = np.sqrt(copy.deepcopy(fields['ClosestEltDistance']))
        wall_dist = wall_dist.reshape(cell_size, order='F')
        I.newDataArray('TurbulentDistance', wall_dist,parent=fs_node)

        # # Wall closest element
        # wall_closest_elt = copy.deepcopy(fields['ClosestEltProjected'])
        # wall_closest_elt = wall_closest_elt.reshape(cell_size, order='F')
        # I.newDataArray('ClosestEltProjected', wall_closest_elt, parent=fs_node)

        # Wall index
        wall_index = np.empty(cell_size, dtype='float64', order='F')
        wall_index.fill(-1);
        I.newDataArray('TurbulentDistanceIndex', wall_index, parent=fs_node)
    else:
      raise ValueError(f"Unable to find BC family(ies) : {self.families} in {skeleton_families}.")
    # I.printTree(self.part_tree)

  def clean_tree(self):
    raise NotImplementedError()

  def dump_times(self):
    self.walldist.dump_times()


def wall_distance(part_tree, families=[], mpi_comm=MPI.COMM_WORLD):
  walldist = WallDistance(part_tree, families, mpi_comm)
  walldist.compute()
  walldist.dump_times()
  return walldist



if __name__ == "__main__":
  import maia.parallel_tree as PT
  from maia.cgns_io      import cgns_io_tree as IOT
  from maia.partitioning import part         as PPA
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
  filename = "AxiT2-new.hdf"
  families = ['WALL']
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

  part_tree = PPA.partitioning(dist_tree, comm, graph_part_tool = 'ptscotch')
  wall_distance(part_tree, mpi_comm=comm)
  par_tree = PT.parallel_tree(comm, dist_tree, part_tree)
  PT.merge_and_save(par_tree, "popo.hdf")


