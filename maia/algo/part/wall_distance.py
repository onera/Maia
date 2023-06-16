import time
from mpi4py import MPI
import numpy as np
import warnings

import Pypdm.Pypdm as PDM

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils                      import np_utils
from maia.utils                      import logging as mlog
from maia                            import transfer as TE
from maia.factory.dist_from_part     import discover_nodes_from_matching

from .point_cloud_utils              import get_point_cloud
from maia.algo.part.extract_boundary import extract_surf_from_bc
from maia.algo.part.geometry         import compute_cell_center
from maia.transfer                   import utils as tr_utils


def _are_same_perio_abs(first, second):
  """ Return True if the two periodic transformation are the same in absolute value"""
  first_center, first_angle, first_trans = first
  second_center, second_angle, second_trans = second
  if np.allclose(first_center, second_center):
    if np.allclose(first_angle, second_angle) and np.allclose(first_trans, second_trans):
      return True
    if np.allclose(first_angle, -second_angle) and np.allclose(first_trans, -second_trans):
      return True
  return False

def detect_wall_families(tree, bcwalls=['BCWall', 'BCWallViscous', 'BCWallViscousHeatFlux', 'BCWallViscousIsothermal']):
  """
  Return the list of Families having a FamilyBC_t node whose value is in bcwalls list
  """
  fam_query = lambda n : PT.get_label(n) == 'Family_t' and \
                         PT.get_child_from_label(n, 'FamilyBC_t') is not None and \
                         PT.get_value(PT.get_child_from_label(n, 'FamilyBC_t')) in bcwalls
  return [PT.get_name(family) for family in PT.iter_children_from_predicates(tree, ['CGNSBase_t', fam_query])]


# ------------------------------------------------------------------------
class WallDistance:
  """ Implementation of wall distance. See compute_wall_distance for full documentation.
  """

  def __init__(self, part_tree, mpi_comm, method="cloud", families=[], point_cloud='CellCenter', out_fs_name='WallDistance', perio=True):
    self.part_tree = part_tree
    self.families  = families
    self.mpi_comm  = mpi_comm
    assert method in ["cloud", "propagation"]
    self.method    = method
    self.point_cloud = point_cloud
    self.out_fs_n  = out_fs_name

    self._walldist = None
    self._keep_alive = []
    self._n_vtx_bnd_tot_idx  = [0]
    self._n_face_bnd_tot_idx = [0]
    
    self.perio = perio
    self.periodicities = []
    
  def _shift_id_and_push_in_global_list(self, parts_datas, all_parts_datas, i_dom):

    face_vtx_bnd_z, face_vtx_bnd_idx_z, face_ln_to_gn_z, vtx_bnd_z, vtx_ln_to_gn_z = parts_datas
    face_vtx_bnd_l, face_vtx_bnd_idx_l, face_ln_to_gn_l, vtx_bnd_l, vtx_ln_to_gn_l = all_parts_datas

    #Find the maximal vtx/face id for this initial domain
    n_face_bnd_t = 0
    for face_ln_to_gn in face_ln_to_gn_z:
      n_face_bnd_t = max(n_face_bnd_t, np.max(face_ln_to_gn, initial=0))
    n_face_bnd_t = self.mpi_comm.allreduce(n_face_bnd_t, op=MPI.MAX)
    self._n_face_bnd_tot_idx.append(self._n_face_bnd_tot_idx[-1] + n_face_bnd_t)

    n_vtx_bnd_t = 0
    for vtx_ln_to_gn in vtx_ln_to_gn_z:
      n_vtx_bnd_t = max(n_vtx_bnd_t, np.max(vtx_ln_to_gn, initial=0))
    n_vtx_bnd_t = self.mpi_comm.allreduce(n_vtx_bnd_t, op=MPI.MAX)
    self._n_vtx_bnd_tot_idx.append(self._n_vtx_bnd_tot_idx[-1] + n_vtx_bnd_t)

    #Shift the face and vertex lngn because PDM does not manage multiple domain. This will avoid
    # overlapping face / vtx coming from different domain but having same id
    face_ln_to_gn_z = [face_ln_to_gn + self._n_face_bnd_tot_idx[i_dom] for face_ln_to_gn in face_ln_to_gn_z]
    vtx_ln_to_gn_z  = [vtx_ln_to_gn + self._n_vtx_bnd_tot_idx[i_dom] for vtx_ln_to_gn in vtx_ln_to_gn_z]

    #Extended global lists
    face_vtx_bnd_l.extend(face_vtx_bnd_z)
    face_vtx_bnd_idx_l.extend(face_vtx_bnd_idx_z)
    vtx_bnd_l.extend(vtx_bnd_z)
    face_ln_to_gn_l.extend(face_ln_to_gn_z)
    vtx_ln_to_gn_l.extend(vtx_ln_to_gn_z)
    
  def _dupl_shift_id_and_push_in_global_list(self, parts_datas, all_parts_datas, i_dom, perio):

    vtx_bnd_z = parts_datas[3]
    vtx_bnd_dupl_z = []
    for vtx_bnd in vtx_bnd_z:
      cx = vtx_bnd[0::3]
      cy = vtx_bnd[1::3]
      cz = vtx_bnd[2::3]
      cx, cy, cz = np_utils.transform_cart_vectors(cx, cy, cz, perio[2], perio[0], perio[1]) #Perio is center, angle, trans
      vtx_bnd_dupl_z.append(np_utils.interweave_arrays([cx, cy, cz]))
    
    dupl_parts_data = [l for l in parts_datas]
    dupl_parts_data[3] = vtx_bnd_dupl_z #Udpate with duplicated coords

    self._shift_id_and_push_in_global_list(dupl_parts_data, all_parts_datas, i_dom)
    return dupl_parts_data

  def _setup_surf_mesh(self, parts_per_dom, families, comm):
    """
    Setup the surfacic mesh for wall distance computing
    """

    #This will concatenate part data of all initial domains
    face_vtx_bnd_l = []
    face_vtx_bnd_idx_l = []
    face_ln_to_gn_l = []
    vtx_bnd_l = []
    vtx_ln_to_gn_l = []

    all_parts_datas = [face_vtx_bnd_l, face_vtx_bnd_idx_l, face_ln_to_gn_l, vtx_bnd_l, vtx_ln_to_gn_l]

    i_dom = -1
    for part_zones in parts_per_dom:
      
      i_dom += 1
      parts_datas = extract_surf_from_bc(part_zones, families, comm)
      self._shift_id_and_push_in_global_list(parts_datas, all_parts_datas, i_dom)
      
      if self.perio:
        parts_surf_to_dupl_l = [parts_datas]
        for perio_val in self.periodicities:
          perio_val_wrong = (perio_val[1], perio_val[0], perio_val[2]) ## TODO PAS BON ! Erreur reproduite pour pas casser les tests
          perio_val_opp = (perio_val[0], -perio_val[1], -perio_val[2]) #Center, angle, translation

          parts_surf_to_dupl_next_l = []
          for parts_surf_to_dupl in parts_surf_to_dupl_l:
            parts_surf_to_dupl_next_l.append(parts_surf_to_dupl)
            i_dom += 1
            dupl_parts_surf = self._dupl_shift_id_and_push_in_global_list(
                    parts_surf_to_dupl, all_parts_datas, i_dom, perio_val_wrong)
            parts_surf_to_dupl_next_l.append(dupl_parts_surf)

            i_dom += 1
            dupl_parts_surf = self._dupl_shift_id_and_push_in_global_list(
                    parts_surf_to_dupl, all_parts_datas, i_dom, perio_val_opp)
            parts_surf_to_dupl_next_l.append(dupl_parts_surf)

          parts_surf_to_dupl_l = parts_surf_to_dupl_next_l

    n_part = len(vtx_bnd_l)
    for i in range(n_part):
    # Keep numpy alive
      for array in (face_vtx_bnd_l[i], face_vtx_bnd_idx_l[i], face_ln_to_gn_l[i], vtx_bnd_l[i], vtx_ln_to_gn_l[i],):
        self._keep_alive.append(array)

    #Get global data (total number of faces / vertices)
    #This create the surf_mesh objects in PDM, thus it must be done before surf_mesh_part_set
    self._walldist.surf_mesh_global_data_set()
    
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
    Setup the volumic mesh for wall distance computing (only for propagation method)
    """
    #Setup global data
    self._walldist.vol_mesh_global_data_set()

    for i_part, part_zone in enumerate(part_zones):

      vtx_coords = np_utils.interweave_arrays(PT.Zone.coordinates(part_zone))
      face_vtx_idx, face_vtx, _ = PT.Zone.ngon_connectivity(part_zone)

      nface = PT.Zone.NFaceNode(part_zone)
      cell_face_idx = PT.get_value(PT.get_child_from_name(nface, 'ElementStartOffset'))
      cell_face     = PT.get_value(PT.get_child_from_name(nface, 'ElementConnectivity'))

      vtx_ln_to_gn, face_ln_to_gn, cell_ln_to_gn = TE.utils.get_entities_numbering(part_zone)

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
    """
    Get results after wall distance computation and store it in the FlowSolution
    node of name self.out_fs_name
    """
    for i_part, part_zone in enumerate(part_zones):

      fields = self._walldist.get(i_domain, i_part) if self.method == "cloud" else self._walldist.get(i_part)

      # Retrieve location
      if self.point_cloud in ['Vertex', 'CellCenter']:
        output_loc = self.point_cloud
      else:
        output_loc = PT.Subset.GridLocation(PT.get_child_from_name(part_zone, self.point_clouds))
      # Test if FlowSolution already exists or create it
      fs_node = PT.get_child_from_name(part_zone, self.out_fs_n)
      if fs_node is None:
        fs_node = PT.new_FlowSolution(name=self.out_fs_n, loc=output_loc, parent=part_zone)
      assert PT.Subset.GridLocation(fs_node) == output_loc
      if output_loc == "CellCenter":
        shape = PT.Zone.CellSize(part_zone)
      elif output_loc == "Vertex":
        shape = PT.Zone.VertexSize(part_zone)
      else:
        raise RuntimeError("Unmanaged output location")

      # Wall distance
      wall_dist = np.sqrt(fields['ClosestEltDistance'])
      PT.new_DataArray('TurbulentDistance', value=wall_dist.reshape(shape,order='F'), parent=fs_node)

      # Closest projected element
      closest_elt_proj = np.copy(fields['ClosestEltProjected'])
      PT.new_DataArray('ClosestEltProjectedX', closest_elt_proj[0::3].reshape(shape,order='F'), parent=fs_node)
      PT.new_DataArray('ClosestEltProjectedY', closest_elt_proj[1::3].reshape(shape,order='F'), parent=fs_node)
      PT.new_DataArray('ClosestEltProjectedZ', closest_elt_proj[2::3].reshape(shape,order='F'), parent=fs_node)

      # Closest gnum element (face)
      closest_elt_gnum = np.copy(fields['ClosestEltGnum'])
      PT.new_DataArray('ClosestEltGnum', closest_elt_gnum.reshape(shape,order='F'), parent=fs_node)

      # Find domain to which the face belongs (mainly for debug)
      n_face_bnd_tot_idx = np.array(self._n_face_bnd_tot_idx, dtype=closest_elt_gnum.dtype)
      closest_surf_domain = np.searchsorted(n_face_bnd_tot_idx, closest_elt_gnum-1, side='right') -1
      closest_surf_domain = closest_surf_domain.astype(closest_elt_gnum.dtype)
      closest_elt_gnuml = closest_elt_gnum - n_face_bnd_tot_idx[closest_surf_domain]
      if self.perio:
        closest_surf_domain = closest_surf_domain//(3**(len(self.periodicities)))
      PT.new_DataArray("ClosestEltDomId", value=closest_surf_domain.reshape(shape,order='F'), parent=fs_node)
      PT.new_DataArray("ClosestEltLocGnum", value=closest_elt_gnuml.reshape(shape,order='F'), parent=fs_node)



  def compute(self):
    """
    Prepare, compute and get wall distance
    """

    #Get a skeleton tree including only Base, Zones and Families
    skeleton_tree = PT.new_CGNSTree()
    discover_nodes_from_matching(skeleton_tree, [self.part_tree], 'CGNSBase_t', self.mpi_comm, child_list=['Family_t'])
    discover_nodes_from_matching(skeleton_tree, [self.part_tree], 'CGNSBase_t/Zone_t', self.mpi_comm,
        merge_rule = lambda path: MT.conv.get_part_prefix(path))
    
        
    if self.method == "cloud":
      is_gc_perio = lambda n: PT.get_label(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t'] \
              and PT.GridConnectivity.isperiodic(n)
      gc_predicate = ['ZoneGridConnectivity_t', is_gc_perio]
      
      # Recover existing periodicities
      for dist_zone_path in PT.predicates_to_paths(skeleton_tree, 'CGNSBase_t/Zone_t'):
        dist_zone = PT.get_node_from_path(skeleton_tree, dist_zone_path)
        part_zones = tr_utils.get_partitioned_zones(self.part_tree, dist_zone_path)
        discover_nodes_from_matching(dist_zone, part_zones, gc_predicate, self.mpi_comm,
          child_list=['GridConnectivityProperty_t', 'GridConnectivityType_t'],
          merge_rule=lambda path: MT.conv.get_split_prefix(path), get_value='leaf')

      all_periodicities, _ = PT.find_periodic_jns(skeleton_tree)
      # Filter periodicities to get only one over two jns
      for perio_val in all_periodicities:
        for u_perio in self.periodicities:
          if _are_same_perio_abs(perio_val, u_perio):
            break
        else:
          self.periodicities.append(perio_val)
      if len(self.periodicities) == 0:
        self.perio = False #Disable perio to avoid unecessary loops

    else:
      warnings.warn("WallDistance do not manage periodicities except for 'cloud' method", RuntimeWarning, stacklevel=2)
      self.perio = False
        
    # Search families if its are not given
    if not self.families:
      self.families = detect_wall_families(skeleton_tree)

    skeleton_families = [PT.get_name(f) for f in PT.iter_nodes_from_label(skeleton_tree, "Family_t")]
    found_families = any([fn in self.families for fn in skeleton_families])

    if found_families:

      # Group partitions by original dist domain
      parts_per_dom = list()
      for dbase, dzone in PT.iter_children_from_predicates(skeleton_tree, 'CGNSBase_t/Zone_t', ancestors=True):
        dzone_path = PT.get_name(dbase) + '/' + PT.get_name(dzone)
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
        if self.perio:
          n_part_surf = n_part_surf*3**(len(self.periodicities))
        self._walldist = PDM.DistCloudSurf(self.mpi_comm, 1, n_part_surf, point_clouds=n_part_per_cloud)


      self._setup_surf_mesh(parts_per_dom, self.families, self.mpi_comm)

      # Prepare mesh depending on method
      if self.method == "cloud":
        for i_domain, part_zones in enumerate(parts_per_dom):
          for i_part, part_zone in enumerate(part_zones):
            points, points_lngn = get_point_cloud(part_zone, self.point_cloud)
            self._keep_alive.extend([points, points_lngn])
            self._walldist.cloud_set(i_domain, i_part, points_lngn.shape[0], points, points_lngn)

      elif self.method == "propagation":
        for i_domain, part_zones in enumerate(parts_per_dom):
          self._walldist.n_part_vol = len(part_zones)
          if len(part_zones) > 0 and PT.Zone.Type(part_zones[0]) != 'Unstructured':
            raise NotImplementedError("Wall_distance computation with method 'propagation' does not support structured blocks")
          self._setup_vol_mesh(i_domain, part_zones, self.mpi_comm)

      #Compute
      self._walldist.compute()

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
def compute_wall_distance(part_tree, comm, *, method="cloud", families=[], point_cloud="CellCenter", out_fs_name="WallDistance", perio=True):
  """Compute wall distances and add it in tree.

  For each volumic point, compute the distance to the nearest face belonging to a BC of kind wall.
  Computation can be done using "cloud" or "propagation" method.

  Note: 
    Propagation method requires ParaDiGMa access and is only available for unstructured cell centered
    NGon connectivities grids. In addition, partitions must have been created from a single initial domain
    with this method.

  Important:
    Distance are computed to the BCs belonging to one of the families specified in families list.
    If list is empty, we try to auto detect wall-like families.
    In both case, families are (for now) the only way to select BCs to include in wall distance computation.
    BCs having no FamilyName_t node are not considered.

  Tree is modified inplace: computed distance are added in a FlowSolution container whose
  name can be specified with out_fs_name parameter.
  
  Args:
    part_tree (CGNSTree): Input partitionned tree
    comm       (MPIComm): MPI communicator
    method ({'cloud', 'propagation'}, optional): Choice of method. Defaults to "cloud".
    point_cloud (str, optional): Points to project on the surface. Can either be one of
      "CellCenter" or "Vertex" (coordinates are retrieved from the mesh) or the name of a FlowSolution
      node in which coordinates are stored. Defaults to CellCenter.
    families (list of str): List of families to consider as wall faces.
    out_fs_name (str, optional): Name of the output FlowSolution_t node storing wall distance data.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #compute_wall_distance@start
        :end-before: #compute_wall_distance@end
        :dedent: 2
  """
  start = time.time()
  walldist = WallDistance(part_tree, comm, method, families, point_cloud, out_fs_name, perio)
  walldist.compute()
  end = time.time()
  #walldist.dump_times()
  mlog.info(f"Wall distance from families {walldist.families} computed ({end-start:.2f} s)")

