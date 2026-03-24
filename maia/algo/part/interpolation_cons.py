
from adaptathon.utils.cgns_to_pmn import cgns_part_zones_to_pdm_pmesh_nodal

import maia
import maia.algo.part.point_cloud_utils as PCU
import maia.algo.part.closest_points    as CLO
from   maia.factory.dist_from_part      import get_parts_per_blocks
import maia.pytree                      as PT
import maia.pytree.maia                 as MT
import maia.transfer.protocols          as EP
from   maia.utils                       import par_utils, py_utils


from   mpi4py import MPI
import numpy as np

import Pypdm.Pypdm as PDM

def compute_dual_volume(tree, comm):
  from   maia.factory.dist_from_part   import get_parts_per_blocks
  assert(len(get_parts_per_blocks(tree, comm).values()) == 1)
  part_zones = list(get_parts_per_blocks(tree, comm).values())[0]

  from  adaptathon.utils.cgns_to_pmn import cgns_part_zones_to_pdm_pmesh_nodal
  pmn = cgns_part_zones_to_pdm_pmesh_nodal(part_zones, comm, needs_bc=False)

  return pmn.dual_volume_get()

def compress_a_to_b_idx(idx_in):
  referenced =  np.where(np.diff(idx_in) != 0)
  unreferenced =  np.where(np.diff(idx_in) == 0)
  idx_out = idx_in[referenced]
  # print(idx_out)

  return referenced, unreferenced, idx_out

def init_orphan(zones, zones2, volume_from_src,comm,cons_fields_names,cons_B,dim):
  """
  For orphan vertices in the tgt mesh, we use a `Closest` method
  """
  # vertices which have a cell whose vol_ratio < 1.e-15 (==0)
  # theses vertices are completely outside the fluid domain should be initialized differently
  # here we choose the Closest strategy


  # phase 1 transfer the cell-based flag to a vtx-based flag
  # any vtx that uses a cell which is flagged will be flagged
  flagged_cell = [np.zeros(PT.Zone.n_cell(zone), dtype=int) for zone in zones2]

  volume2 = [PT.get_value(PT.get_node_from_path(zone, _measure_name(dim))) for zone in zones2]

  if(len(zones2) != 0):
    for i,_ in enumerate(zones2):
      # [0] is for np.where
      flagged_cell[i][np.where(volume_from_src[i]/volume2[i] < 1.e-15)[0]] = 1
      flagged_cell[i] = np.repeat(flagged_cell[i],dim+1)

  primal_vtx_id = [MT.Zone.vtx_globalnumbering(zone) for zone in zones2]

  # MT.Zone.vtx_globalnumbering
  tri_vtx2 =  [gnum[PT.get_np_value(
        PT.find_child_from_name(PT.find_child_from_predicate(
        zone, PT.pred.is_element_of_type(_simplicial_elt_type(dim))), 'ElementConnectivity')) -1] for gnum,zone in zip(primal_vtx_id,zones2)]

  val = 0
  if(len(zones2) != 0):
    val = max(max(MT.Zone.vtx_globalnumbering(z)) for z in zones2)

  nb_tot_vertex = comm.allreduce(val, MPI.MAX)
  vtx_distri  = par_utils.uniform_distribution(nb_tot_vertex,  comm)

  flag_vtx_distrib = EP.part_to_block(flagged_cell,
                                      vtx_distri, tri_vtx2, comm, reduce_op=EP.ReduceOp.MAX, gnum_offset=1)


  flags_vtx = EP.block_to_part(flag_vtx_distrib, vtx_distri, primal_vtx_id, comm, gnum_offset=1)
  orphan_vtx = [np.where(val == 1)[0].astype(np.int32) for val in flags_vtx]

  #
  nb_tot_orphan = 0
  for i, _ in enumerate(zones2):
    nb_tot_orphan += orphan_vtx[i].size
  nb_tot_orphan = comm.allreduce(nb_tot_orphan,MPI.SUM)

  if(comm.rank == 0 and nb_tot_orphan > 0):
    print(f'total orphan vtx (init strategy: Closest): {nb_tot_orphan}')

  # create transfer protocol for flagged vtx
  # ici on fait une sorte de Closest sur l'ensemble de noeud `flagged_vtx`
  closest_out:List[List[Dict[str, VStrideArray]]]  = []

  src_loc = "Vertex"
  tgt_loc = "Vertex"
  src_clouds = [[PCU.get_point_cloud(part, src_loc) for part in zones]]
  tgt_clouds = [[PCU.get_point_cloud(part, tgt_loc) for part in zones2]]
  tgt_need_shift = True
  tgt_clouds = [[PCU.extract_sub_cloud(*cloud, flag) for cloud,flag in zip(tgt_clouds[0], orphan_vtx)]]

  closest_out = CLO._mdom_closest_points(src_clouds, tgt_clouds, comm, False, n_pts=1, need_shift=tgt_need_shift)

  all_closest = py_utils.to_flat_list(closest_out)

  ptp = PDM.PartToPart(
    comm,
    [tgt_cloud[1] for tgt_cloud in tgt_clouds[0]],
    [src_cloud[1] for src_cloud in src_clouds[0]],
    [np.arange(closest["closest_src_gnum"].size+1,dtype=np.int32) for closest in all_closest],
    [closest["closest_src_gnum"] for closest in all_closest]
  )

  for field in cons_fields_names:
    to_swap = [PT.get_value(PT.get_node_from_path(zone,f"Fields@Vertex@End/{field}")) for zone in zones]

    request = ptp.reverse_iexch(
    PDM._PDM_MPI_COMM_KIND_P2P,
    PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART2,
    to_swap,
    )

    _, temp= ptp.reverse_wait(request)

    for i, (flag_vtx, _temp) in enumerate(zip(orphan_vtx, temp)):
      cons_B[field][i][flag_vtx] = _temp

def calc_volume(comm,volume):
  """sums volume on all zones"""

  tot_vol = 0.
  for val in volume:
    tot_vol = np.sum(val)
  tot_vol =  comm.allreduce(tot_vol, MPI.SUM)

  return tot_vol

def _measure_name(dim):
  if(dim == 2):
    return 'Geometry_2d/Measure'
  else:
    return 'Geometry_3d/Measure'

def _simplicial_elt_type(dim):
  if(dim == 2):
    return 'TRI_3'
  else:
    return 'TETRA_4'

def transfer_cons_vars_vtx_2_cell(zones, dim, comm, containers, flowSol_iter):
  """
  Simple conservative transfer from vtx to cell on simplicial surface/volume meshes

  Returns : dict of vals at cells and integral value (volume + containers)
  """
  integral = dict()

  integral_vars_node  = dict()
  cons_vars_cells = dict()
  for field in containers:
    integral_vars_node[field] = [PT.get_value(PT.get_node_from_name(PT.get_node_from_name(zone,"Fields@Vertex@"+flowSol_iter), field)) for zone in zones]

  volume1 = [PT.get_value(PT.get_node_from_path(zone, _measure_name(dim))) for zone in zones]
  integral['volume'] = calc_volume(comm,volume1)

  cell_vtx_idx = [np.arange(0, (PT.Zone.n_cell(zone)+1)*(dim+1), dim+1, dtype=np.int32) for zone in zones]

  # TODO: assert all elements are simplices

  tri_vtx = [PT.get_np_value(PT.find_child_from_name(PT.find_child_from_predicate(zone, PT.pred.is_element_of_type(_simplicial_elt_type(dim))), 'ElementConnectivity')) for zone in zones]
  for field in containers:
    cons_vars_cells[field] = [np.add.reduceat(integral_vars_node[field][i][tri_vtx[i]-1]/(dim+1), cell_vtx_idx[i][:-1]) for i, _ in enumerate(zones)]
  mass1_temp = dict()

  if(comm.rank == 0):
    print("here")

  for field in containers:
    mass1_temp[field] = 0

  for field in containers:
    for i, _ in enumerate(zones):
      mass1_temp[field] += np.sum(cons_vars_cells[field][i] * volume1[i])
    integral[field] = comm.allreduce(mass1_temp[field], MPI.SUM)
  # TODO: fin a extraire

  return cons_vars_cells, integral

def _reduce_val_at_cells(zones, dim, comm, containers, cons_vars_subcells, a_to_b):
  """
  intput: `cons_vars_subcells` contains vals of conservative fields on intersection cells (subcells)

  returns: cons variables at cells and integral of those values
  """

  cons_B_cell = dict()
  integral = dict()

  volume = [PT.get_value(PT.get_node_from_path(zone, _measure_name(dim))) for zone in zones]
  integral['volume'] = calc_volume(comm,volume)

  if("a_to_b_idx" in a_to_b):
    referenced_b, _, new_a_to_b_idx = compress_a_to_b_idx(a_to_b["a_to_b_idx"])

  for field in containers:
    # FIXME: this [0] shouldn't be
    cons_B_cell[field] = [np.zeros(PT.Zone.n_cell(zone), dtype = cons_vars_subcells[field][0].dtype) for zone in zones]

    if("a_to_b_weight" in a_to_b):
      cons_B_cell[field][0][referenced_b] = \
        np.add.reduceat(cons_vars_subcells[field][0] * a_to_b['a_to_b_weight']/(dim+1) , \
                        new_a_to_b_idx)

  volume_from_src = [np.zeros(PT.Zone.n_cell(zone),dtype=np.float64) for zone in zones]

  #FIXME:
  if("a_to_b_weight" in a_to_b):
    volume_from_src[0][referenced_b] = np.add.reduceat(a_to_b['a_to_b_weight'],new_a_to_b_idx)

  return cons_B_cell, integral, volume_from_src


def update_mass_2(containers, zones, cons_B_cells,comm, integral):
  """
  modifies in-place `integral` to add the sum of all values in cons_B_cells
  """

  for field in containers:
    temp = 0
    for i, _ in enumerate(zones):
      temp += np.sum(cons_B_cells[field][i])
    integral[field] = comm.allreduce(temp,MPI.SUM)


def calculate_mesh_intersection(part_tree, part_tree2, comm, dim):
  """
  wrapping of PDM.MeshIntersection and its compute() method

  returns a PDM.MeshIntersection object
  """
  list_pmn1 = list()
  list_pmn2 = list()
  for part_zones in get_parts_per_blocks(part_tree, comm).values():
    list_pmn1.append(cgns_part_zones_to_pdm_pmesh_nodal(part_zones, comm, needs_bc=False))

  for part_zones in get_parts_per_blocks(part_tree2, comm).values():
    list_pmn2.append(cgns_part_zones_to_pdm_pmesh_nodal(part_zones, comm, needs_bc=False))

  mi2 = PDM.MeshIntersection(comm,
                            PDM._PDM_MESH_INTERSECTION_KIND_WEIGHT,
                            dim,
                            dim,
                            1,
                            1)

  #
  # /!\ CAREFUL HERE : mesh1 is part_2, mesh2 is part_1 /!\
  #
  mi2.part_nodal_set(1, list_pmn1[0])
  mi2.part_nodal_set(0, list_pmn2[0])

  mi2.compute()

  if(comm.rank == 0):
    print("intersection computed")
  return mi2

def cut_cell_correction(zones, dim, volume_from_src, comm, cons_B_cells, containers, OUTPUT=True):
  """
  for curved surfaces we do a non-conservative correction for cells in the target_mesh which have received only partial information from the src_mesh
  """

  volume = [PT.get_value(PT.get_node_from_path(zone, _measure_name(dim))) for zone in zones]

  vol_ratio = [volume_from_src[i]/volume[i] for i, _ in enumerate(zones)]

  # le [0] est obligatoire avec le np.where
  to_correct_idx = [np.where(volume_from_src[i]/volume[i]*(1- volume_from_src[i]/volume[i]) >  1.e-15)[0].astype(np.int32) for i, _ in enumerate(zones)]

  nb_partial_cells = 0
  for i, _ in enumerate(zones):
    nb_partial_cells += to_correct_idx[i].size

  nb_partial_cells += comm.allreduce(nb_partial_cells,MPI.SUM)

  if(comm.rank == 0 and OUTPUT):
    print('************')
    print(f'Exotic cases:')
    print(f'total partial intersection in primal mesh: {nb_partial_cells}')

    # TODO: code volume change
    # print(f'change in simulated volume: {change_vol}')
  min_val = 1.0
  for i,_ in enumerate(zones):
    min_val = min(min_val, np.min(vol_ratio[i][to_correct_idx]))

  max_amplification_factor = comm.allreduce(np.max(1/min_val),MPI.MAX)
  if(comm.rank == 0 and OUTPUT):
    print(f'max conservative amplification factor    : {max_amplification_factor}')

  # non cons correction
  for field in containers:
    for i, _ in enumerate(zones):
      cons_B_cells[field][i][to_correct_idx] /= vol_ratio[i][to_correct_idx]

def _volume_change_stats(volume_from_src, integral1, integral2, comm, OUTPUT=True):
  """
  simple analysis of the calculation volume associated with the conservative conservative variable
  """

  total_vol_received = np.sum(volume_from_src)

  total_vol_received = comm.allreduce(total_vol_received, MPI.SUM)
  if(comm.rank == 0 and OUTPUT):
    eps_vol = 1.e-16
    print('******')
    print("Total fluid volume from src:",integral1["volume"] )
    print("Fluid Volume received :",total_vol_received)
    print("Fluid volume tgt:",integral2["volume"])
    print("Discrepency:",np.abs(integral2["volume"]-total_vol_received), np.abs(integral2["volume"]-total_vol_received)/(integral2["volume"]+eps_vol)*100, "%")

def transfer_cell_to_vtx(zones2, comm, dim, cons_B_cells, containers, part_tree2):

  for field in containers:
    for i,_ in enumerate(zones2):
      cons_B_cells[field][i] = np.repeat(cons_B_cells[field][i],(dim+1))

  val = 0
  if(len(zones2) != 0):
    val = max(max(MT.Zone.vtx_globalnumbering(z)) for z in zones2)

  nb_tot_vertex = comm.allreduce(val, MPI.MAX)

  primal_vtx_id = [MT.Zone.vtx_globalnumbering(zone) for zone in zones2]
  tri_vtx2 =  [gnum[PT.get_np_value(
        PT.find_child_from_name(PT.find_child_from_predicate(
        zone, PT.pred.is_element_of_type(_simplicial_elt_type(dim))), 'ElementConnectivity')) -1] for gnum,zone in zip(primal_vtx_id,zones2)]

  vtx_distri  = par_utils.uniform_distribution(nb_tot_vertex,  comm)

  cons_O2_distrib = EP.part_to_block(cons_B_cells,
                                      vtx_distri, tri_vtx2, comm, reduce_op=EP.ReduceOp.SUM, gnum_offset=1)

  dual_vols_B = compute_dual_volume(part_tree2,comm)
  cons_B = EP.block_to_part(cons_O2_distrib, vtx_distri, primal_vtx_id, comm, gnum_offset=1)#[0] / surf_Dual

  for i, _ in enumerate(zones2):
    for field in containers:
      cons_B[field][i]     /= dual_vols_B[i]
      # print(field,cons_B[field][i])

  return cons_B

def transfer_to_part_2(mesh_intersection,zones,zones2, containers,cons_vars_cells):
  ptp = mesh_intersection.part_to_part_get()
  gnum1_come_from = ptp.get_gnum1_come_from()
  referenced = ptp.get_referenced_lnum2()[0]

  a_to_b = dict()
  if(len(zones) > 0 and len(zones2)> 0 ):
    a_to_b = mesh_intersection.a_to_b_get(0)

  # indirection array for gnum1_come_from
  idx_cells_to_transfer = [np.repeat(referenced, np.diff((gnum1_come_from[i]["come_from_idx"]))) for i,_ in enumerate(zones)]

  # indirection in the data from part1 using idx_cells_to_transfer (by-product of gnum1_come_from)
  cons_copied = dict()
  for field in containers:
    # idx is 1-based
    cons_copied[field] = [cons_vars_cells[field][i][idx_cells_to_transfer[i] - 1] for i, _ in enumerate(zones)]

  # region - Transfer to mesh B
  cons_vars_subcells_B = dict()
  for field in containers:
    request = ptp.reverse_iexch(
    PDM._PDM_MPI_COMM_KIND_P2P,
    PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_GNUM1_COME_FROM,
    cons_copied[field],
    )

    _, temp = ptp.reverse_wait(request)
    cons_vars_subcells_B[field] = temp
  return cons_vars_subcells_B, a_to_b

def interpCons(containers, part_tree, part_tree2, comm,  flowSol_iter="End",
  OUTPUT=True):
  '''First-order accurate interpolation, conservative between two 2D or 3D part trees made only of simplicial elements on their flowSolution. (TRI_3 or TETRA_4)
  - Only the conservative variables can be used 'Density', 'Momentum', 'EnergyStagnationDensity' and turbulent/multispecies-specific variables -> assert on the containers before calling interpCons()
  - Works for vertex-centered based solution, but could also be called on cell-centered solutions
  Vertex-based solution is transferred at cells on mesh 1 then to cells on mesh 2 and finally on vertices

  For partials cells, a non-conservative correction is implemented
  For orphan issues a Closest method is called

  returns Volume and integral values of conservatives variables of both meshes

  '''

  base_n = PT.get_child_from_label(part_tree,"CGNSBase_t")
  dim = PT.Base.CellDimension(base_n)

  # region - Geometry Calculation
  for tree in [part_tree, part_tree2]:
    maia.algo.compute_elements_measure(tree, dim, comm)

  zones = PT.get_all_Zone_t(part_tree)

  # the mesh intersection calculation is only on the primal mesh, thus we transfer the vertex-fields to the cells in a conservative manner
  cons_vars_cells, integral1 = transfer_cons_vars_vtx_2_cell(zones, dim, comm, containers, flowSol_iter)

  # set up transfer protocol ptp
  mesh_intersection = calculate_mesh_intersection(part_tree, part_tree2, comm, dim)
  # get info from the ptp
  zones2 = PT.get_all_Zone_t(part_tree2)
  # FIXME: strangely if a_to_b is not _get() here, then it crashes.
  # probably some dark python property
  cons_vars_subcells, a_to_b = transfer_to_part_2(mesh_intersection,zones,zones2, containers,cons_vars_cells)

  # sum subcells from second mesh point of view
  cons_B_cells, integral2, volume_from_src = _reduce_val_at_cells(zones2, dim, comm, containers, cons_vars_subcells, a_to_b)

  # display info
  _volume_change_stats(volume_from_src, integral1, integral2, comm, OUTPUT)

  # 1rst correction of the received values
  cut_cell_correction(zones2, dim, volume_from_src,comm, cons_B_cells,containers)

  # in-place modification of integral2 to add values of conservative fields
  update_mass_2(containers, zones2, cons_B_cells, comm, integral2)

  cons_B = transfer_cell_to_vtx(zones2, comm, dim, cons_B_cells, containers, part_tree2)

  # 2nd correction - Closest on orphan vtx
  init_orphan(zones, zones2, volume_from_src, comm, containers, cons_B, dim)

  # put cons_B in part_tree2
  for i,zone in enumerate(zones2):
    # PT.print_tree(zone)
    fs_n = PT.get_node_from_name(zone,"Fields@Vertex@"+flowSol_iter)
    if fs_n is None:
      fs_n = PT.new_FlowSolution(name="Fields@Vertex@"+flowSol_iter, loc="Vertex", parent=zone)
    for field in containers:
      # print(field)
      PT.new_DataArray(field, cons_B[field][i], dtype='R8', parent = fs_n)

  return integral1, integral2