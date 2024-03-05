import copy
import itertools
import mpi4py.MPI      as MPI
import numpy           as np
import pytest_parallel

import maia
import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia.algo.part.point_cloud_utils as PCU

from maia                     import npy_pdm_gnum_dtype    as pdm_gnum_dtype
from maia.algo.dist           import merge_degen_bc        as MDB
from maia.algo.dist           import remove_element        as RME
from maia.algo.dist.merge_ids import merge_distributed_ids
from maia.transfer            import protocols             as EP
from maia.utils               import par_utils

import Pypdm.Pypdm as PDM



@pytest_parallel.mark.parallel(1)
def test_merge_degen_bc(comm):
  #----------------------------
  # Parameters
  nx = 5
  ny = 5
  nz = 5
  sector_angle = np.pi/4
  
  fam_l = ['INLET', 'OUTLET', 'AXIS', 'FARFIELD', 'PER1', 'PER2']
  bc2fam = {}
  bc2fam['Xmin'] = fam_l[0]
  bc2fam['Xmax'] = fam_l[1]
  bc2fam['Ymin'] = fam_l[2]
  bc2fam['Ymax'] = fam_l[3]
  bc2fam['Zmin'] = fam_l[4]
  bc2fam['Zmax'] = fam_l[5]
  
  #----------------------------
  # Generate cube
  dist_tree = maia.factory.generate_dist_block([nx,ny,nz], 'HEXA_8', comm)
  maia.algo.scale_mesh(dist_tree, [nx-1,ny-1,nz-1])
  
  #----------------------------
  # Move nodes to generate sector of cylinder
  theta_x = sector_angle/(nz-1)
  theta_y = 0.
  theta_z = 0.
  
  zone_n = PT.get_node_from_label(dist_tree, 'Zone_t')
  
  coords_n = PT.get_node_from_label(dist_tree, 'GridCoordinates_t')
  coord_x_n, coord_y_n, coord_z_n = [PT.get_node_from_name(coords_n, f"Coordinate{suffix}") for suffix in ['X', 'Y', 'Z']]
  coord_x,   coord_y,   coord_z   = [PT.get_value(coord) for coord in [coord_x_n, coord_y_n, coord_z_n]]

  for i in range(nx-1):
    beg = (i+1)*nx*ny
    end = (i+2)*nx*ny
    # print(i+1, coord_y[beg:end], coord_z[beg:end])
    coord_z[beg:end] = 0.
    coord_x[beg:end], coord_y[beg:end], coord_z[beg:end] = maia.utils.ndarray.np_utils.transform_cart_vectors(coord_x[beg:end], coord_y[beg:end], coord_z[beg:end], rotation_angle = np.array([(i+1)*theta_x,theta_y,theta_z]))
  
  #----------------------------
  # Add families
  base_n = PT.get_all_CGNSBase_t(dist_tree)[0] 
  for bc_n in PT.get_nodes_from_predicates(dist_tree, 'CGNSBase_t/Zone_t/ZoneBC_t/BC_t'):
      PT.set_value(bc_n, 'FamilySpecified')
      PT.update_child(bc_n,'FamilyName',label='FamilyName_t',value=bc2fam[bc_n[0]])
  
  for fam in fam_l:
      PT.update_child(base_n,fam,label='Family_t')
  
  #----------------------------
  # Convert to ngon
  maia.algo.dist.convert_elements_to_ngon(dist_tree, comm)
  # maia.io.dist_tree_to_file(dist_tree, f'/stck/sbouras/dev/dev-Fun/DegeneratedLine/eighth_cylinder_5x5x5_ready_CI.cgns', comm)
  
  ####
  # TODO: change code to be // independant to generate test case
  ####
  #-------------
  # Begin of the test

  zone_names_to_remove = []
  for zone_n in PT.get_nodes_from_label(dist_tree, 'Zone_t'):
    if not PT.Zone.has_nface_elements(zone_n):
      maia.algo.pe_to_nface(zone_n, comm)
    for bc in PT.get_nodes_from_label(zone_n, 'BC_t'):
      fam_n = PT.get_node_from_label(bc, 'FamilyName_t')
      if fam_n is None:
        if bc[0].startswith("BCDegene"):
          # print(zone_n[0], "DegeneratedLine found :", bc[0])
          zone_names_to_remove.append(PT.get_name(zone_n))
        elif bc[0].startswith("BCSymmetry"):
          # print(zone_n[0], "SymmetryPlane found :", bc[0])
          pass
      else:
        fam = PT.get_value(fam_n)
        if (fam == 'AXIS') or (fam == 'DGL'):
          # print(zone_n[0], "AXIS found :", bc[0])
          zone_names_to_remove.append(PT.get_name(zone_n))
        elif fam == 'SIDE1':
          # print(zone_n[0], "SIDE1 found :", bc[0])
          pass
        elif bc[0] == "Ymin":
          # print(zone_n[0], "DegeneratedLine found :", bc[0])
          zone_names_to_remove.append(PT.get_name(zone_n))
  
  new_dist_tree = copy.deepcopy(dist_tree)
  new_base_n = PT.get_node_from_label(new_dist_tree, 'CGNSBase_t')
  for zone_name in zone_names_to_remove:
    PT.rm_node_from_path(new_base_n, zone_name)
  
  for zone_n in PT.get_nodes_from_label(dist_tree, 'Zone_t'):
    
    ngon_n = PT.Zone.NGonNode(zone_n)
    ngon_ec  = PT.get_value(PT.get_node_from_name(ngon_n, 'ElementConnectivity'))
    ngon_eso = PT.get_value(PT.get_node_from_name(ngon_n, 'ElementStartOffset'))
    
    degen_bc_n = None
    intersect_degen_bc_n = None
    for bc in PT.get_nodes_from_label(zone_n, 'BC_t'):
      fam_n = PT.get_node_from_label(bc, 'FamilyName_t')
      if fam_n is None:
        if bc[0].startswith("BCDegene"):
          degen_bc_n = bc
        elif bc[0].startswith("BCSymmetry"):
          intersect_degen_bc_n = bc
      else:
        fam = PT.get_value(fam_n)
        if (fam == 'AXIS') or (fam == 'DGL'):
          degen_bc_n = bc
        elif fam == 'SIDE1':
          intersect_degen_bc_n = bc
        elif bc[0] == "Ymin":
          degen_bc_n = bc
        elif bc[0] == "Zmax":
          intersect_degen_bc_n = bc
    if degen_bc_n is None:
      continue
    if intersect_degen_bc_n is None: #Search in GC the first perio
      for gc in PT.get_nodes_from_predicates(zone_n, ['ZoneGridConnectivity','GridConnectivity_t']):
        if PT.get_nodes_from_label(gc, 'Periodic_t') is not None:
          intersect_degen_bc_n = gc
          break
    if intersect_degen_bc_n is None:
      print("Error : 'intersect_degen_bc' is not defined !")
      exit()
    # print(zone_n[0], degen_bc_n[0], intersect_degen_bc_n[0])
    # exit()
    
    pl_degen_bc = PT.get_value(PT.get_node_from_name(degen_bc_n, 'PointList'))[0]
    pl_intersect_degen_bc = PT.get_value(PT.get_node_from_name(intersect_degen_bc_n, 'PointList'))[0]
    # print([comm.rank,pl_degen_bc])
    # exit()
    
    # Listes continues
    def distribute_unique_vtx_ids_from_face_ids(pl_faces, ngon_n, comm):
      # Get the nodes ids of all faces in pl_faces
      __, nodes_pl = maia.algo.dist.vertex_list.face_ids_to_vtx_ids(pl_faces, ngon_n, comm)
      # Make unique
      PTB = EP.PartToBlock(None, [nodes_pl], comm, weight=True, keep_multiple=False)
      nodes_pl = PTB.getBlockGnumCopy()
      # Because result could be badly distributed, redistribute it
      distrib_nodes_pl_init   = par_utils.gather_and_shift(len(nodes_pl), comm, pdm_gnum_dtype)
      distrib_nodes_pl_wanted = par_utils.uniform_distribution(distrib_nodes_pl_init[-1], comm)
      return EP.block_to_block(nodes_pl, distrib_nodes_pl_init, distrib_nodes_pl_wanted, comm)
      
    __, nodes_degen_bc           = maia.algo.dist.vertex_list.face_ids_to_vtx_ids(pl_degen_bc, ngon_n, comm)
    __, nodes_intersect_degen_bc = maia.algo.dist.vertex_list.face_ids_to_vtx_ids(pl_intersect_degen_bc, ngon_n, comm)
    
    PTB = EP.PartToBlock(None, [nodes_degen_bc], comm, weight=True, keep_multiple=False)
    nodes_degen_bc = PTB.getBlockGnumCopy()
  
    distrib_nodes_degen_bc_init = par_utils.gather_and_shift(len(nodes_degen_bc), comm, pdm_gnum_dtype)
    #Result is badly distributed, we can do a BlockToBlock to have a uniform distribution
    distrib_nodes_degen_bc_wanted      = par_utils.uniform_distribution(distrib_nodes_degen_bc_init[-1], comm)
    nodes_degen_bc = EP.block_to_block(nodes_degen_bc, distrib_nodes_degen_bc_init, distrib_nodes_degen_bc_wanted, comm)
    # print([comm.rank, len(nodes_degen_bc), nodes_degen_bc])
  
    PTB = EP.PartToBlock(None, [nodes_intersect_degen_bc], comm, weight=True, keep_multiple=False)
    nodes_intersect_degen_bc = PTB.getBlockGnumCopy()
    # print([comm.rank, len(nodes_intersect_degen_bc), nodes_intersect_degen_bc])
    
    nodes_degen_bc           = distribute_unique_vtx_ids_from_face_ids(pl_degen_bc,           ngon_n, comm)
    nodes_intersect_degen_bc = distribute_unique_vtx_ids_from_face_ids(pl_intersect_degen_bc, ngon_n, comm)
    
    # print(len(nodes_intersect_degen_bc))
    # print(nodes_degen_bc)
    
    # nodes_degen_line = np.intersect1d(nodes_degen_bc,nodes_intersect_degen_bc)
    nodes_degen_bc_in = maia.utils.parallel.algo.gnum_isin(nodes_degen_bc,nodes_intersect_degen_bc, comm)
    nodes_degen_line = nodes_degen_bc[nodes_degen_bc_in]
    # print([comm.rank, len(nodes_degen_line), nodes_degen_line])
    
    # print(len(nodes_degen_line))
    # exit()
    
    zsr1_n = PT.new_ZoneSubRegion(name='DegenLine', point_list=[nodes_degen_line], loc='Vertex', parent=zone_n)
    full_distri1 = par_utils.gather_and_shift(len(nodes_degen_line), comm)
    partial_distri1 = full_distri1[[comm.Get_rank(), comm.Get_rank()+1, comm.Get_size()]]
    PT.maia.newDistribution({"Index" : partial_distri1}, zsr1_n)
    # PT.print_tree(dist_tree, 'toto.txt')
    # # maia.io.dist_tree_to_file(dist_tree, 'toto.cgns', comm)
    zsr2_n = PT.new_ZoneSubRegion(name='NodesFromBCToDelete', point_list=[nodes_degen_bc], loc='Vertex', parent=zone_n)
    full_distri2 = par_utils.gather_and_shift(len(nodes_degen_bc), comm)
    partial_distri2 = full_distri2[[comm.Get_rank(), comm.Get_rank()+1, comm.Get_size()]]
    PT.maia.newDistribution({"Index" : partial_distri2}, zsr2_n)
    # exit()
    # maia.io.dist_tree_to_file(dist_tree, f'toto_{comm.size}.cgns', comm)
    # exit()
  
    # dist_tree_ngon = copy.deepcopy(dist_tree)
    # maia.algo.dist.rearrange_element_sections(dist_tree_ngon, comm)
    # maia.algo.dist.convert_elements_to_ngon(dist_tree_ngon, comm)
    dist_tree_ngon = PT.shallow_copy(dist_tree)
    PT.rm_nodes_from_predicate(
      dist_tree_ngon,
      lambda n: PT.get_label(n)=='Zone_t' and PT.get_name(zone_n) not in PT.get_name(n))
    
    # PT.print_tree(dist_tree_ngon, f'dist_tree_ngon_{comm.rank}p.txt')
    part_tree_ngon = maia.factory.partition_dist_tree(dist_tree_ngon, comm)
    # exit()
    
    pzone_n = PT.get_node_from_label(part_tree_ngon, 'Zone_t')  
    
    extractor_degen_line = maia.algo.part.extract_part.create_extractor_from_zsr(part_tree_ngon, 'DegenLine', comm)
    degen_line_tree = extractor_degen_line.get_extract_part_tree()
    dom_name = PT.get_name(PT.get_node_from_label(dist_tree_ngon, 'Zone_t'))
    dom_path = f'{PT.get_name(PT.get_all_CGNSBase_t(part_tree_ngon)[0])}/{dom_name}'
    parent_vtx = extractor_degen_line.exch_tool_box[dom_path]['parent_elt']['Vertex']
    # print([comm.rank, len(parent_vtx), parent_vtx])
    # exit()
    
    
    maia.algo.part.find_closest_points(degen_line_tree, part_tree_ngon, 'Vertex', comm)
    # PT.print_tree(part_tree_ngon, 'tata.txt')
    # maia.io.part_tree_to_file(part_tree, 'out.cgns', comm)
    
    try:
      src_id = PT.get_value(PT.get_node_from_name(part_tree_ngon, 'SrcId'))
    except TypeError:
      src_id = np.empty(0, dtype=pdm_gnum_dtype)
    # print([comm.rank, "len(src_id) =", len(src_id)])
    # exit()
    
    if pzone_n is None:
      p_nodes_degen_bc = np.zeros((0), dtype=np.int32)
    else:
      pzsr2_n = PT.get_node_from_name(pzone_n, 'NodesFromBCToDelete')
      if pzsr2_n is None:
        p_nodes_degen_bc = np.zeros((0), dtype=np.int32)
      else:
        p_nodes_degen_bc = PT.get_value(PT.get_node_from_name(pzsr2_n, 'PointList'))[0]
      # print(p_nodes_degen_bc)
    # print([comm.rank, len(p_nodes_degen_bc), np.min(p_nodes_degen_bc) if len(p_nodes_degen_bc)>0 else None, np.max(p_nodes_degen_bc) if len(p_nodes_degen_bc)>0 else None])
    # pos_in_degen_line = src_id[p_nodes_degen_bc-1]
    # print([comm.rank, len(pos_in_degen_line), pos_in_degen_line])
    # exit()
    
    # print([comm.rank, PT.print_tree(degen_line_tree)])
    # exit()
    
    # old_to_new_degen_bc_nodes = parent_vtx[pos_in_degen_line-1]
    pdgenline_n = PT.get_node_from_label(degen_line_tree, 'Zone_t') 
    # print([comm.rank, "len(PT.get_value(MT.getGlobalNumbering(pzone_n, 'Vertex'))) =", len(PT.get_value(MT.getGlobalNumbering(pzone_n, 'Vertex')))])
    # if comm.rank == 0:
    #   PT.print_tree(degen_line_tree)
    #   print(PT.get_value(MT.getGlobalNumbering(pdgenline_n, 'Vertex')))
    # exit()
    # part1_gnum_vtx_n = MT.getGlobalNumbering(pzone_n, 'Vertex')
    # part2_gnum_vtx_n = MT.getGlobalNumbering(pdgenline_n, 'Vertex')
    # if part1_gnum_vtx_n is None:
    #   part1_gnum_vtx = []
    # else:
    #   part1_gnum_vtx = PT.get_value(part1_gnum_vtx_n)
    # if part2_gnum_vtx_n is None:
    #   part2_gnum_vtx = []
    # else:
    #   part2_gnum_vtx = PT.get_value(part2_gnum_vtx_n)
    if pzone_n is None:
      part1_gnum_vtx = np.empty(0, dtype=pdm_gnum_dtype)
    else:
      part1_gnum_vtx = PT.get_value(MT.getGlobalNumbering(pzone_n, 'Vertex'))
    if pdgenline_n is None:
      part2_gnum_vtx = np.empty(0, dtype=pdm_gnum_dtype)
    else:
      part2_gnum_vtx = PT.get_value(MT.getGlobalNumbering(pdgenline_n, 'Vertex'))
    ptp = PDM.PartToPart(comm, [part1_gnum_vtx], [part2_gnum_vtx], [np.arange(len(src_id)+1,dtype=src_id.dtype)], [src_id])
    # request1 = ptp.reverse_iexch(PDM._PDM_MPI_COMM_KIND_P2P, PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART1, [parent_vtx])
    request1 = ptp.reverse_iexch(PDM._PDM_MPI_COMM_KIND_P2P, PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART2, [parent_vtx])
    # _, part_data = ptp.wait(request1)
    _, part_data = ptp.reverse_wait(request1)
    # print([comm.rank, f"SB/BA : {len(src_id)} vs {len(part_data[0])} !"])
    # exit()
    # print([comm.rank, "SB : ptp done !"])
    # print([comm.rank, len(part_data[0]), part_data[0]])
    # old_to_new_degen_bc_nodes = part_data[0][pos_in_degen_line-1]
    old_to_new_degen_bc_nodes = part_data[0][p_nodes_degen_bc-1]
    # print(old_to_new_degen_bc_nodes)
    # print([comm.rank, len(old_to_new_degen_bc_nodes), old_to_new_degen_bc_nodes])
    # print([comm.rank, f"SB : len(old_to_new_degen_bc_nodes) = {len(old_to_new_degen_bc_nodes)}"])
    # print([comm.rank, f"SB : len(p_nodes_degen_bc) = {len(p_nodes_degen_bc)}"])
    # exit()
    
    pzsr_n = PT.new_ZoneSubRegion(name='ZSR_DegenLine', point_list=[p_nodes_degen_bc], loc='Vertex', parent=pzone_n)
    PT.new_DataArray("OldToNew", old_to_new_degen_bc_nodes, parent=pzsr_n)
    # PT.maia.newGlobalNumbering({'Index' : np.arange(len(p_nodes_degen_bc), dtype=np.int32)+1}, parent=pzsr_n)
    if pzone_n is None:
      local_gnum = np.empty(0, dtype=pdm_gnum_dtype)
    else:
      local_gnum = PT.get_value(MT.getGlobalNumbering(pzone_n, 'Vertex'))[p_nodes_degen_bc-1]
    zsr_gnum = PCU.create_sub_numbering([local_gnum], comm)[0]
    PT.maia.newGlobalNumbering({'Index' : zsr_gnum}, parent=pzsr_n)
    # print([comm.rank, f"SB : newGlobalNumbering 'Index' = {np.arange(len(p_nodes_degen_bc), dtype=np.int32)+1}"])
    # print([comm.rank, f"SB : zsr_gnum 'Index' = {zsr_gnum}"])
    # print([comm.rank, f"SB : p_nodes_degen_bc = {p_nodes_degen_bc}"])
    # exit()
    # if comm.rank == 1:
    #   PT.print_tree(part_tree_ngon)
    # exit()
    
    maia.transfer.part_tree_to_dist_tree_only_labels(dist_tree_ngon, part_tree_ngon, ['ZoneSubRegion_t'], comm)
    # PT.print_tree(dist_tree_ngon, "titi.txt")
    # exit()
  
    shallow_zone_n  = PT.get_node_from_label(dist_tree_ngon, 'Zone_t')
    
    for bc in PT.get_nodes_from_label(shallow_zone_n, 'BC_t'):
      fam_n = PT.get_node_from_label(bc, 'FamilyName_t')
      if fam_n is None:
        if bc[0].startswith("BCDegene"):
          degen_bc_n = bc
      else:
        fam = PT.get_value(PT.get_node_from_label(bc, 'FamilyName_t'))
        if fam == 'AXIS':
          degen_bc_n = bc
    pl_degen_bc = PT.get_value(PT.get_node_from_name(degen_bc_n, 'PointList'))[0]
    ngon_n  = PT.Zone.NGonNode(shallow_zone_n)
    # # degen_bc_vtx = maia.algo.dist.vertex_list.face_ids_to_vtx_ids(pl_degen_bc, ngon_n, comm)
    # # print(degen_bc_vtx)
    # print(nodes_degen_bc)
    # print([comm.rank, f"SB : pl_degen_bc = {pl_degen_bc}"])
    
    zsr_n = PT.get_node_from_name(shallow_zone_n, 'ZSR_DegenLine')
    # PT.print_tree(zsr_n)
    if zsr_n is None:
      p_nodes_degen_bc = np.zeros((0), dtype=np.int32)
    else:
      p_nodes_degen_bc = PT.get_value(PT.get_node_from_name(zsr_n, 'PointList'))[0]
    # print([comm.rank, f"SB : p_nodes_degen_bc = {p_nodes_degen_bc}"])
    # print([comm.rank, f"SB : nodes_degen_bc = {nodes_degen_bc}"])
    # exit()
    old_to_new_degen_bc_vtx = PT.get_value(PT.get_node_from_name(zsr_n, 'OldToNew'))
    distrib_old_to_new_degen_bc_vtx = PT.get_value(PT.maia.getDistribution(zsr_n, 'Index')).copy()
    # print(old_to_new_degen_bc_vtx)
    
    ref_vtx         = np.unique(old_to_new_degen_bc_vtx)
    # print([comm.rank, f"SB : ref_vtx = {ref_vtx}"])
    # print([comm.rank, f"SB before: nodes_degen_bc : {len(nodes_degen_bc)} => {nodes_degen_bc}"])
    # size_nodes_degen_bc_per_proc = comm.allgather(len(nodes_degen_bc))
    # # print([comm.rank, f"SB : size_nodes_degen_bc_per_proc : {size_nodes_degen_bc_per_proc}"])
    # distrib_beg = int(np.sum(size_nodes_degen_bc_per_proc[:comm.rank]))
    # distrib_end = int(np.sum(size_nodes_degen_bc_per_proc[:comm.rank+1]))
    # distrib_tot = int(np.sum(size_nodes_degen_bc_per_proc[:comm.size]))
    # distrib_nodes_degen_bc = np.asarray([distrib_beg, distrib_end, distrib_tot])
    # # print([comm.rank, f"SB : distrib_old_to_new_degen_bc_vtx : {distrib_old_to_new_degen_bc_vtx}"])
    # # print([comm.rank, f"SB : distrib_nodes_degen_bc : {distrib_nodes_degen_bc}"])
    # nodes_degen_bc = EP.block_to_block(nodes_degen_bc, distrib_nodes_degen_bc, distrib_old_to_new_degen_bc_vtx, comm)
    # print([comm.rank, f"SB after: nodes_degen_bc : {len(nodes_degen_bc)} => {nodes_degen_bc}"])
    # exit()
    # # index_to_remove = np.concatenate([np.where(ref_vtx[i] == nodes_degen_bc) for i in range(len(ref_vtx))])
    # index_to_remove = np.concatenate([np.where(nodes_degen_bc == ref_vtx[i]) for i in range(len(ref_vtx))])
    # vtx_to_remove   = np.delete(nodes_degen_bc, index_to_remove)
    in_or_not = maia.utils.parallel.algo.gnum_isin(nodes_degen_bc,ref_vtx, comm)
    index_to_remove = np.where(in_or_not == True)
    # print([comm.rank, f"SB : index_to_remove = {index_to_remove}"])
    vtx_to_remove   = np.delete(nodes_degen_bc, index_to_remove)
    # print([comm.rank, f"SB : vtx_to_remove = {vtx_to_remove}"])
    # exit()
    # print(len(ref_vtx))
    # print(len(vtx_to_remove))
    # print("-------")
    
    face_to_remove = pl_degen_bc
    # print(face_to_remove)
    # print([comm.rank, f"SB : face_to_remove = {face_to_remove}"])
    n_rmvd_face    = comm.allreduce(len(face_to_remove), op=MPI.SUM)
    # print(n_rmvd_face)
    # print([comm.rank, f"SB : n_rmvd_face = {n_rmvd_face}"])
    # exit()
    
    vtx_distri_ini  = PT.get_value(PT.maia.getDistribution(shallow_zone_n, 'Vertex'))
    face_distri_ini = PT.get_value(PT.maia.getDistribution(ngon_n, 'Element')).copy()
    
    # print([comm.rank, f"SB : len(nodes_degen_bc) = {len(nodes_degen_bc)}"])
    # print([comm.rank, f"SB : len(old_to_new_degen_bc_vtx) = {len(old_to_new_degen_bc_vtx)}"])
    old_to_new_degen_bc_vtx_remove = np.delete(old_to_new_degen_bc_vtx, index_to_remove)
    # print([comm.rank, f"SB : old_to_new_degen_bc_vtx_remove = {old_to_new_degen_bc_vtx_remove}"])
    old_to_new_vtx  = merge_distributed_ids(vtx_distri_ini, vtx_to_remove, old_to_new_degen_bc_vtx_remove, comm)
    # print(old_to_new_vtx)
    # print([old_to_new_vtx[i] for i in range(len(old_to_new_vtx))])
    # exit()
  
    # full_tree = maia.factory.dist_to_full_tree(dist_tree, comm)
    # if comm.rank==0:
    #   maia.io.write_tree(full_tree, f'tmp_pre_{comm.size}p.cgns')
      
    
    MDB._new_update_ngon(ngon_n, face_to_remove, vtx_distri_ini, old_to_new_vtx, comm)
    # maia.io.dist_tree_to_file(dist_tree_ngon, 'temp_ngon.cgns', comm)
    # maia.io.dist_tree_to_file(dist_tree, 'temp.cgns', comm)
    # full_tree = maia.factory.dist_to_full_tree(dist_tree_ngon, comm)
    # if comm.rank==0:
    #   maia.io.write_tree(full_tree, f'tmp_ngon_{comm.size}p.cgns')
    # exit()
    
    # Because some faces are removed we trick the distribution by creating a new
    # face with number nb_faces + 1
    nb_faces = face_distri_ini[2]
    face_distri_ext = copy.deepcopy(face_distri_ini)
    if face_distri_ext[1] == nb_faces:
      face_distri_ext[1] += 1
    face_distri_ext[2] += 1
    # print(nb_faces, face_distri_ext[2])
    
    old_to_new_face_to_remove = (nb_faces+1)*np.ones(len(face_to_remove), dtype=np.int32)
    # print(face_to_remove)
    # print(old_to_new_face_to_remove)
    old_to_new_face = merge_distributed_ids(face_distri_ext, face_to_remove, old_to_new_face_to_remove, comm)
    # print(old_to_new_face)
    # exit()
      
    # # exit()
    nface_n = PT.Zone.NFaceNode(shallow_zone_n)
    if nface_n:
      # TO DO : mettre a jour la fonction maia.algo.dist._update_nface ci-dessous :
      MDB._new_update_nface(nface_n, face_distri_ext, old_to_new_face, n_rmvd_face, comm)
      # # For now : delete nface node
      # PT.rm_node_from_path(shallow_zone_n, nface_n[0])
    # full_tree = maia.factory.dist_to_full_tree(dist_tree_ngon, comm)
    # if comm.rank==0:
    #   maia.io.write_tree(full_tree, f'tmp_nface_{comm.size}p.cgns')
    # exit()
    
    # maia.algo.dist.merge_jn._update_vtx_data(shallow_zone_n, vtx_to_remove, comm)
    MDB._new_update_vtx_data(shallow_zone_n, vtx_to_remove, comm)
    
    
    base_name = PT.get_name(PT.get_node_from_label(dist_tree_ngon, "CGNSBase_t"))
    # print(base_name)
    
    MDB._new_update_cgns_subsets(shallow_zone_n, 'Vertex', vtx_distri_ini, old_to_new_vtx, base_name, comm)
    
    #Shift all CellCenter PL by the number of removed faces
    if PT.Element.Range(ngon_n)[0] == 1:
      MDB._new_shift_cgns_subsets(shallow_zone_n, 'CellCenter', -n_rmvd_face)
    
    old_to_new_face_unsg = np.abs(old_to_new_face)
    MDB._new_update_cgns_subsets(shallow_zone_n, 'FaceCenter', face_distri_ext, old_to_new_face_unsg, base_name, comm)
    
    # TO DO: delete in all FaceCenter* PL all "nb_faces+1" numbered face
    # for now: del degen_bc
    PT.rm_node_from_path(shallow_zone_n, f'ZoneBC/{degen_bc_n[0]}')
    
    # Suppression des artefacts d'algo
    PT.rm_nodes_from_label(shallow_zone_n,'ZoneSubRegion_t')
  
    # if (shallow_zone_n[0] == 'blk-15') or (shallow_zone_n[0] == 'sillage'):
    #   shallow_ngon_n = PT.Zone.NGonNode(shallow_zone_n)
    #   shallow_ngon_ec  = PT.get_value(PT.get_node_from_name(shallow_ngon_n, 'ElementConnectivity'))
    #   shallow_ngon_eso = PT.get_value(PT.get_node_from_name(shallow_ngon_n, 'ElementStartOffset'))
    #   shallow_coords = PT.get_node_from_label(shallow_zone_n, 'GridCoordinates_t')
    #   shallow_coord_x = PT.get_value(PT.get_node_from_name(shallow_coords, 'CoordinateX'))
    #   shallow_bc_farfield_n = PT.get_node_from_name(shallow_zone_n, 'BCFar*')
    #   if (shallow_zone_n[0] == 'blk-15'):
    #     shallow_coord_x[14700] = shallow_coord_x[14638]
    #   if (shallow_zone_n[0] == 'sillage'):
    #     shallow_coord_x[44] = shallow_coord_x[59505]
    #     # for face in PT.get_value(PT.get_node_from_name(shallow_bc_farfield_n, 'PointList'))[0]:
    #     #   nodes = shallow_ngon_ec[shallow_ngon_eso[face-1]:shallow_ngon_eso[face]]
    #     #   print(face, nodes, shallow_coord_x[nodes-1])
    
    # shallow_zone_n[0] = shallow_zone_n[0]+'_SB'
    # print('SB1 : ', zone_n[0], shallow_zone_n[0])
    # zone_n = shallow_zone_n
    # print('SB2 : ', zone_n[0], shallow_zone_n[0])
    # maia.io.dist_tree_to_file(dist_tree, 'temp.cgns', comm)
    # maia.io.dist_tree_to_file(dist_tree_ngon, 'temp_ngon.cgns', comm)
    # dist_tree_elem = copy.deepcopy(dist_tree_ngon)
    # maia.algo.dist.ngons_to_elements(dist_tree_elem, comm)
    # maia.io.dist_tree_to_file(dist_tree_elem, 'temp_elem.cgns', comm)
    PT.add_child(new_base_n, shallow_zone_n)
    
  
  PT.rm_nodes_from_label(new_dist_tree,'ZoneSubRegion_t')
  PT.rm_nodes_from_name(new_dist_tree, 'BCDegene*')
  
  # full_tree = maia.factory.dist_to_full_tree(new_dist_tree, comm)
  # if comm.rank==0:
  #   maia.io.write_tree(full_tree, f'test_{comm.size}p.cgns')
  
  maia.io.dist_tree_to_file(new_dist_tree, f'/stck/sbouras/dev/dev-Fun/DegeneratedLine/eighth_cylinder_5x5x5_ready_ngon_without_degenline_{comm.size}p_CI.cgns', comm)
  
  # dist_tree_elem = copy.deepcopy(new_dist_tree)
  # maia.algo.dist.ngons_to_elements(dist_tree_elem, comm)
  
  # # PT.print_tree(dist_tree_elem)
  # maia.io.dist_tree_to_file(dist_tree_elem, f'{name}_elem_without_degenline{extension}', comm)
  
  # # new_new_tree = CP.convertFile2PyTree('out_ngon.cgns')
  # # old_new_tree = convert_CGNS_fromNew2Old(new_new_tree)
  # # CP.convertPyTree2File(old_new_tree, 'AXImesh_noDGelements.cgns')

