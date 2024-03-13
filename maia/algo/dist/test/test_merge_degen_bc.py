import copy
import itertools
import mpi4py.MPI      as MPI
import numpy           as np

import pytest
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



@pytest_parallel.mark.parallel([1,2,7,11,23,59])
# @pytest.mark.parametrize("ZSR", [False, True])
@pytest.mark.parametrize("ZSR", [False])
def test_merge_degen_bc(ZSR,comm):
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
  vtx_distri = PT.get_value(PT.maia.getDistribution(zone_n, 'Vertex'))
  
  coords_n = PT.get_node_from_label(dist_tree, 'GridCoordinates_t')
  coord_x_n, coord_y_n, coord_z_n = [PT.get_node_from_name(coords_n, f"Coordinate{suffix}") for suffix in ['X', 'Y', 'Z']]
  coord_x,   coord_y,   coord_z   = [PT.get_value(coord) for coord in [coord_x_n, coord_y_n, coord_z_n]]
  
  multiple, remainder = np.divmod(vtx_distri, nx*ny)
  assert remainder[2] == 0
  
  start = max(multiple[0], 1) # max because the first plane does not move
  stop  = multiple[1] if remainder[1] == 0 else multiple[1] + 1
  
  for i in np.arange(start,stop):
    beg = max(i*nx*ny, vtx_distri[0]) - vtx_distri[0]
    end = min((i+1)*nx*ny, vtx_distri[1]) - vtx_distri[0]
    coord_z[beg:end] = 0.
    coord_x[beg:end], coord_y[beg:end], coord_z[beg:end] = maia.utils.ndarray.np_utils.transform_cart_vectors(coord_x[beg:end], coord_y[beg:end], coord_z[beg:end], rotation_angle = np.array([i*theta_x,theta_y,theta_z]))
  
  #----------------------------
  # Add families
  base_n = PT.get_all_CGNSBase_t(dist_tree)[0] 
  for bc_n in PT.get_nodes_from_predicates(dist_tree, 'CGNSBase_t/Zone_t/ZoneBC_t/BC_t'):
      PT.set_value(bc_n, 'FamilySpecified')
      PT.update_child(bc_n,'FamilyName',label='FamilyName_t',value=bc2fam[bc_n[0]])
  
  for fam in fam_l:
      PT.update_child(base_n,fam,label='Family_t')
  
  if ZSR:
    ymin_n = PT.get_node_from_predicates(dist_tree, 'CGNSBase_t/Zone_t/ZoneBC_t/Ymin')
    PT.rm_nodes_from_name(ymin_n, 'FamilyName')
    PT.print_tree(ymin_n)
    zone_n = PT.get_node_from_predicates(dist_tree, 'CGNSBase_t/Zone_t')
    PT.new_ZoneSubRegion(name='ZSR_Ymin', bc_name='Ymin', family=bc2fam['Ymin'], parent=zone_n)
  
  #----------------------------
  # Convert to ngon
  # When convert_elements_to_ngon will be parallel independant, this part
  # could be reduce to
  # maia.algo.dist.convert_elements_to_ngon(dist_tree, comm)
  maia.algo.dist.redistribute_tree(dist_tree, 'gather', comm)
  group = comm.Get_group()
  newGroup = group.Incl([0])
  sub_comm  = comm.Create(newGroup)
  if comm.rank == 0:
    maia.algo.dist.convert_elements_to_ngon(dist_tree, sub_comm)
    full_tree = maia.factory.dist_to_full_tree(dist_tree, sub_comm, target=0)
  else:
    full_tree = None
  dist_tree = maia.factory.full_to_dist_tree(full_tree, comm, owner=0)
  # maia.io.dist_tree_to_file(dist_tree, f'/stck/sbouras/dev/dev-Fun/DegeneratedLine/eighth_cylinder_5x5x5_ready_CI_{comm.size}p.cgns', comm)
  
  #-------------
  # Begin of the test
  
  fam_to_remove        = 'AXIS'
  fam_for_intersection = 'PER2'
  
  new_dist_tree = MDB.delete_degen_faces_from_family(dist_tree, fam_to_remove, fam_for_intersection, comm)
  
  if ZSR:
    PT.rm_nodes_from_name(new_dist_tree, 'ZSR_Ymin')
    PT.rm_nodes_from_name(new_dist_tree, 'Ymin')
  
  maia.algo.dist.redistribute_tree(new_dist_tree, 'uniform', comm)
  ref_dist_tree = maia.io.file_to_dist_tree(f'/stck/sbouras/dev/dev-Fun/DegeneratedLine/eighth_cylinder_5x5x5_ready_ngon_without_degenline_REF.cgns', comm)
  assert maia.pytree.is_same_tree(ref_dist_tree, new_dist_tree)


  # maia.io.dist_tree_to_file(new_dist_tree, f'/stck/sbouras/dev/dev-Fun/DegeneratedLine/eighth_cylinder_5x5x5_ready_ngon_without_degenline_{comm.size}p_CI.cgns', comm)
