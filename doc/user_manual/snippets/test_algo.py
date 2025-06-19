import pytest
import shutil

feflo_exists = shutil.which('feflo.a') is not None

def test_convert_s_to_u():
  #convert_s_to_u@start
  from mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir
  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'S_twoblocks.yaml', MPI.COMM_WORLD)

  maia.algo.dist.convert_s_to_u(dist_tree, 'Poly', MPI.COMM_WORLD)
  for zone in maia.pytree.get_all_Zone_t(dist_tree):
    assert maia.pytree.Zone.Type(zone) == "Unstructured"
  #convert_s_to_u@end

def test_transform_affine():
  #transform_affine@start
  from mpi4py import MPI
  import maia
  dist_tree = maia.factory.generate_dist_block(10, 'Poly', MPI.COMM_WORLD)
  zone = maia.pytree.get_all_Zone_t(dist_tree)[0]

  maia.algo.transform_affine(zone, translation=[3,0,0])
  #transform_affine@end

def test_scale_mesh():
  #scale_mesh@start
  from mpi4py import MPI
  import maia
  dist_tree = maia.factory.generate_dist_block(10, 'Poly', MPI.COMM_WORLD)

  assert maia.pytree.get_node_from_name(dist_tree, 'CoordinateX')[1].max() <= 1.
  maia.algo.scale_mesh(dist_tree, [3.0, 2.0, 1.0])
  assert maia.pytree.get_node_from_name(dist_tree, 'CoordinateX')[1].max() <= 3.
  #scale_mesh@end

def test_generate_jns_vertex_list():
  #generate_jns_vertex_list@start
  from mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir
  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'S_twoblocks.yaml', MPI.COMM_WORLD)
  maia.algo.dist.convert_s_to_ngon(dist_tree, MPI.COMM_WORLD)

  maia.algo.dist.generate_jns_vertex_list(dist_tree, MPI.COMM_WORLD)
  assert len(maia.pytree.get_nodes_from_name(dist_tree, 'match*#Vtx')) == 2
  #generate_jns_vertex_list@end

def test_remove_degen_faces():
  #remove_degen_faces_from_family@start
  from mpi4py import MPI
  import maia
  from   maia.utils.test_utils import sample_mesh_dir
  dist_tree = maia.io.file_to_dist_tree(sample_mesh_dir/'degen_faces.yaml', MPI.COMM_WORLD)
  dist_zone = maia.pytree.get_node_from_label(dist_tree, 'Zone_t') #Only one zone

  assert maia.pytree.Zone.n_face(dist_zone) == 240
  maia.algo.dist.remove_degen_faces_from_family(dist_tree, 'DEGEN_AXIS', MPI.COMM_WORLD)
  assert maia.pytree.Zone.n_face(dist_zone) == 240 - 16 # 16 faces in degenerated family
  #remove_degen_faces_from_family@end

def test_duplicate_from_rotation_jns_to_360():
  #duplicate_from_rotation_to_360@start
  from mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir
  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'U_ATB_45.yaml', MPI.COMM_WORLD)
  left_jns  = ['Base/bump_45/ZoneGridConnectivity/matchA']
  right_jns = ['Base/bump_45/ZoneGridConnectivity/matchB']
  maia.algo.dist.duplicate_from_rotation_jns_to_360(dist_tree,
                                                    ['Base/bump_45'],
                                                    (left_jns, right_jns),
                                                    MPI.COMM_WORLD)
  assert len(maia.pytree.get_all_Zone_t(dist_tree)) == 45
  #duplicate_from_rotation_to_360@end

def test_duplicate_family_from_periodic_jns():
  #duplicate_family_from_periodic_jns@start
  from mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir
  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'U_ATB_45.yaml', MPI.COMM_WORLD)
  maia.algo.dist.duplicate_family_from_periodic_jns(dist_tree,
                                                    'ATB',
                                                    17,
                                                    MPI.COMM_WORLD)
  assert len(maia.pytree.get_all_Zone_t(dist_tree)) == 18
  #duplicate_family_from_periodic_jns@end

def test_extrude_2d():
  #extrude@start
  from mpi4py import MPI
  import maia
  import maia.pytree as PT

  dist_tree = maia.factory.generate_dist_block(11, 'TRI_3', MPI.COMM_WORLD)
  maia.algo.dist.extrude(dist_tree, [0,0,0.5], MPI.COMM_WORLD)

  assert PT.Zone.CellDimension(PT.get_node_from_label(dist_tree, 'Zone_t')) == 3
  #extrude@end

def test_merge_zones():
  #merge_zones@start
  from mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir
  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'U_Naca0012_multizone.yaml', MPI.COMM_WORLD)
  assert len(maia.pytree.get_all_Zone_t(dist_tree)) == 3

  maia.algo.dist.merge_zones(dist_tree, ["BaseA/blk1", "BaseB/blk2"], MPI.COMM_WORLD)
  assert len(maia.pytree.get_all_Zone_t(dist_tree)) == 2
  #merge_zones@end

def test_merge_zones_from_family():
  #merge_zones_from_family@start
  from mpi4py import MPI
  import maia
  import maia.pytree as PT
  from   maia.utils.test_utils import mesh_dir
  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'U_Naca0012_multizone.yaml', MPI.COMM_WORLD)

  # FamilyName are not included in the mesh
  for zone in PT.get_all_Zone_t(dist_tree):
    PT.new_child(zone, 'FamilyName', 'FamilyName_t', 'Naca0012')

  maia.algo.dist.merge_zones_from_family(dist_tree, 'Naca0012', MPI.COMM_WORLD)

  zones = PT.get_all_Zone_t(dist_tree)
  assert len(zones) == 1 and PT.get_name(zones[0]) == 'naca0012'
  #merge_zones_from_family@end

def test_merge_connected_zones():
  #merge_connected_zones@start
  from mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir
  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'U_Naca0012_multizone.yaml', MPI.COMM_WORLD)

  maia.algo.dist.merge_connected_zones(dist_tree, MPI.COMM_WORLD)
  assert len(maia.pytree.get_all_Zone_t(dist_tree)) == 1
  #merge_connected_zones@end

def test_compute_elements_center():
  #compute_elements_center@start
  from mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir
  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'U_ATB_45.yaml', MPI.COMM_WORLD)

  maia.algo.compute_elements_center(dist_tree, 3, MPI.COMM_WORLD)
  assert maia.pytree.get_node_from_name(dist_tree, 'Geometry_3d') is not None
  #compute_elements_center@end

def test_compute_elements_measure():
  #compute_elements_measure@start
  from mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir
  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'Uelt_M6Wing.yaml', MPI.COMM_WORLD)

  maia.algo.compute_elements_measure(dist_tree, 3, MPI.COMM_WORLD)
  assert maia.pytree.get_node_from_name(dist_tree, 'Geometry_3d') is not None
  #compute_elements_measure@end

def test_compute_elements_normal():
  #compute_elements_normal@start
  from mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir
  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'Uelt_M6Wing.yaml', MPI.COMM_WORLD)

  maia.algo.compute_elements_normal(dist_tree, MPI.COMM_WORLD)
  assert maia.pytree.get_node_from_name(dist_tree, 'Geometry_2d') is not None
  #compute_elements_normal@end

def test_compute_wall_distance():
  #compute_wall_distance@start
  from mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir
  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'U_ATB_45.yaml', MPI.COMM_WORLD)
  part_tree = maia.factory.partition_dist_tree(dist_tree, MPI.COMM_WORLD)

  maia.algo.part.compute_wall_distance(part_tree, MPI.COMM_WORLD)
  assert maia.pytree.get_node_from_name(part_tree, "WallDistance") is not None
  #compute_wall_distance@end

def test_compute_iso_surface():
  #compute_iso_surface@start
  from mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir
  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'U_ATB_45.yaml', MPI.COMM_WORLD)
  part_tree = maia.factory.partition_dist_tree(dist_tree, MPI.COMM_WORLD, preserve_orientation=True)
  maia.algo.part.compute_wall_distance(part_tree, MPI.COMM_WORLD, point_cloud='Vertex')

  part_tree_iso = maia.algo.part.iso_surface(part_tree, "WallDistance/TurbulentDistance", iso_val=0.25,\
      containers_name=['WallDistance'], comm=MPI.COMM_WORLD)

  assert maia.pytree.get_node_from_name(part_tree_iso, "WallDistance") is not None
  #compute_iso_surface@end

def test_compute_plane_slice():
  #compute_plane_slice@start
  from mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir

  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'U_Naca0012_multizone.yaml', MPI.COMM_WORLD)
  maia.algo.dist.merge_connected_zones(dist_tree, MPI.COMM_WORLD) # Isosurf requires single block mesh
  part_tree = maia.factory.partition_dist_tree(dist_tree, MPI.COMM_WORLD, preserve_orientation=True)

  slice_tree = maia.algo.part.plane_slice(part_tree, [0,0,1,0.5], MPI.COMM_WORLD, elt_type='QUAD_4')
  #compute_plane_slice@end

def test_compute_spherical_slice():
  #compute_spherical_slice@start
  from mpi4py import MPI
  import numpy
  import maia
  import maia.pytree as PT
  dist_tree = maia.factory.generate_dist_block(11, 'Poly', MPI.COMM_WORLD)
  part_tree = maia.factory.partition_dist_tree(dist_tree, MPI.COMM_WORLD, preserve_orientation=True)

  # Add solution
  zone      = PT.get_node_from_label(part_tree, "Zone_t")
  vol_rank  = MPI.COMM_WORLD.Get_rank() * numpy.ones(PT.Zone.n_cell(zone))
  src_sol   = PT.new_FlowSolution('FlowSolution', loc='CellCenter', fields={'i_rank' : vol_rank}, parent=zone)

  slice_tree = maia.algo.part.spherical_slice(part_tree, [0.5,0.5,0.5,0.25], MPI.COMM_WORLD, \
      ["FlowSolution"], elt_type="NGON_n")

  assert maia.pytree.get_node_from_name(slice_tree, "FlowSolution") is not None
  #compute_spherical_slice@end

def test_extract_from_zsr():
  #extract_from_zsr@start
  from   mpi4py import MPI
  import numpy as np
  import maia
  import maia.pytree as PT
  from   maia.utils.test_utils import mesh_dir

  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'U_ATB_45.yaml', MPI.COMM_WORLD)
  part_tree = maia.factory.partition_dist_tree(dist_tree, MPI.COMM_WORLD)

  maia.algo.part.compute_wall_distance(part_tree, MPI.COMM_WORLD, point_cloud='Vertex')

  # Create a ZoneSubRegion on procs for extracting odd cells
  for part_zone in PT.get_all_Zone_t(part_tree):
    ncell       = PT.Zone.n_cell(part_zone)
    start_range = PT.Element.Range(PT.Zone.NFaceNode(part_zone))[0]
    point_list  = np.arange(start_range, start_range+ncell, 2, dtype=np.int32).reshape((1,-1), order='F')
    PT.new_ZoneSubRegion(name='ZoneSubRegion', point_list=point_list, loc='CellCenter', parent=part_zone)

  extracted_tree = maia.algo.part.extract_part_from_zsr(part_tree, 'ZoneSubRegion', MPI.COMM_WORLD,
                                                        containers_name=["WallDistance"])

  assert maia.pytree.get_node_from_name(extracted_tree, "WallDistance") is not None
  #extract_from_zsr@end

def test_extract_from_bc_name():
  #extract_from_bc_name@start
  from   mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir

  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'U_ATB_45.yaml', MPI.COMM_WORLD)
  part_tree = maia.factory.partition_dist_tree(dist_tree, MPI.COMM_WORLD)

  maia.algo.part.compute_wall_distance(part_tree, MPI.COMM_WORLD, point_cloud='Vertex')

  extracted_bc = maia.algo.part.extract_part_from_bc_name(part_tree, \
                 'wall', MPI.COMM_WORLD, containers_name=["WallDistance"])

  assert maia.pytree.get_node_from_name(extracted_bc, "WallDistance") is not None
  #extract_from_bc_name@end

def test_extract_from_family():
  #extract_from_family@start
  from   mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir

  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'U_ATB_45.yaml', MPI.COMM_WORLD)
  part_tree = maia.factory.partition_dist_tree(dist_tree, MPI.COMM_WORLD)

  maia.algo.part.compute_wall_distance(part_tree, MPI.COMM_WORLD, point_cloud='Vertex')

  extracted_bc = maia.algo.part.extract_part_from_family(part_tree, \
                 'WALL', MPI.COMM_WORLD, containers_name=["WallDistance"])

  assert maia.pytree.get_node_from_name(extracted_bc, "WallDistance") is not None
  #extract_from_family@end

def test_compute_elliptical_slice():
  #compute_elliptical_slice@start
  from mpi4py import MPI
  import maia
  import maia.pytree as PT
  from   maia.algo.part import isosurf
  dist_tree = maia.factory.generate_dist_block(11, 'Poly', MPI.COMM_WORLD)
  part_tree = maia.factory.partition_dist_tree(dist_tree, MPI.COMM_WORLD, preserve_orientation=True)

  slice_tree = isosurf.elliptical_slice(part_tree, [0.5,0.5,0.5,.5,1.,1.,.25**2], \
      MPI.COMM_WORLD, elt_type='NGON_n')
  #compute_elliptical_slice@end

def test_localize_points():
  #localize_points@start
  import mpi4py
  import maia
  import maia.pytree as PT
  from   maia.utils.test_utils import mesh_dir
  comm = mpi4py.MPI.COMM_WORLD

  tree_src = maia.io.file_to_dist_tree(mesh_dir/'U_ATB_45.yaml', comm)
  tree_tgt = maia.io.file_to_dist_tree(mesh_dir/'U_ATB_45.yaml', comm)
  for tgt_zone in maia.pytree.get_all_Zone_t(tree_tgt):
    maia.algo.transform_affine(tgt_zone, rotation_angle=[170*3.14/180,0,0], translation=[0,0,3])

  maia.algo.localize_points(tree_src, tree_tgt, 'CellCenter', comm)
  for tgt_zone in maia.pytree.get_all_Zone_t(tree_tgt):
    loc_container = PT.get_child_from_name(tgt_zone, 'Localization')
    assert PT.Subset.GridLocation(loc_container) == 'CellCenter'
  #localize_points@end

def test_find_closest_points():
  #find_closest_points@start
  import mpi4py
  import maia
  import maia.pytree as PT
  from   maia.utils.test_utils import mesh_dir
  comm = mpi4py.MPI.COMM_WORLD

  dist_tree_src = maia.io.file_to_dist_tree(mesh_dir/'U_ATB_45.yaml', comm)
  dist_tree_tgt = maia.io.file_to_dist_tree(mesh_dir/'U_ATB_45.yaml', comm)
  for tgt_zone in maia.pytree.get_all_Zone_t(dist_tree_tgt):
    maia.algo.transform_affine(tgt_zone, rotation_angle=[170*3.14/180,0,0], translation=[0,0,3])
  part_tree_src = maia.factory.partition_dist_tree(dist_tree_src, comm)
  part_tree_tgt = maia.factory.partition_dist_tree(dist_tree_tgt, comm)

  maia.algo.find_closest_points(part_tree_src, part_tree_tgt, 'Vertex', comm)
  for tgt_zone in maia.pytree.get_all_Zone_t(part_tree_tgt):
    loc_container = PT.get_child_from_name(tgt_zone, 'ClosestPoint')
    assert PT.Subset.GridLocation(loc_container) == 'Vertex'
  #find_closest_points@end

def test_interpolate():
  #interpolate@start
  import mpi4py
  import numpy
  import maia
  import maia.pytree as PT
  comm = mpi4py.MPI.COMM_WORLD

  dist_tree_src = maia.factory.generate_dist_block(11, 'Poly', comm)
  dist_tree_tgt = maia.factory.generate_dist_block(20, 'Poly', comm)
  part_tree_src = maia.factory.partition_dist_tree(dist_tree_src, comm)
  part_tree_tgt = maia.factory.partition_dist_tree(dist_tree_tgt, comm)
  # Create fake solution
  zone = maia.pytree.get_node_from_label(part_tree_src, "Zone_t")
  src_sol = maia.pytree.new_FlowSolution('FlowSolution', loc='CellCenter', parent=zone)
  PT.new_DataArray("Field", numpy.random.rand(PT.Zone.n_cell(zone)), parent=src_sol)

  maia.algo.interpolate(part_tree_src, part_tree_tgt, comm,\
      ['FlowSolution'], 'Vertex')
  tgt_sol = PT.get_node_from_name(part_tree_tgt, 'FlowSolution')
  assert tgt_sol is not None and PT.Subset.GridLocation(tgt_sol) == 'Vertex'
  #interpolate@end

def test_centers_to_nodes():
  #centers_to_nodes@start
  import mpi4py
  import maia
  import maia.pytree as PT
  from   maia.utils.test_utils import mesh_dir
  comm = mpi4py.MPI.COMM_WORLD

  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'U_Naca0012_multizone.yaml', comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

  # Init fields located at Cells (output container is Geometry_3d)
  maia.algo.compute_elements_center(part_tree, 3)

  maia.algo.part.centers_to_nodes(part_tree, comm, ['Geometry_3d'])

  for part in PT.get_all_Zone_t(part_tree):
    vtx_sol = PT.get_node_from_name(part, 'Geometry_3d#Vtx')
    assert PT.Subset.GridLocation(vtx_sol) == 'Vertex'
  #centers_to_nodes@end

def test_nodes_to_centers():
  #nodes_to_centers@start
  import mpi4py
  import maia
  import maia.pytree as PT
  comm = mpi4py.MPI.COMM_WORLD

  dist_tree = maia.factory.generate_dist_sphere(3, 'TETRA_4', comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

  # Init a FlowSolution located at Nodes
  for part in PT.get_all_Zone_t(part_tree):
    cx, cy, cz = PT.Zone.coordinates(part)
    fields = {'cX': cx, 'cY': cy, 'cZ': cz}
    PT.new_FlowSolution('FSol', loc='Vertex', fields=fields, parent=part)

  maia.algo.part.nodes_to_centers(part_tree, comm, ['FSol'])

  for part in PT.get_all_Zone_t(part_tree):
    cell_sol = PT.get_node_from_name(part, 'FSol#Cell')
    assert PT.Subset.GridLocation(cell_sol) == 'CellCenter'
  #nodes_to_centers@end

def test_pe_to_nface():
  #pe_to_nface@start
  from mpi4py import MPI
  import maia
  tree = maia.factory.generate_dist_block(6, 'Poly', MPI.COMM_WORLD)

  for zone in maia.pytree.get_all_Zone_t(tree):
    maia.algo.pe_to_nface(zone, MPI.COMM_WORLD)
    assert maia.pytree.get_child_from_name(zone, 'NFaceElements') is not None
  #pe_to_nface@end

def test_nface_to_pe():
  #nface_to_pe@start
  from mpi4py import MPI
  import maia
  tree = maia.factory.generate_dist_block(6, 'NFace_n', MPI.COMM_WORLD)

  maia.algo.nface_to_pe(tree, MPI.COMM_WORLD)
  assert maia.pytree.get_node_from_name(tree, 'ParentElements') is not None
  #nface_to_pe@end

def test_edge_pe_to_ngon():
  #edge_pe_to_ngon@start
  from mpi4py import MPI
  import maia
  tree = maia.factory.generate_dist_sphere(5, 'NGON_n', MPI.COMM_WORLD)

  maia.pytree.rm_nodes_from_name(tree, 'NGonElements')
  maia.algo.edge_pe_to_ngon(tree, MPI.COMM_WORLD)
  assert maia.pytree.get_node_from_name(tree, 'NGonElements') is not None
  #edge_pe_to_ngon@end

def test_ngon_to_edge_pe():
  #ngon_to_edge_pe@start
  from mpi4py import MPI
  import maia
  tree = maia.factory.generate_dist_sphere(5, 'NGON_n', MPI.COMM_WORLD)

  maia.pytree.rm_nodes_from_name(tree, 'ParentElements')
  maia.algo.ngon_to_edge_pe(tree, MPI.COMM_WORLD)
  assert maia.pytree.get_node_from_name(tree, 'ParentElements') is not None
  #ngon_to_edge_pe@end

def test_poly_new_to_old():
  #poly_new_to_old@start
  import maia
  from   maia.utils.test_utils import mesh_dir

  tree = maia.io.read_tree(mesh_dir/'U_ATB_45.yaml')
  assert maia.pytree.get_node_from_name(tree, 'ElementStartOffset') is not None

  maia.algo.seq.poly_new_to_old(tree)
  assert maia.pytree.get_node_from_name(tree, 'ElementStartOffset') is None
  #poly_new_to_old@end

def test_poly_old_to_new():
  #poly_old_to_new@start
  import maia
  from   maia.utils.test_utils import mesh_dir

  tree = maia.io.read_tree(mesh_dir/'U_ATB_45.yaml')
  maia.algo.seq.poly_new_to_old(tree)
  assert maia.pytree.get_node_from_name(tree, 'ElementStartOffset') is None

  maia.algo.seq.poly_old_to_new(tree)
  assert maia.pytree.get_node_from_name(tree, 'ElementStartOffset') is not None
  #poly_old_to_new@end

def test_enforce_ngon_pe_local():
  #enforce_ngon_pe_local@start
  import maia
  from   maia.utils.test_utils import mesh_dir

  import maia.pytree as PT

  tree = maia.io.read_tree(mesh_dir/'U_ATB_45.yaml')
  zone = PT.get_node_from_label(tree, 'Zone_t')
  n_cell = PT.Zone.n_cell(zone)

  assert PT.get_node_from_name(zone, 'ParentElements')[1].max() > n_cell
  maia.algo.seq.enforce_ngon_pe_local(tree)
  assert PT.get_node_from_name(zone, 'ParentElements')[1].max() <= n_cell
  #enforce_ngon_pe_local@end

def test_elements_to_ngons():
  #elements_to_ngons@start
  from mpi4py import MPI
  import maia
  from maia.utils.test_utils import mesh_dir

  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'Uelt_M6Wing.yaml', MPI.COMM_WORLD)
  maia.algo.dist.convert_elements_to_ngon(dist_tree, MPI.COMM_WORLD, stable_sort=True)
  #elements_to_ngons@end

def test_convert_ngon_to_elements():
  #convert_ngon_to_elements@start
  from mpi4py import MPI
  import maia
  import maia.pytree as PT

  dist_tree = maia.factory.generate_dist_block(11, 'Poly', MPI.COMM_WORLD)
  maia.algo.dist.convert_ngon_to_elements(dist_tree, MPI.COMM_WORLD)

  elts = PT.get_nodes_from_label(dist_tree, 'Elements_t')
  assert [PT.Element.CGNSName(e) for e in elts] == ['QUAD_4', 'HEXA_8']
  #convert_ngon_to_elements@end

def test_convert_elements_to_ngon():
  #convert_elements_to_ngon@start
  from mpi4py import MPI
  import maia
  from maia.utils.test_utils import mesh_dir

  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'Uelt_M6Wing.yaml', MPI.COMM_WORLD)
  maia.algo.dist.convert_elements_to_ngon(dist_tree, MPI.COMM_WORLD)
  #convert_elements_to_ngon@end

def test_convert_elements_to_mixed():
  #convert_elements_to_mixed@start
  from mpi4py import MPI
  import maia
  from maia.utils.test_utils import mesh_dir

  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'Uelt_M6Wing.yaml', MPI.COMM_WORLD)
  maia.algo.dist.convert_elements_to_mixed(dist_tree, MPI.COMM_WORLD)
  #convert_elements_to_mixed@end

def test_convert_mixed_to_elements():
  #convert_mixed_to_elements@start
  from mpi4py import MPI
  import maia
  from maia.utils.test_utils import mesh_dir

  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'Uelt_M6Wing.yaml', MPI.COMM_WORLD)
  maia.algo.dist.convert_elements_to_mixed(dist_tree, MPI.COMM_WORLD)
  maia.algo.dist.convert_mixed_to_elements(dist_tree, MPI.COMM_WORLD)
  #convert_mixed_to_elements@end

def test_reorder_elt_sections_from_dim():
  #reorder_elt_sections_from_dim@start
  from mpi4py import MPI
  import maia
  import maia.pytree as PT

  dist_tree = maia.factory.generate_dist_block(11, 'PYRA_5', MPI.COMM_WORLD)
  for zone in PT.get_all_Zone_t(dist_tree):
    assert PT.Zone.elt_ordering_by_dim(zone) != 1 # Elts are not increasing by dim

  maia.algo.dist.reorder_elt_sections_from_dim(dist_tree)
  for zone in PT.get_all_Zone_t(dist_tree):
    assert PT.Zone.elt_ordering_by_dim(zone) == 1 # Now, yes
  #reorder_elt_sections_from_dim@end

def test_concatenate_elt_sections():
  #concatenate_elt_sections@start
  from mpi4py import MPI
  import maia
  import maia.pytree as PT
  from maia.utils.test_utils import mesh_dir

  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'H_elt_and_s.yaml', MPI.COMM_WORLD)

  is_quad_elt = lambda n : PT.get_label(n) == 'Elements_t' and \
                           PT.Element.CGNSName(n) == 'QUAD_4'

  assert len(PT.get_nodes_from_predicate(dist_tree, is_quad_elt)) > 1 # Several QUAD sections
  maia.algo.dist.concatenate_elt_sections(dist_tree, MPI.COMM_WORLD)
  assert len(PT.get_nodes_from_predicate(dist_tree, is_quad_elt)) == 1 # Now, only one
  #concatenate_elt_sections@end

def test_recover1to1():
  #recover1to1@start
  from mpi4py import MPI
  from numpy  import array, pi
  import maia
  import maia.pytree as PT
  from maia.utils.test_utils import mesh_dir

  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'U_ATB_45.yaml', MPI.COMM_WORLD)

  # Remove data that should be created
  PT.rm_nodes_from_name(dist_tree, 'PointListDonor')
  PT.rm_nodes_from_name(dist_tree, 'GridConnectivityProperty')

  # Create FamilyName on interface nodes
  PT.new_node('FamilyName', 'FamilyName_t', 'Side1',
          parent=PT.get_node_from_name(dist_tree, 'matchA'))
  PT.new_node('FamilyName', 'FamilyName_t', 'Side2',
          parent=PT.get_node_from_name(dist_tree, 'matchB'))

  maia.algo.dist.connect_1to1_families(dist_tree, ('Side1', 'Side2'), MPI.COMM_WORLD,
          periodic={'rotation_angle' : array([-2*pi/45.,0.,0.])})

  assert len(PT.get_nodes_from_name(dist_tree, 'PointListDonor')) == 2
  #recover1to1@end

def test_redistribute_dist_tree():
  #redistribute_dist_tree@start
  from mpi4py import MPI
  import maia

  dist_tree = maia.factory.generate_dist_block(21, 'Poly', MPI.COMM_WORLD)
  maia.algo.dist.redistribute_tree(dist_tree, 'gather.0', MPI.COMM_WORLD)
  #redistribute_dist_tree@end

@pytest.mark.skipif(not feflo_exists, reason="Require Feflo.a")
def test_adapt_with_feflo():
  #adapt_with_feflo@start
  import mpi4py.MPI as MPI
  import maia
  import maia.pytree as PT

  from maia.algo.dist import adapt_mesh_with_feflo

  dist_tree = maia.factory.generate_dist_block(5, 'TETRA_4', MPI.COMM_WORLD)
  zone = PT.get_node_from_label(dist_tree, 'Zone_t')

  # > Create a metric field
  cx, cy, cz = PT.Zone.coordinates(zone)
  fields= {'metric' : (cx-0.5)**5+(cy-0.5)**5 - 1}
  PT.new_FlowSolution("FlowSolution", loc="Vertex", fields=fields, parent=zone)

  # > Adapt mesh according to scalar metric
  adpt_dist_tree = adapt_mesh_with_feflo(dist_tree,
                                         "FlowSolution/metric",
                                         MPI.COMM_WORLD,
                                         container_names=["FlowSolution"],
                                         feflo_opts="-c 100 -cmax 100 -p 4")
  #adapt_with_feflo@end


def test_change_basis():
  #change_basis@start
  import mpi4py.MPI as MPI
  import numpy      as np
  import maia

  dist_tree = maia.factory.generate_dist_block(5, 'S', MPI.COMM_WORLD)
  part_tree = maia.factory.partition_dist_tree(dist_tree, MPI.COMM_WORLD)

  maia.algo.transform.auxiliary_coords_system(part_tree, np.array([[0,1,0],[-1,0,0],[0,0,-1]]))

  assert maia.pytree.get_node_from_name(part_tree, 'CoordinateZeta') is not None
  #change_basis@end

def test_cartesian_to_cylindrical():
  #cartesian_to_cylindrical@start
  import mpi4py.MPI as MPI
  import maia
  from   maia.utils.test_utils import mesh_dir

  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'U_ATB_45.yaml', MPI.COMM_WORLD)
  maia.algo.cartesian_to_cylindrical(dist_tree, (1,0,0), MPI.COMM_WORLD)

  assert maia.pytree.get_node_from_name(dist_tree, 'CoordinateR') is not None
  #cartesian_to_cylindrical@end

def test_cylindrical_to_cartesian():
  #cylindrical_to_cartesian@start
  import mpi4py.MPI as MPI
  import maia
  import maia.pytree as PT
  from   maia.utils.test_utils import mesh_dir

  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'U_ATB_45.yaml', MPI.COMM_WORLD)
  maia.algo.cartesian_to_cylindrical(dist_tree, (1,0,0), MPI.COMM_WORLD)

  # Create a vector field on cylindrical mesh
  for zone in PT.get_nodes_from_label(dist_tree, 'Zone_t'):
    cr, ctheta, cz = PT.Zone.coordinates(zone)
    fields= {'VelocityR': cr**2, 'VelocityTheta': ctheta, 'VelocityZ': 0*cz}
    PT.new_FlowSolution("FlowSolution", loc="Vertex", fields=fields, parent=zone)

  maia.algo.cylindrical_to_cartesian(dist_tree, (1,0,0), MPI.COMM_WORLD)

  assert PT.get_node_from_name(dist_tree, 'VelocityX') is not None
  #cylindrical_to_cartesian@end

def test_find_ridges():
  #retrieve_ridges@start
  import mpi4py.MPI as MPI
  import maia
  import maia.pytree as PT
  from   maia.utils.test_utils import mesh_dir

  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'U_ATB_45.yaml', MPI.COMM_WORLD)

  maia.algo.dist.find_ridges(dist_tree, [['wall'], 'AMONT'], MPI.COMM_WORLD)

  assert PT.get_node_from_path(dist_tree, 'Base/bump_45/topo_edge') is not None
  #retrieve_ridges@end

def test_extract_edges():
  #extract_edges@start
  import mpi4py.MPI as MPI
  import maia
  import maia.pytree as PT
  from   maia.utils.test_utils import mesh_dir
  import numpy

  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'axisym_mesh.yaml', MPI.COMM_WORLD)

  point_list = [PT.Subset.getPatch(n)[1][0] \
    for n in PT.get_nodes_from_predicate(dist_tree, PT.pred.is_bc_of_location('EdgeCenter'))]
  domain_pl = {'cube/zone': numpy.concatenate(point_list)}

  edge_dist_tree = maia.algo.dist.extract_part.extract_edges(dist_tree, domain_pl, MPI.COMM_WORLD)

  #extract_edges@end

def test_concat_from_fam():
  #concat_from_fam@start
  import mpi4py.MPI as MPI
  import maia
  import maia.pytree as PT
  from   maia.utils.test_utils import mesh_dir

  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'axisym_mesh.yaml', MPI.COMM_WORLD)

  maia.algo.dist.concatenate_subsets_from_families(dist_tree, MPI.COMM_WORLD, families=['RIDGE'])

  is_ridge_bc = PT.pred.label_is('BC_t') & PT.pred.belongs_to_family('RIDGE')
  assert len(PT.get_nodes_from_predicate(dist_tree, is_ridge_bc)) == 1
  #concat_from_fam@end

def test_deconcatenate_from_families():
  #deconcatenate_from_fam@start
  import mpi4py.MPI as MPI
  import maia
  import maia.pytree as PT
  from   maia.utils.test_utils import mesh_dir

  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'axisym_mesh.yaml', MPI.COMM_WORLD)

  maia.algo.dist.concatenate_subsets_from_families(dist_tree, MPI.COMM_WORLD, families=['RIDGE'])
  maia.algo.dist.deconcatenate_subsets_from_families(dist_tree, MPI.COMM_WORLD, families=['RIDGE'])

  is_ridge_bc = PT.pred.label_is('BC_t') & PT.pred.belongs_to_family('RIDGE')
  assert len(PT.get_nodes_from_predicate(dist_tree, is_ridge_bc)) == 9
  #deconcatenate_from_fam@end
