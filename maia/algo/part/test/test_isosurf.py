import pytest
import pytest_parallel
import numpy as np

import maia
import maia.pytree        as PT

from maia.algo.part import isosurf as ISO

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype
dtype = 'I4' if pdm_gnum_dtype == np.int32 else 'I8'


def test_copy_referenced_families():
  source_base = PT.yaml.to_node(
  """
  Base CGNSBase_t:
    Toto Family_t:
    Tata Family_t:
    Titi Family_t:
  """)
  target_base = PT.yaml.to_node(
  """
  Base CGNSBase_t:
    Tyty Family_t: #Already in target tree
    ZoneA Zone_t:
      FamilyName FamilyName_t "Toto":
      AddFamilyName AdditionalFamilyName_t "Tutu": #Not in source tree
    ZoneB Zone_t:
      AdditionalFamilyName AdditionalFamilyName_t "Titi":
  """)
  ISO.copy_referenced_families(source_base, target_base)
  assert PT.get_child_from_name(target_base, 'Tyty') is not None
  assert PT.get_child_from_name(target_base, 'Toto') is not None
  assert PT.get_child_from_name(target_base, 'Titi') is not None
  assert PT.get_child_from_name(target_base, 'Tata') is None


@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("from_api", [False, True])
def test_exchange_field_one_domain(from_api, comm):
  if comm.Get_rank() == 0:
    yt_vol = f"""
    VolZone.P0.N0 Zone_t:
      NGonElements Elements_t [22,0]:
        ElementRange IndexRange_t [1,8]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Element DataArray_t {dtype} [1,3,5,7]:
      ZoneBC ZoneBC_t:
        Zmin BC_t:
          GridLocation GridLocation_t "FaceCenter":
          PointList    IndexArray_t {dtype} [[1,2,3,4]]:
      FSolVtx FlowSolution_t:
        GridLocation GridLocation_t "Vertex":
        fieldC DataArray_t [60., 40, 20, 50, 30, 10]:
      FSolCell FlowSolution_t:
        GridLocation GridLocation_t "CellCenter":
        fieldA DataArray_t [40., 30., 20., 10.]:
        fieldB DataArray_t [400., 300., 200., 100.]:
      FSolBC ZoneSubRegion_t:
        BCRegionName Descriptor_t "Zmin":
        fieldD DataArray_t R8 [-1., -3., -5., -7.]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Cell DataArray_t {dtype} [4,3,2,1]:
        Vertex DataArray_t {dtype} [6,4,2,5,3,1]:
    """
    yt_surf = f"""
    VolZone.P0.N0 Zone_t:
      BAR_2 Elements_t [3,0]:
        ElementRange IndexRange_t [1,3]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Element DataArray_t {dtype} [3,2]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Cell DataArray_t {dtype} [2]:
        Vertex DataArray_t {dtype} [1,2]:
      maia#surface_data UserDefinedData_t:
        Vtx_parent_weight DataArray_t [1., 1.]:
        Vtx_parent_gnum DataArray_t {dtype} [6,5]:
        Vtx_parent_idx DataArray_t I4 [0,1,2]:
        Cell_parent_gnum DataArray_t {dtype} [4]:
        Face_parent_bnd_edges DataArray_t {dtype} [5, 1]:
    """
  else:
    yt_surf = f"""
    VolZone.P1.N0 Zone_t:
      BAR_2 Elements_t [3,0]:
        ElementRange IndexRange_t [1,3]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Element DataArray_t {dtype} [1]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Cell DataArray_t {dtype} [1,3]:
        Vertex DataArray_t {dtype} [2,3]:
      maia#surface_data UserDefinedData_t:
        Vtx_parent_weight DataArray_t [1., .5, .5]:
        Vtx_parent_gnum DataArray_t {dtype} [5,1,2]:
        Vtx_parent_idx DataArray_t I4 [0,1,3]:
        Cell_parent_gnum DataArray_t {dtype} [3, 1]:
        Face_parent_bnd_edges DataArray_t {dtype} [3]:
    """
    yt_vol = f"""
    VolZone.P1.N0 Zone_t:
      NGonElements Elements_t [22,0]:
        ElementRange IndexRange_t [1,8]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Element DataArray_t {dtype} [2,4,6,8]:
      FSolVtx FlowSolution_t:
        GridLocation GridLocation_t "Vertex":
        fieldC DataArray_t [70., 80]:
      FSolCell FlowSolution_t:
        GridLocation GridLocation_t "CellCenter":
        fieldA DataArray_t [50.]:
        fieldB DataArray_t [500.]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Cell DataArray_t {dtype} [5]:
        Vertex DataArray_t {dtype} [7,8]:
    """

  if comm.Get_rank() == 0:
    expected_A = np.array([40.])
    expected_B = np.array([400.])
    expected_C = np.array([60., 50.])
    expected_D = np.array([-5., -1.])
  else:
    expected_A = np.array([30., 10.])
    expected_B = np.array([300., 100.])
    expected_C = np.array([50., 15.])
    expected_D = np.array([-3.])

  if from_api:
    iso_tree  = PT.yaml.to_cgns_tree(yt_surf)
    vol_tree  = PT.yaml.to_cgns_tree(yt_vol)
    ISO._exchange_field(vol_tree, iso_tree, ["FSolCell", "FSolVtx", "FSolBC"], comm)
    iso_zone = PT.get_all_Zone_t(iso_tree)[0]
  else:
    iso_zone  = PT.yaml.to_node(yt_surf)
    vol_zones = PT.yaml.to_nodes(yt_vol)
    ISO.exchange_field_one_domain(vol_zones, iso_zone, ["FSolCell", "FSolVtx", "FSolBC"], comm)

  assert PT.Subset.GridLocation(PT.get_node_from_name(iso_zone, "FSolCell")) == "CellCenter"
  assert PT.Subset.GridLocation(PT.get_node_from_name(iso_zone, "FSolVtx")) == "Vertex"
  assert np.array_equal(PT.get_node_from_path(iso_zone, "FSolCell/fieldA")[1], expected_A)
  assert np.array_equal(PT.get_node_from_path(iso_zone, "FSolCell/fieldB")[1], expected_B)
  assert np.array_equal(PT.get_node_from_path(iso_zone, "FSolVtx/fieldC")[1], expected_C)
  assert np.array_equal(PT.get_node_from_path(iso_zone, "FSolBC/fieldD")[1], expected_D)
  

@pytest.mark.skipif(not maia.pdma_enabled, reason="Require ParaDiGMA")
@pytest_parallel.mark.parallel(2)
def test_isosurf_one_domain(comm):
  dist_tree = maia.factory.generate_dist_block(3, "Poly", comm)
  print(dist_tree)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
  #print(part_tree)
  part_zones = PT.get_all_Zone_t(part_tree)
  iso_zone = ISO.iso_surface_one_domain(part_zones, "PLANE", [1,0,0,0.25], "TRI_3", "hilbert", comm)

  assert PT.Zone.n_cell(iso_zone) == 16 and PT.Zone.n_vtx(iso_zone) == 15
  assert (PT.get_node_from_name(iso_zone, 'CoordinateX')[1] == 0.25).all()
  assert (PT.get_child_from_predicates(iso_zone, 'TRI_3/ElementRange')[1] == np.array([ 1, 16], dtype=np.int32)).all()
  assert (PT.get_child_from_predicates(iso_zone, 'BAR_2/ElementRange')[1] == np.array([17, 24], dtype=np.int32)).all()

  assert PT.get_label(PT.get_child_from_name(iso_zone, "maia#surface_data")) == 'UserDefinedData_t'

# @pytest_parallel.mark.parallel(2)
# def test_surface_from_equation(comm):
#   elt_type        = ["elt_type", "TRI_3"] 
#   graph_part_tool = ["graph_part_tool", "ptscotch"] 
#   dist_tree = maia.factory.generate_dist_block(3, "TRI_3", comm)
#   print(dist_tree)
#   part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
#   ellipse_eq = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]  # (x^2 + y^2 + z^2 = 1)
#   iso_part_tree =  ISO._surface_from_equation(part_tree, 'ELLIPSE', ellipse_eq, elt_type, graph_part_tool, comm) 
#   assert iso_part_tree is not None
  #assert PT.get_label(iso_part_tree) == 'CGNSTree'

  # Vérification de la structure de l'arbre résultant
  # iso_bases = PT.get_nodes_from_label(iso_part_tree, 'CGNSBase_t')
  # assert len(iso_bases) == 1
  # iso_base = iso_bases[0]
  # assert PT.get_name(iso_base) == 'Base'
  # iso_zones = PT.get_nodes_from_label(iso_base, 'Zone_t')
  # assert len(iso_zones) == 1
  # iso_zone = iso_zones[0]
  # assert PT.get_name(iso_zone).startswith('Zone_part')
  # # Vérification que la zone contient des cellules
  # assert PT.Zone.n_cell(iso_zone) > 0
  # # Vérification des conteneurs de champs transférés
  # iso_fs_nodes = PT.get_nodes_from_label(iso_zone, 'FlowSolution_t')
  # assert len(iso_fs_nodes) == 1
  # iso_field_nodes = PT.get_nodes_from_label(iso_fs_nodes[0], 'DataArray_t')
  # assert len(iso_field_nodes) > 0

# # Fonction pour simuler l'extraction de surface à partir d'une équation
# def _surface_from_equation(part_tree, eq_type, eq_params, elt_type, graph_part_tool, comm):
#     # Simuler la création d'une zone iso
#     iso_zone = PT.new_Zone('Zone_part', [[3, 2, 0]], 'Unstructured')
#     fs_node = PT.new_FlowSolution('FlowSolution', loc='Vertex', parent=iso_zone)
#     field_node = PT.new_DataArray('Field', np.array([1.0, 2.0, 3.0])), parent=fs_node)
#     iso_tree = PT.new_CGNSTree()
#     base = PT.new_CGNSBase('Base', 3, 3, parent=iso_tree)
#     PT.add_child(base, iso_zone)
#     return iso_tree

# # Fonction pour simuler l'échange de champs
# def _exchange_field(src_tree, tgt_tree, containers_name, comm):
#     # Simuler le transfert des champs
#     src_zones = PT.get_nodes_from_label(src_tree, 'Zone_t')
#     tgt_zones = PT.get_nodes_from_label(tgt_tree, 'Zone_t')
#     for src_zone, tgt_zone in zip(src_zones, tgt_zones):
#         for name in containers_name:
#             src_fs_node = PT.get_child_from_name(src_zone, name)
#             tgt_fs_node = PT.get_child_from_name(tgt_zone, name)
#             if src_fs_node is not None and tgt_fs_node is not None:
#                 for src_field_node in PT.get_children_from_label(src_fs_node, 'DataArray_t'):
#                     tgt_field_node = PT.copy_tree(src_field_node)
#                     PT.add_child(tgt_fs_node, tgt_field_node)

# # Remplacement des fonctions externes pour le test
# PT.get_parts_per_blocks = get_parts_per_blocks
# PT.maia.conv.add_part_suffix = lambda name, rank, _: f'{name}_part{rank}'
#   #print(part_tree)
  

  
  