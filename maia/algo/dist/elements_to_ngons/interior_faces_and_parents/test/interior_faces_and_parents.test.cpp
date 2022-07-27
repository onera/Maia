#if __cplusplus > 201703L
#include "std_e/unit_test/doctest_pybind_mpi.hpp"

#include "maia/io/file.hpp"
#include "maia/utils/mesh_dir.hpp"
#include "cpp_cgns/tree_manip.hpp"
#include "maia/pytree/maia/element_sections.hpp"

#include "maia/algo/dist/elements_to_ngons/interior_faces_and_parents/interior_faces_and_parents.hpp"
#include "std_e/parallel/mpi/base.hpp"
#include "std_e/multi_array/utils.hpp"


using namespace cgns;


PYBIND_MPI_TEST_CASE("generate_interior_faces_and_parents - seq",1) {
  std::string file_name = maia::mesh_dir+"hex_2_prism_2.yaml";
  tree t = maia::file_to_dist_tree(file_name,MPI_COMM_SELF);
  tree& z = cgns::get_node_by_matching(t,"Base/Zone");

  auto old_hexa_first_id = cgns::get_node_value_by_matching<I4>(z,"Hexas/ElementRange")[0];

  maia::generate_interior_faces_and_parents(z,test_comm);

  auto new_hexa_first_id = cgns::get_node_value_by_matching<I4>(z,"Hexas/ElementRange")[0];
  auto n_in_faces = new_hexa_first_id-old_hexa_first_id;
  CHECK( n_in_faces == 4 );

  // tri ext
  auto pe_tri_ext     = cgns::get_node_value_by_matching<I4,2>(z,"Tris/ParentElements");
  auto pp_tri_ext = cgns::get_node_value_by_matching<I4,2>(z,"Tris/ParentElementsPosition");
  CHECK( pe_tri_ext.extent() == std_e::multi_index<I8,2>{2,2} );
  CHECK( pe_tri_ext == cgns::md_array<I4,2>{{21,0},{22,0}} );
  CHECK( pp_tri_ext == cgns::md_array<I4,2>{{ 4,0},{ 5,0}} );

  // tri in
  auto elt_type_tri_in = cgns::get_node_value_by_matching<I4>(z,"TRI_3_interior")[0];
  auto range_tri_in    = cgns::get_node_value_by_matching<I4>(z,"TRI_3_interior/ElementRange");
  auto connec_tri_in   = cgns::get_node_value_by_matching<I4>(z,"TRI_3_interior/ElementConnectivity");
  auto pe_tri_in       = cgns::get_node_value_by_matching<I4,2>(z,"TRI_3_interior/ParentElements");
  auto pp_tri_in       = cgns::get_node_value_by_matching<I4,2>(z,"TRI_3_interior/ParentElementsPosition");

  CHECK( elt_type_tri_in == (I4)cgns::TRI_3 );
  CHECK( range_tri_in == std::vector<I4>{15,15} );
  CHECK( connec_tri_in == std::vector<I4>{7,8,10} );
  CHECK( pe_tri_in.extent() == std_e::multi_index<I8,2>{1,2} );
  CHECK( pe_tri_in == cgns::md_array<I4,2>{{21,22}} );
  CHECK( pp_tri_in == cgns::md_array<I4,2>{{ 5, 4}} );

  // quad ext
  auto pe_quad_ext = cgns::get_node_value_by_matching<I4,2>(z,"Quads/ParentElements");
  auto pp_quad_ext = cgns::get_node_value_by_matching<I4,2>(z,"Quads/ParentElementsPosition");
  CHECK( pe_quad_ext.extent() == std_e::multi_index<I8,2>{12,2} );
  CHECK( pe_quad_ext == cgns::md_array<I4,2>{{19,0},{20,0},{21,0},{22,0},{19,0},{21,0},{20,0},{22,0},{19,0},{20,0},{19,0},{20,0}} );
  CHECK( pp_quad_ext == cgns::md_array<I4,2>{{ 5,0},{ 5,0},{ 2,0},{ 2,0},{ 2,0},{ 1,0},{ 2,0},{ 1,0},{ 4,0},{ 4,0},{ 1,0},{ 6,0}} );

  // quad in
  auto elt_type_quad_in = cgns::get_node_value_by_matching<I4>(z,"QUAD_4_interior")[0];
  auto range_quad_in    = cgns::get_node_value_by_matching<I4>(z,"QUAD_4_interior/ElementRange");
  auto connec_quad_in   = cgns::get_node_value_by_matching<I4>(z,"QUAD_4_interior/ElementConnectivity");
  auto pe_quad_in       = cgns::get_node_value_by_matching<I4,2>(z,"QUAD_4_interior/ParentElements");
  auto pp_quad_in       = cgns::get_node_value_by_matching<I4,2>(z,"QUAD_4_interior/ParentElementsPosition");

  CHECK( elt_type_quad_in == (I4)cgns::QUAD_4 );
  CHECK( range_quad_in == std::vector<I4>{16,18} );
  CHECK( connec_quad_in == std::vector<I4>{2,5,10,7, 6,7,10,9, 7,10,15,12} );
  CHECK( pe_quad_in.extent() == std_e::multi_index<I8,2>{3,2} );
  CHECK( pe_quad_in == cgns::md_array<I4,2>{{19,21},{19,20},{20,22}} );
  CHECK( pp_quad_in == cgns::md_array<I4,2>{{ 3, 3},{ 6, 1},{ 3, 3}} );

  // hex
  auto hex_cell_face = cgns::get_node_value_by_matching<I4>(z,"Hexas/CellFace");
  CHECK( hex_cell_face == std::vector<I4>{11,5,16,9,1,17,  -17,7,18,10,2,12} );
  // prisms
  auto prism_cell_face = cgns::get_node_value_by_matching<I4>(z,"Prisms/CellFace");
  CHECK( prism_cell_face == std::vector<I4>{6,3,-16,13,15,  8,4,-18,-15,14} );
}


PYBIND_MPI_TEST_CASE("generate_interior_faces_and_parents",2) {
  std::string file_name = maia::mesh_dir+"hex_2_prism_2.yaml";
  tree t = maia::file_to_dist_tree(file_name,test_comm);
  tree& z = cgns::get_node_by_matching(t,"Base/Zone");

  maia::generate_interior_faces_and_parents(z,test_comm);
  // tri ext
  auto pe_tri_ext     = cgns::get_node_value_by_matching<I4,2>(z,"Tris/ParentElements");
  auto pp_tri_ext = cgns::get_node_value_by_matching<I4,2>(z,"Tris/ParentElementsPosition");
  CHECK( pe_tri_ext.extent() == std_e::multi_index<I8,2>{1,2} );
  MPI_CHECK(0, pe_tri_ext == cgns::md_array<I4,2>{{21,0}} );
  MPI_CHECK(1, pe_tri_ext == cgns::md_array<I4,2>{{22,0}} );
  MPI_CHECK(0, pp_tri_ext == cgns::md_array<I4,2>{{ 4,0}       } );
  MPI_CHECK(1, pp_tri_ext == cgns::md_array<I4,2>{       { 5,0}} );

  // tri in
  auto elt_type_tri_in = cgns::get_node_value_by_matching<I4>(z,"TRI_3_interior")[0];
  auto range_tri_in    = cgns::get_node_value_by_matching<I4>(z,"TRI_3_interior/ElementRange");
  auto connec_tri_in   = cgns::get_node_value_by_matching<I4>(z,"TRI_3_interior/ElementConnectivity");
  auto pe_tri_in       = cgns::get_node_value_by_matching<I4,2>(z,"TRI_3_interior/ParentElements");
  auto pp_tri_in       = cgns::get_node_value_by_matching<I4,2>(z,"TRI_3_interior/ParentElementsPosition");

  CHECK( elt_type_tri_in == (I4)cgns::TRI_3 );
  CHECK( range_tri_in == std::vector<I4>{15,15} );
  MPI_CHECK(0, connec_tri_in == std::vector<I4>{} );
  MPI_CHECK(1, connec_tri_in == std::vector<I4>{7,8,10} );
  MPI_CHECK(0, pe_tri_in.extent() == std_e::multi_index<I8,2>{0,2} );
  MPI_CHECK(1, pe_tri_in.extent() == std_e::multi_index<I8,2>{1,2} );
  MPI_CHECK(0, pe_tri_in == cgns::md_array<I4,2>(0,2) );
  MPI_CHECK(1, pe_tri_in == cgns::md_array<I4,2>{{21,22}} );
  MPI_CHECK(0, pp_tri_in == cgns::md_array<I4,2>(0,2) );
  MPI_CHECK(1, pp_tri_in == cgns::md_array<I4,2>{{ 5, 4}} );

  // quad ext
  auto pe_quad_ext = cgns::get_node_value_by_matching<I4,2>(z,"Quads/ParentElements");
  auto pp_quad_ext = cgns::get_node_value_by_matching<I4,2>(z,"Quads/ParentElementsPosition");
  CHECK( pe_quad_ext.extent() == std_e::multi_index<I8,2>{6,2} );
  MPI_CHECK(0, pe_quad_ext == cgns::md_array<I4,2>{{19,0},{20,0},{21,0},{22,0},{19,0},{21,0}                                          } );
  MPI_CHECK(1, pe_quad_ext == cgns::md_array<I4,2>{                                          {20,0},{22,0},{19,0},{20,0},{19,0},{20,0}} );
  MPI_CHECK(0, pp_quad_ext == cgns::md_array<I4,2>{{ 5,0},{ 5,0},{ 2,0},{ 2,0},{ 2,0},{ 1,0}                                          } );
  MPI_CHECK(1, pp_quad_ext == cgns::md_array<I4,2>{                                          { 2,0},{ 1,0},{ 4,0},{ 4,0},{ 1,0},{ 6,0}} );

  // quad in
  auto elt_type_quad_in = cgns::get_node_value_by_matching<I4>(z,"QUAD_4_interior")[0];
  auto range_quad_in    = cgns::get_node_value_by_matching<I4>(z,"QUAD_4_interior/ElementRange");
  auto connec_quad_in   = cgns::get_node_value_by_matching<I4>(z,"QUAD_4_interior/ElementConnectivity");
  auto pe_quad_in       = cgns::get_node_value_by_matching<I4,2>(z,"QUAD_4_interior/ParentElements");
  auto pp_quad_in       = cgns::get_node_value_by_matching<I4,2>(z,"QUAD_4_interior/ParentElementsPosition");

  CHECK( elt_type_quad_in == (I4)cgns::QUAD_4 );
  CHECK( range_quad_in == std::vector<I4>{16,18} );
  MPI_CHECK(0, connec_quad_in == std::vector<I4>{2,5,10,7} );
  MPI_CHECK(1, connec_quad_in == std::vector<I4>{6,7,10,9, 7,10,15,12} );
  MPI_CHECK(0, pe_quad_in.extent() == std_e::multi_index<I8,2>{1,2} );
  MPI_CHECK(1, pe_quad_in.extent() == std_e::multi_index<I8,2>{2,2} );
  MPI_CHECK(0, pe_quad_in == cgns::md_array<I4,2>{{19,21}} );
  MPI_CHECK(1, pe_quad_in == cgns::md_array<I4,2>{{19,20},{20,22}} );
  MPI_CHECK(0, pp_quad_in == cgns::md_array<I4,2>{{ 3, 3}} );
  MPI_CHECK(1, pp_quad_in == cgns::md_array<I4,2>{{ 6, 1},{ 3, 3}} );

  // hex
  auto hex_cell_face = cgns::get_node_value_by_matching<I4>(z,"Hexas/CellFace");
  MPI_CHECK(0, hex_cell_face == std::vector<I4>{11,5,16,9,1,17                 } );
  MPI_CHECK(1, hex_cell_face == std::vector<I4>{                -17,7,18,10,2,12} );
  // prisms
  auto prism_cell_face = cgns::get_node_value_by_matching<I4>(z,"Prisms/CellFace");
  MPI_CHECK(0, prism_cell_face == std::vector<I4>{6,3,-16,13,15              } );
  MPI_CHECK(1, prism_cell_face == std::vector<I4>{              8,4,-18,-15,14} );
}
#endif // C++>17
