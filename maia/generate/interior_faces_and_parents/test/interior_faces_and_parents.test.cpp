#include "std_e/unit_test/doctest_pybind_mpi.hpp"

#include "maia/utils/yaml/parse_yaml_cgns.hpp"
#include "maia/utils/mesh_dir.hpp"
#include "std_e/utils/file.hpp"
#include "cpp_cgns/tree_manip.hpp"
#include "maia/sids/element_sections.hpp"

#include "maia/generate/interior_faces_and_parents/interior_faces_and_parents.hpp"
#include "std_e/parallel/mpi/base.hpp"


using namespace cgns;


PYBIND_MPI_TEST_CASE("generate_interior_faces_and_parents - seq",1) {
  std::string yaml_tree = std_e::file_to_string(maia::mesh_dir+"hex_2_prism_2_dist_01.yaml");
  tree b = maia::to_node(yaml_tree);

  tree& z = cgns::get_node_by_name(b,"Zone");

  SUBCASE("generate") {
    auto faces_and_parents = maia::gen_interior_faces_and_parents(maia::element_sections(z));
    CHECK(faces_and_parents.tris .connectivities().range() == std::vector{2,5,3, 12,13,15, 2,5,3, 7,8,10, 7,10,8, 12,13,15} );
    CHECK(faces_and_parents.tris .parents() == std::vector{13,14,17,17,18,18});
  }
  SUBCASE("final") {
    maia::generate_interior_faces_and_parents(z,test_comm);
    // tri ext
    auto parent_tri_ext = cgns::get_node_value_by_matching<I4,2>(b,"Zone/Tris/ParentElements");
    CHECK( parent_tri_ext.extent() == std_e::multi_index<I8,2>{2,2} );
    CHECK( parent_tri_ext == cgns::md_array<I4,2>{{17,0},{18,0}} );

    // tri in
    auto elt_type_tri_in = cgns::get_node_value_by_matching<I4>(b,"Zone/TRI_3_interior")[0];
    auto range_tri_in    = cgns::get_node_value_by_matching<I4>(b,"Zone/TRI_3_interior/ElementRange");
    auto connec_tri_in   = cgns::get_node_value_by_matching<I4>(b,"Zone/TRI_3_interior/ElementConnectivity");
    auto parent_tri_in   = cgns::get_node_value_by_matching<I4,2>(b,"Zone/TRI_3_interior/ParentElements");

    CHECK( elt_type_tri_in == (I4)cgns::TRI_3 );
    CHECK( range_tri_in == std::vector<I4>{19,19} );
    CHECK( connec_tri_in == std::vector<I4>{7,8,10} );
    CHECK( parent_tri_in.extent() == std_e::multi_index<I8,2>{1,2} );
    CHECK( parent_tri_in == cgns::md_array<I4,2>{{17,18}} );

    // quad ext
    auto parent_quad_ext = cgns::get_node_value_by_matching<I4,2>(b,"Zone/Quads/ParentElements");
    CHECK( parent_quad_ext.extent() == std_e::multi_index<I8,2>{12,2} );
    CHECK( parent_quad_ext == cgns::md_array<I4,2>{{15,0},{16,0},{17,0},{18,0},{15,0},{17,0},{16,0},{18,0},{15,0},{16,0},{15,0},{16,0}} );

    // quad in
    auto elt_type_quad_in = cgns::get_node_value_by_matching<I4>(b,"Zone/QUAD_4_interior")[0];
    auto range_quad_in    = cgns::get_node_value_by_matching<I4>(b,"Zone/QUAD_4_interior/ElementRange");
    auto connec_quad_in   = cgns::get_node_value_by_matching<I4>(b,"Zone/QUAD_4_interior/ElementConnectivity");
    auto parent_quad_in   = cgns::get_node_value_by_matching<I4,2>(b,"Zone/QUAD_4_interior/ParentElements");

    CHECK( elt_type_quad_in == (I4)cgns::QUAD_4 );
    CHECK( range_quad_in == std::vector<I4>{20,22} );
    CHECK( connec_quad_in == std::vector<I4>{2,7,10,5, 6,7,10,9, 7,10,15,12} );
    CHECK( parent_quad_in.extent() == std_e::multi_index<I8,2>{3,2} );
    CHECK( parent_quad_in == cgns::md_array<I4,2>{{17,15},{15,16},{16,18}} );
  }
}


PYBIND_MPI_TEST_CASE("generate_interior_faces_and_parents",2) {
  int rk = std_e::rank(test_comm);
  std::string yaml_tree = std_e::file_to_string(maia::mesh_dir+"hex_2_prism_2_dist_"+std::to_string(rk)+".yaml");
  tree b = maia::to_node(yaml_tree);

  tree& z = cgns::get_node_by_name(b,"Zone");

  SUBCASE("generate") {
    auto faces_and_parents = maia::gen_interior_faces_and_parents(maia::element_sections(z));
    MPI_CHECK(0, faces_and_parents.tris .connectivities().range() == std::vector{2,5,3, 2,5,3, 7,8,10} );
    MPI_CHECK(0, faces_and_parents.tris .parents() == std::vector{13,17,17});
    MPI_CHECK(1, faces_and_parents.tris .connectivities().range() == std::vector{12,13,15, 7,10,8, 12,13,15} );
    MPI_CHECK(1, faces_and_parents.tris .parents() == std::vector{14,18,18});
  }
  SUBCASE("final") {
    maia::generate_interior_faces_and_parents(z,test_comm);
    // tri ext
    auto parent_tri_ext = cgns::get_node_value_by_matching<I4,2>(b,"Zone/Tris/ParentElements");
    CHECK( parent_tri_ext.extent() == std_e::multi_index<I8,2>{1,2} );
    MPI_CHECK(0, parent_tri_ext == cgns::md_array<I4,2>{{17,0}} );
    MPI_CHECK(1, parent_tri_ext == cgns::md_array<I4,2>{{18,0}} );

    // tri in
    auto elt_type_tri_in = cgns::get_node_value_by_matching<I4>(b,"Zone/TRI_3_interior")[0];
    auto range_tri_in    = cgns::get_node_value_by_matching<I4>(b,"Zone/TRI_3_interior/ElementRange");
    auto connec_tri_in   = cgns::get_node_value_by_matching<I4>(b,"Zone/TRI_3_interior/ElementConnectivity");
    auto parent_tri_in   = cgns::get_node_value_by_matching<I4,2>(b,"Zone/TRI_3_interior/ParentElements");

    CHECK( elt_type_tri_in == (I4)cgns::TRI_3 );
    CHECK( range_tri_in == std::vector<I4>{19,19} );
    MPI_CHECK(0, connec_tri_in == std::vector<I4>{7,8,10} );
    MPI_CHECK(1, connec_tri_in == std::vector<I4>{} );
    MPI_CHECK(0, parent_tri_in.extent() == std_e::multi_index<I8,2>{1,2} );
    MPI_CHECK(1, parent_tri_in.extent() == std_e::multi_index<I8,2>{0,2} );
    MPI_CHECK(0, parent_tri_in == cgns::md_array<I4,2>{{17,18}} );
    MPI_CHECK(1, parent_tri_in == cgns::md_array<I4,2>(0,2) );

    // quad ext
    auto parent_quad_ext = cgns::get_node_value_by_matching<I4,2>(b,"Zone/Quads/ParentElements");
    CHECK( parent_quad_ext.extent() == std_e::multi_index<I8,2>{6,2} );
    MPI_CHECK(0, parent_quad_ext == cgns::md_array<I4,2>{{15,0},{16,0},{17,0},{18,0},{15,0},{17,0}                                          } );
    MPI_CHECK(1, parent_quad_ext == cgns::md_array<I4,2>{                                          {16,0},{18,0},{15,0},{16,0},{15,0},{16,0}} );

    // quad in
    auto elt_type_quad_in = cgns::get_node_value_by_matching<I4>(b,"Zone/QUAD_4_interior")[0];
    auto range_quad_in    = cgns::get_node_value_by_matching<I4>(b,"Zone/QUAD_4_interior/ElementRange");
    auto connec_quad_in   = cgns::get_node_value_by_matching<I4>(b,"Zone/QUAD_4_interior/ElementConnectivity");
    auto parent_quad_in   = cgns::get_node_value_by_matching<I4,2>(b,"Zone/QUAD_4_interior/ParentElements");

    CHECK( elt_type_quad_in == (I4)cgns::QUAD_4 );
    CHECK( range_quad_in == std::vector<I4>{20,22} );
    MPI_CHECK(0, connec_quad_in == std::vector<I4>{2,5,10,7, 6,7,10,9, 7,12,15,10} ); // Note: the first and last faces are flipped
                                                                                      // compared to sequential...
    MPI_CHECK(1, connec_quad_in == std::vector<I4>{} );
    MPI_CHECK(0, parent_quad_in.extent() == std_e::multi_index<I8,2>{3,2} );
    MPI_CHECK(1, parent_quad_in.extent() == std_e::multi_index<I8,2>{0,2} );
    MPI_CHECK(0, parent_quad_in == cgns::md_array<I4,2>{{15,17},{15,16},{18,16}} ); // ... and so are the parent elements. So this is coherent
                                                                                    // The difference comes from the fact that we use std::sort,
                                                                                    // not std::stable_sort
    MPI_CHECK(1, parent_quad_in == cgns::md_array<I4,2>(0,2) );
  }
}
