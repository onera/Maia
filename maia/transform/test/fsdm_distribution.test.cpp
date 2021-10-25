#include "std_e/unit_test/doctest_mpi.hpp"

#include "maia/transform/fsdm_distribution.hpp"

// using std::vector;
// MPI_TEST_CASE("distribute_bc_ids_to_match_face_dist",2) {
//   vector<int> tri_dist = {0,10,21};
//   vector<int> quad_dist = {0,14,27};

//   vector<int> element_intervals = {1,21,48}; // tris in [1,21), quads in [21,48)

//   vector<int> point_list;
//   vector<vector<double>> values;
//   if (test_rank==0) {
//     point_list = {40 , 1 , 2 ,13 ,41 };
//     values   = { { 0., 1., 2., 3., 4.},
//                  {10.,11.,12.,13.,14.} };
//   } else {
//     STD_E_ASSERT(test_rank==1);
//     //point_list = {42 , 5 ,11 };
//     point_list = {42 , 5 ,10, 11 }; // Note: 10 is on rank 0, because
//     values   = { { 5., 6., 7., 8.},
//                  {15.,16.,17.,18.} };
//   }

//   SUBCASE("repartition_by_distributions") {
//     auto partition_indices = maia::repartition_by_distributions(
//       vector{tri_dist,quad_dist},
//       element_intervals,
//       point_list,
//       values
//     );
//     MPI_CHECK( 0 , point_list == vector{ 2 , 1 ,40 ,13 ,41 } );
//     MPI_CHECK( 0 , values[0]  == vector{ 2., 1., 0., 3., 4.} );
//     MPI_CHECK( 0 , values[1]  == vector{12.,11.,10.,13.,14.} );
//     MPI_CHECK( 0 , partition_indices == vector{0,2,5} );

//     MPI_CHECK( 1 , point_list == vector{10 , 5 ,42 ,11 } );
//     MPI_CHECK( 1 , values[0]  == vector{ 7., 6., 5., 8.} );
//     MPI_CHECK( 1 , values[1]  == vector{17.,16.,15.,18.} );
//     MPI_CHECK( 1 , partition_indices == vector{0,2,4} );
//   }


//   SUBCASE("final") {
//     auto [pl_new,values_new] = maia::redistribute_to_match_face_dist(
//       vector{tri_dist,quad_dist},
//       element_intervals,
//       point_list,
//       values,
//       test_comm
//     );

//     MPI_CHECK( 0 , pl_new        == vector{ 2 , 1 ,10 , 5 } );
//     MPI_CHECK( 0 , values_new[0] == vector{ 2., 1., 7., 6.} );
//     MPI_CHECK( 0 , values_new[1] == vector{12.,11.,17.,16.} );
//     MPI_CHECK( 1 , pl_new        == vector{40 ,13 ,41 ,42 ,11 } );
//     MPI_CHECK( 1 , values_new[0] == vector{ 0., 3., 4., 5., 8.} );
//     MPI_CHECK( 1 , values_new[1] == vector{10.,13.,14.,15.,18.} );
//   }
// }
