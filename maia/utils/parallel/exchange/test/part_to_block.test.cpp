//#include "doctest/extensions/doctest_mpi.h"
//
//#include "maia/utils/parallel/exchange/part_to_block.hpp"
//#include "maia/utils/parallel/exchange/test/example.hpp"
//#include "std_e/log.hpp" // TODO
//
//MPI_TEST_CASE("part_to_block",2) {
//  SUBCASE("cells") {
//    auto dist = distribution_cells();
//    auto LN_to_GN = LN_to_GN_cells(test_rank);
//
//    pdm::part_to_unique_block_protocol ptb_protocol(test_comm,dist,LN_to_GN);
//
//    SUBCASE("variadic size") {
//      auto rho_part = density_string_part(test_rank);
//      auto rho_dist = pdm::exchange(ptb_protocol,rho_part);
//
//      MPI_REQUIRE( 0 , rho_part[0] == "rho_1_a"     );
//      MPI_REQUIRE( 0 , rho_part[1] == "rho_0"       );
//      MPI_REQUIRE( 0 , rho_part[2] == "rho_5_abcde" );
//                 
//      MPI_REQUIRE( 1 , rho_part[0] == "rho_3_abc"   );
//      MPI_REQUIRE( 1 , rho_part[1] == "rho_2_ab"    );
//      MPI_REQUIRE( 1 , rho_part[2] == "rho_4_abcd"  );
//
//      MPI_REQUIRE( 0 , LN_to_GN == std::vector{1,0,5} );
//      MPI_REQUIRE( 1 , LN_to_GN == std::vector{3,2,4} );
//
//      MPI_CHECK  ( 0 , rho_dist[0] == "rho_0"       );
//      MPI_CHECK  ( 0 , rho_dist[1] == "rho_1_a"     );
//      MPI_CHECK  ( 0 , rho_dist[2] == "rho_2_ab"    );
//      
//      MPI_CHECK  ( 1 , rho_dist[0] == "rho_3_abc"   );
//      MPI_CHECK  ( 1 , rho_dist[1] == "rho_4_abcd"  );
//      MPI_CHECK  ( 1 , rho_dist[2] == "rho_5_abcde" );
//    }
//
//    SUBCASE("constant size") {
//      auto rho_part = density_part(test_rank);
//      auto rho_dist = pdm::exchange(ptb_protocol,rho_part);
//
//      auto rho = density_field(); // of course, here we have the complete field for testing,
//                                  // but in a distributed context we don't want to build this object!
//      MPI_CHECK( 0 , rho_dist.size() == 3 );
//      MPI_CHECK( 0 , rho_dist[0] == rho[0] );
//      MPI_CHECK( 0 , rho_dist[1] == rho[1] );
//      MPI_CHECK( 0 , rho_dist[2] == rho[2] );
//      MPI_CHECK( 1 , rho_dist.size() == 3 );
//      MPI_CHECK( 1 , rho_dist[0] == rho[0+dist[1]] );
//      MPI_CHECK( 1 , rho_dist[1] == rho[1+dist[1]] );
//      MPI_CHECK( 1 , rho_dist[2] == rho[2+dist[1]] );
//    }
//  }
//
//  SUBCASE("vertices") {
//    SUBCASE("constant size") {
//      SUBCASE("unique") {
//        auto dist = distribution_vertices();
//        auto LN_to_GN = LN_to_GN_vertices(test_rank);
//
//        pdm::part_to_unique_block_protocol btp_protocol(test_comm,dist,LN_to_GN);
//
//        auto X_p = X_part(test_rank);
//        auto X_dist = pdm::exchange(btp_protocol,X_p);
//
//        auto X = X_field();
//
//        MPI_CHECK( 0 , X_dist.size() == 6 );
//        MPI_CHECK( 0 , X_dist[0] == X[0] );
//        MPI_CHECK( 0 , X_dist[1] == X[1] );
//        MPI_CHECK( 0 , X_dist[2] == X[2] );
//        MPI_CHECK( 0 , X_dist[3] == X[3] );
//        MPI_CHECK( 0 , X_dist[4] == X[4] );
//        MPI_CHECK( 0 , X_dist[5] == X[5] );
//        MPI_CHECK( 1 , X_dist.size() == 6 );
//        MPI_CHECK( 1 , X_dist[0] == X[0+dist[1]] );
//        MPI_CHECK( 1 , X_dist[1] == X[1+dist[1]] );
//        MPI_CHECK( 1 , X_dist[2] == X[2+dist[1]] );
//        MPI_CHECK( 1 , X_dist[3] == X[3+dist[1]] );
//        MPI_CHECK( 1 , X_dist[4] == X[4+dist[1]] );
//        MPI_CHECK( 1 , X_dist[5] == X[5+dist[1]] );
//      }
//      SUBCASE("merge multiple") {
//        auto dist = distribution_vertices();
//        auto LN_to_GN = LN_to_GN_vertices(test_rank);
//
//        pdm::part_to_merged_block_protocol btp_protocol(test_comm,dist,LN_to_GN);
//
//        auto X_p = X_part(test_rank);
//        auto X_dist = pdm::exchange2(btp_protocol,X_p,std::plus<>{});
//
//        // NOTE: ln to gn:
//        // part0: {2,11,3,10,9,1,8,0};
//        // part1: {3,4,9,10,5,8,7,6};
//        // Here we compute the two parts, but of course in reality, each proc get its on parts
//        auto X_part_0 = X_part(0);
//        auto X_part_1 = X_part(1);
//
//        MPI_CHECK( 0 , X_dist.size() == 6 );
//        MPI_CHECK( 0 , X_dist[0] == X_part_0[7] );
//        MPI_CHECK( 0 , X_dist[1] == X_part_0[5] );
//        MPI_CHECK( 0 , X_dist[2] == X_part_0[0] );
//        MPI_CHECK( 0 , X_dist[3] == X_part_0[2]+X_part_1[0] );
//        MPI_CHECK( 0 , X_dist[4] == X_part_1[1] );
//        MPI_CHECK( 0 , X_dist[5] == X_part_1[4] );
//        MPI_CHECK( 1 , X_dist.size() == 6 );
//        MPI_CHECK( 1 , X_dist[0] == X_part_1[7] );
//        MPI_CHECK( 1 , X_dist[1] == X_part_1[6] );
//        MPI_CHECK( 1 , X_dist[2] == X_part_0[7]+X_part_1[6] );
//        MPI_CHECK( 1 , X_dist[3] == X_part_0[4]+X_part_1[2] );
//        MPI_CHECK( 1 , X_dist[4] == X_part_0[3]+X_part_1[3] );
//        MPI_CHECK( 1 , X_dist[5] == X_part_0[1] );
//      }
//    }
//    //SUBCASE("constant size - multiple part") {
//    //  auto dist = distribution_vertices();
//    //  auto LN_to_GNs = LN_to_GN_vertices_3(test_rank);
//
//    //  pdm::block_to_parts_protocol btp_protocol(test_comm,dist,LN_to_GNs);
//
//    //  auto X_dist = X_block(test_rank);
//    //  auto X_part = pdm::exchange(btp_protocol,X_dist);
//
//    //  auto X = X_field();
//
//    //  MPI_CHECK( 0 , X_part.size() == 1 );
//    //  MPI_CHECK( 0 , X_part[0].size() == 8 );
//    //  MPI_CHECK( 0 , X_part[0][0] == X[LN_to_GNs[0][0]] );
//    //  MPI_CHECK( 0 , X_part[0][1] == X[LN_to_GNs[0][1]] );
//    //  MPI_CHECK( 0 , X_part[0][2] == X[LN_to_GNs[0][2]] );
//    //  MPI_CHECK( 0 , X_part[0][3] == X[LN_to_GNs[0][3]] );
//    //  MPI_CHECK( 0 , X_part[0][4] == X[LN_to_GNs[0][4]] );
//    //  MPI_CHECK( 0 , X_part[0][5] == X[LN_to_GNs[0][5]] );
//    //  MPI_CHECK( 0 , X_part[0][6] == X[LN_to_GNs[0][6]] );
//    //  MPI_CHECK( 0 , X_part[0][7] == X[LN_to_GNs[0][7]] );
//
//    //  MPI_CHECK( 1 , X_part.size() == 2 );
//    //  MPI_CHECK( 1 , X_part[0].size() == 4 );
//    //  MPI_CHECK( 1 , X_part[1].size() == 6 );
//    //  MPI_CHECK( 1 , X_part[0][0] == X[LN_to_GNs[0][0]] );
//    //  MPI_CHECK( 1 , X_part[0][1] == X[LN_to_GNs[0][1]] );
//    //  MPI_CHECK( 1 , X_part[0][2] == X[LN_to_GNs[0][2]] );
//    //  MPI_CHECK( 1 , X_part[0][3] == X[LN_to_GNs[0][3]] );
//
//    //  MPI_CHECK( 1 , X_part[1][0] == X[LN_to_GNs[1][0]] );
//    //  MPI_CHECK( 1 , X_part[1][1] == X[LN_to_GNs[1][1]] );
//    //  MPI_CHECK( 1 , X_part[1][2] == X[LN_to_GNs[1][2]] );
//    //  MPI_CHECK( 1 , X_part[1][3] == X[LN_to_GNs[1][3]] );
//    //  MPI_CHECK( 1 , X_part[1][4] == X[LN_to_GNs[1][4]] );
//    //  MPI_CHECK( 1 , X_part[1][5] == X[LN_to_GNs[1][5]] );
//    //}
//  }
//}
//
