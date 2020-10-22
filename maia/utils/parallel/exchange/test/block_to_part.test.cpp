#include "doctest/extensions/doctest_mpi.h"

#include "maia/utils/parallel/exchange/block_to_part.hpp"
#include "maia/utils/parallel/exchange/test/example.hpp"

MPI_TEST_CASE("block_to_part",2) {
  SUBCASE("cells") {
    auto dist = distribution_cells();
    auto LN_to_GN = LN_to_GN_cells(test_rank);

    pdm::block_to_part_protocol btp_protocol(test_comm,dist,LN_to_GN);

    SUBCASE("variadic size") {
      auto rho_dist = density_string_block(test_rank);
      auto rho_part = pdm::exchange(btp_protocol,rho_dist);

      MPI_REQUIRE( 0 , rho_dist[0] == "rho_0"       );
      MPI_REQUIRE( 0 , rho_dist[1] == "rho_1_a"     );
      MPI_REQUIRE( 0 , rho_dist[2] == "rho_2_ab"    );

      MPI_REQUIRE( 1 , rho_dist[0] == "rho_3_abc"   );
      MPI_REQUIRE( 1 , rho_dist[1] == "rho_4_abcd"  );
      MPI_REQUIRE( 1 , rho_dist[2] == "rho_5_abcde" );

      MPI_REQUIRE( 0 , LN_to_GN == std::vector{1,0,5} );
      MPI_REQUIRE( 1 , LN_to_GN == std::vector{3,2,4} );

      MPI_CHECK  ( 0 , rho_part[0] == "rho_1_a"     );
      MPI_CHECK  ( 0 , rho_part[1] == "rho_0"       );
      MPI_CHECK  ( 0 , rho_part[2] == "rho_5_abcde" );

      MPI_CHECK  ( 1 , rho_part[0] == "rho_3_abc"   );
      MPI_CHECK  ( 1 , rho_part[1] == "rho_2_ab"    );
      MPI_CHECK  ( 1 , rho_part[2] == "rho_4_abcd"  );
    }

    SUBCASE("constant size") {
      auto rho_dist = density_block(test_rank);
      auto rho_part = pdm::exchange(btp_protocol,rho_dist);

      auto rho = density_field(); // of course, here we have the complete field for testing,
                                        // but in a distributed context we don't want to build this object!
      // note: the equality is of the same form on rank 0 and 1, but the value tested are actually different
      CHECK( rho_part.size() == 3 );
      CHECK( rho_part[0] == rho[LN_to_GN[0]] );
      CHECK( rho_part[1] == rho[LN_to_GN[1]] );
      CHECK( rho_part[2] == rho[LN_to_GN[2]] );
    }
  }

  SUBCASE("vertices") {
    SUBCASE("constant size") {
      auto dist = distribution_vertices();
      auto LN_to_GN = LN_to_GN_vertices(test_rank);

      pdm::block_to_part_protocol btp_protocol(test_comm,dist,LN_to_GN);

      auto X_dist = X_block(test_rank);
      auto X_part = pdm::exchange(btp_protocol,X_dist);

      auto X = X_field();

       // note: the equality is of the same form on rank 0 and 1, but the value tested are actually different
      CHECK( X_part.size() == 8 );
      CHECK( X_part[0] == X[LN_to_GN[0]] );
      CHECK( X_part[1] == X[LN_to_GN[1]] );
      CHECK( X_part[2] == X[LN_to_GN[2]] );
      CHECK( X_part[3] == X[LN_to_GN[3]] );
      CHECK( X_part[4] == X[LN_to_GN[4]] );
      CHECK( X_part[5] == X[LN_to_GN[5]] );
      CHECK( X_part[6] == X[LN_to_GN[6]] );
      CHECK( X_part[7] == X[LN_to_GN[7]] );
    }
    SUBCASE("constant size - multiple part") {
      auto dist = distribution_vertices();
      auto LN_to_GNs = LN_to_GN_vertices_3(test_rank);

      pdm::block_to_parts_protocol btp_protocol(test_comm,dist,LN_to_GNs);

      auto X_dist = X_block(test_rank);
      auto X_part = pdm::exchange(btp_protocol,X_dist);

      auto X = X_field();

      MPI_CHECK( 0 , X_part.size() == 1 );
      MPI_CHECK( 0 , X_part[0].size() == 8 );
      MPI_CHECK( 0 , X_part[0][0] == X[LN_to_GNs[0][0]] );
      MPI_CHECK( 0 , X_part[0][1] == X[LN_to_GNs[0][1]] );
      MPI_CHECK( 0 , X_part[0][2] == X[LN_to_GNs[0][2]] );
      MPI_CHECK( 0 , X_part[0][3] == X[LN_to_GNs[0][3]] );
      MPI_CHECK( 0 , X_part[0][4] == X[LN_to_GNs[0][4]] );
      MPI_CHECK( 0 , X_part[0][5] == X[LN_to_GNs[0][5]] );
      MPI_CHECK( 0 , X_part[0][6] == X[LN_to_GNs[0][6]] );
      MPI_CHECK( 0 , X_part[0][7] == X[LN_to_GNs[0][7]] );

      MPI_CHECK( 1 , X_part.size() == 2 );
      MPI_CHECK( 1 , X_part[0].size() == 4 );
      MPI_CHECK( 1 , X_part[1].size() == 6 );
      MPI_CHECK( 1 , X_part[0][0] == X[LN_to_GNs[0][0]] );
      MPI_CHECK( 1 , X_part[0][1] == X[LN_to_GNs[0][1]] );
      MPI_CHECK( 1 , X_part[0][2] == X[LN_to_GNs[0][2]] );
      MPI_CHECK( 1 , X_part[0][3] == X[LN_to_GNs[0][3]] );

      MPI_CHECK( 1 , X_part[1][0] == X[LN_to_GNs[1][0]] );
      MPI_CHECK( 1 , X_part[1][1] == X[LN_to_GNs[1][1]] );
      MPI_CHECK( 1 , X_part[1][2] == X[LN_to_GNs[1][2]] );
      MPI_CHECK( 1 , X_part[1][3] == X[LN_to_GNs[1][3]] );
      MPI_CHECK( 1 , X_part[1][4] == X[LN_to_GNs[1][4]] );
      MPI_CHECK( 1 , X_part[1][5] == X[LN_to_GNs[1][5]] );
    }
  }
}

