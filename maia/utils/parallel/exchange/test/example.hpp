#pragma once

#include <vector>
#include "std_e/future/contract.hpp"
#include "std_e/utils/concatenate.hpp"
#include "maia/utils/parallel/distribution.hpp"

/*

    ______________________________________________________________
  2|                  11|                   3|                   4|
   |                    |                    |                    |
   |                    |                    |                    |
   |                    |                    |                    |
   |                    |                    |                    |
   |         1          |         5          |         4          |
   |                    |                    |                    |
   |                    |                    |                    |
   |                    |                    |                    |
   |                    |                    |                    |
   |____________________|____________________|____________________|
  1|                   9|                  10|                   5|
   |                    |                    |                    |
   |                    |                    |                    |
   |                    |                    |                    |
   |                    |                    |                    |
   |         0          |         3          |         2          |
   |                    |                    |                    |
   |                    |                    |                    |
   |                    |                    |                    |
   |                    |                    |                    |
   |____________________|____________________|____________________|
  0                    8                    7                    6


    _________________________________________
 0 |                  1 |                  2 |\
(2)|                (11)|                 (3)|
   |                    |                    |  \
   |                    |                    |
   |                    |                    |    \ ____________________
   |         0          |         2          |   0 |                  1 |
   |        (1)         |        (5)         |  (3)|                 (4)|
   |                    |                    |     |                    |
   |                    |                    |     |                    |
   |                    |                    |     |                    |
   |____________________|____________________|     |         2          |
 5 |                  4 |\                  3 \    |        (4)         |
(1)|                 (9)|                 (10)     |                    |
   |                    |  \                    \  |                    |
   |                    |                          |                    |
   |                    |    \ ___________________\|____________________|
   |         1          |   2 |                  3 |                  4 |
   |        (0)         |  (9)|                (10)|                 (5)|
   |                    |     |                    |                    |
   |                    |     |                    |                    |
   |                    |     |                    |                    |
   |____________________|     |         0          |         1          |
 7                     6 \    |        (3)         |        (2)         |
(0)                   (8)     |                    |                    |
                           \  |                    |                    |
                              |                    |                    |
                             \|____________________|____________________|
                             5                    6                    7
                            (8)                  (7)                  (6)

*/

// distribution {
inline auto
distribution_cells() -> distribution_vector<int> {
  return {0,3,6};
}
inline auto
LN_to_GN_cells(int rank) -> std::vector<int> {
  switch (rank) {
    case 0: return {1,0,5};
    case 1: return {3,2,4};
    default: STD_E_ASSERT(0); return {};
  }
}

inline auto
distribution_vertices() -> distribution_vector<int> {
  return {0,6,12};
}
inline auto
LN_to_GN_vertices(int rank) -> std::vector<int> {
  switch (rank) {
    case 0: return {2,11,3,10,9,1,8,0};
    case 1: return {3,4,9,10,5,8,7,6};
    default: STD_E_ASSERT(0); return {};
  }
}
// distribution }


// cell string field {
inline auto
density_string_block(int rank) -> std::vector<std::string> {
  switch (rank) {
    case 0: return {"rho_0"     , "rho_1_a"   , "rho_2_ab"   };
    case 1: return {"rho_3_abc" , "rho_4_abcd", "rho_5_abcde"};
    default: STD_E_ASSERT(0); return {};
  }
}
inline auto
density_string_field() -> std::vector<std::string> {
  return std_e::concatenate(density_string_block(0),density_string_block(1));
}
inline auto
density_string_part(int rank) -> std::vector<std::string> {
  switch (rank) {
    case 0: return {"rho_1_a"   , "rho_0"    , "rho_5_abcde"};
    case 1: return {"rho_3_abc" , "rho_2_ab" , "rho_4_abcd" };
    default: STD_E_ASSERT(0); return {};
  }
}
TEST_CASE("density_string_block, density_string_part, LN_to_GN coherency") {
  auto rho = density_string_field();

  CHECK( density_string_part(0)[0] == rho[LN_to_GN_cells(0)[0]] );
  CHECK( density_string_part(0)[1] == rho[LN_to_GN_cells(0)[1]] );
  CHECK( density_string_part(0)[2] == rho[LN_to_GN_cells(0)[2]] );

  CHECK( density_string_part(1)[0] == rho[LN_to_GN_cells(1)[0]] );
  CHECK( density_string_part(1)[1] == rho[LN_to_GN_cells(1)[1]] );
  CHECK( density_string_part(1)[2] == rho[LN_to_GN_cells(1)[2]] );
}
// cell string field }


// cell double field {
inline auto
density_block(int rank) -> std::vector<double> {
  switch (rank) {
    case 0: return {1.1, 1.9, 1.6};
    case 1: return {1.8, 1.7, 1.2};
    default: STD_E_ASSERT(0); return {};
  }
}
inline auto
density_field() -> std::vector<double> {
  return std_e::concatenate(density_block(0),density_block(1));
}
inline auto
density_part(int rank) -> std::vector<double> {
  switch (rank) {
    case 0: return {1.9, 1.1, 1.2};
    case 1: return {1.8, 1.6, 1.7};
    default: STD_E_ASSERT(0); return {};
  }
}
TEST_CASE("density_block, density_part, LN_to_GN coherency") {
  auto rho = density_field();

  CHECK( density_part(0)[0] == rho[LN_to_GN_cells(0)[0]] );
  CHECK( density_part(0)[1] == rho[LN_to_GN_cells(0)[1]] );
  CHECK( density_part(0)[2] == rho[LN_to_GN_cells(0)[2]] );

  CHECK( density_part(1)[0] == rho[LN_to_GN_cells(1)[0]] );
  CHECK( density_part(1)[1] == rho[LN_to_GN_cells(1)[1]] );
  CHECK( density_part(1)[2] == rho[LN_to_GN_cells(1)[2]] );
}
// cell double field }


// vertices double field {
inline auto
X_block(int rank) -> std::vector<double> {
  switch (rank) {
    case 0: return {0.,0.,0.,2.,3.,3.};
    case 1: return {3.,2.,1.,1.,2.,1.};
    default: STD_E_ASSERT(0); return {};
  }
}
inline auto
X_field() -> std::vector<double> {
  return std_e::concatenate(X_block(0),X_block(1));
}
inline auto
X_part(int rank) -> std::vector<double> {
  switch (rank) {
    case 0: return {0.,1.,2.,2.,1.,0.,1.,0.};
    case 1: return {2.,3.,1.,2.,3.,1.,2.,3.};
    default: STD_E_ASSERT(0); return {};
  }
}
TEST_CASE("X_block, X_part, LN_to_GN coherency") {
  auto X = X_field();

  CHECK( X_part(0)[0] == X[LN_to_GN_vertices(0)[0]] );
  CHECK( X_part(0)[1] == X[LN_to_GN_vertices(0)[1]] );
  CHECK( X_part(0)[2] == X[LN_to_GN_vertices(0)[2]] );
  CHECK( X_part(0)[3] == X[LN_to_GN_vertices(0)[3]] );
  CHECK( X_part(0)[4] == X[LN_to_GN_vertices(0)[4]] );
  CHECK( X_part(0)[5] == X[LN_to_GN_vertices(0)[5]] );
  CHECK( X_part(0)[6] == X[LN_to_GN_vertices(0)[6]] );
  CHECK( X_part(0)[7] == X[LN_to_GN_vertices(0)[7]] );

  CHECK( X_part(1)[0] == X[LN_to_GN_vertices(1)[0]] );
  CHECK( X_part(1)[1] == X[LN_to_GN_vertices(1)[1]] );
  CHECK( X_part(1)[2] == X[LN_to_GN_vertices(1)[2]] );
  CHECK( X_part(1)[3] == X[LN_to_GN_vertices(1)[3]] );
  CHECK( X_part(1)[4] == X[LN_to_GN_vertices(1)[4]] );
  CHECK( X_part(1)[5] == X[LN_to_GN_vertices(1)[5]] );
  CHECK( X_part(1)[6] == X[LN_to_GN_vertices(1)[6]] );
  CHECK( X_part(1)[7] == X[LN_to_GN_vertices(1)[7]] );
}
// vertices double field }

/*
                      part 0                               part 1           part 2
                        |                                    |                |
                        v                                    |                |
    _________________________________________                |                |
 0 |                  1 |                  2 |-              v                |
(2)|                (11)|                 (3)|  \   ____________________      |
   |                    |                    |    -| 0                1 |     |
   |                    |                    |     |(3)              (4)|     |
   |                    |                    |     |                    |     |
   |         0          |         2          |     |                    |     |
   |        (1)         |        (5)         |     |                    |     |
   |                    |                    |     |         0          |     |
   |                    |                    |     |        (4)         |     |
   |                    |                    |     |                    |     |
   |____________________|____________________|-    |                    |     |
 5 |                  4 |\                  3 \ \  |                    |     |
(1)|                 (9)|                 (10)    -|____________________|     |
   |                    |  \                    \   3                 2       |
   |                    |                         (10)               (5)      |
   |                    |    \ ___________________\_____________________     /
   |         1          |   2 |                  3 |                  4 |  <-
   |        (0)         |  (9)|                (10)|                 (5)|
   |                    |     |                    |                    |
   |                    |     |                    |                    |
   |                    |     |                    |                    |
   |____________________|     |         0          |         1          |
 7                     6 \    |        (3)         |        (2)         |
(0)                   (8)     |                    |                    |
                           \  |                    |                    |
                              |                    |                    |
                             \|____________________|____________________|
                             1                    0                    5
                            (8)                  (7)                  (6)

*/

// distribution {
inline auto
LN_to_GN_cells_3(int rank) -> std::vector<std::vector<int>> {
  switch (rank) {
    case 0: return {{1,0,5}};
    case 1: return {{4},{3,2}};
    default: STD_E_ASSERT(0); return {};
  }
}

inline auto
LN_to_GN_vertices_3(int rank) -> std::vector<std::vector<int>> {
  switch (rank) {
    case 0: return {{2,11,3,10,9,1,8,0}};
    case 1: return {{3,4,5,10},{7,8,9,10,5,6}};
    default: STD_E_ASSERT(0); return {};
  }
}
// distribution }

// vertices double field {
inline auto
X_part_3(int rank) -> std::vector<std::vector<double>> {
  switch (rank) {
    case 0: return {std::vector{0.,1.,2.,2.,1.,0.,1.,0.}};
    case 1: return {{2.,3.,3.,2.},{2.,1.,1.,2.,3.,3.}};
    default: STD_E_ASSERT(0); return {};
  }
}
TEST_CASE("X_block, X_part_3, LN_to_GN coherency") {
  auto X = X_field();

  auto X_part_0 = X_part_3(0)[0];
  auto X_part_1 = X_part_3(1)[0];
  auto X_part_2 = X_part_3(1)[1];
  auto LN_to_GN_0 = LN_to_GN_vertices_3(0)[0];
  auto LN_to_GN_1 = LN_to_GN_vertices_3(1)[0];
  auto LN_to_GN_2 = LN_to_GN_vertices_3(1)[1];
  CHECK( X_part_0[0] == X[LN_to_GN_0[0]] );
  CHECK( X_part_0[1] == X[LN_to_GN_0[1]] );
  CHECK( X_part_0[2] == X[LN_to_GN_0[2]] );
  CHECK( X_part_0[3] == X[LN_to_GN_0[3]] );
  CHECK( X_part_0[4] == X[LN_to_GN_0[4]] );
  CHECK( X_part_0[5] == X[LN_to_GN_0[5]] );
  CHECK( X_part_0[6] == X[LN_to_GN_0[6]] );
  CHECK( X_part_0[7] == X[LN_to_GN_0[7]] );

  CHECK( X_part_1[0] == X[LN_to_GN_1[0]] );
  CHECK( X_part_1[1] == X[LN_to_GN_1[1]] );
  CHECK( X_part_1[2] == X[LN_to_GN_1[2]] );
  CHECK( X_part_1[3] == X[LN_to_GN_1[3]] );

  CHECK( X_part_2[0] == X[LN_to_GN_2[0]] );
  CHECK( X_part_2[1] == X[LN_to_GN_2[1]] );
  CHECK( X_part_2[2] == X[LN_to_GN_2[2]] );
  CHECK( X_part_2[3] == X[LN_to_GN_2[3]] );
  CHECK( X_part_2[4] == X[LN_to_GN_2[4]] );
  CHECK( X_part_2[5] == X[LN_to_GN_2[5]] );
}
// vertices double field }
