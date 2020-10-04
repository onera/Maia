#include "doctest/extensions/doctest_mpi.h"
#include "maia/utils/addition.hpp"


MPI_TEST_CASE("[1p] block_to_part",1) {
  auto res = add(1, 2);
  CHECK(res == 3);
  MPI_CHECK(0, res == 3);
  std::cout << "ollll" << std::endl;
}
