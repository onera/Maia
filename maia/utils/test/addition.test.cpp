#include "doctest/extensions/doctest_mpi.h"
#include "pdm.h"
#include "pdm_doctest.h"
#include "maia/utils/addition.hpp"


MPI_TEST_CASE("[1p] block_to_part",1) {
  auto res = addition(1, 2);
  CHECK(res == 3);
}
