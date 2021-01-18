#include "std_e/logging/mpi_log.hpp"

// Contrary to other .test.cpp files, this one does not test anything
// It is only here to initialize the test logger
// It has the .test.cpp suffix convention in order to be picked up when building the test executable

namespace {

bool init_maia_test_logger() {
  // maia test logger
  //   - always off
  //   - this file is only compiled for tests, so overwrites the maia default logger
  std_e::logger maia_test_logger = {"maia",std_e::off,std::make_unique<std_e::mpi_rank_0_stdout_printer>()};
  if (!std_e::has_logger("maia")) {
    std_e::add_logger( std::move(maia_test_logger) );
  } else {
    std_e::get_logger("maia") = std::move(maia_test_logger); // overwrite default
  }
  return true;
}

// constant here just to trigger initialization
const bool _ = init_maia_test_logger();

}
