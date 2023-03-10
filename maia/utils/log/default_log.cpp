#include "std_e/logging/mpi_log.hpp"

namespace {

bool init_maia_default_loggers() {
  // maia default logger
  //   - explicit_call by default: does not print anything unless explicitly called
  //   - only created if nothing else (does not overwrite if something else exists)
  if (!std_e::has_logger("maia")) {
    std_e::add_logger({"maia",std_e::explicit_call,std::make_unique<std_e::mpi_rank_0_stdout_printer>()});
  }
  std_e::add_logger({"maia perf level 0",std_e::full,std::make_unique<std_e::mpi_rank_0_stdout_printer>()});
  std_e::add_logger({"maia perf level 1",std_e::full,std::make_unique<std_e::mpi_stdout_printer>()});
  std_e::add_logger({"maia perf level 2",std_e::full,std::make_unique<std_e::mpi_stdout_printer>()});
  return true;
}

// constant here just to trigger initialization
const bool _ = init_maia_default_loggers();

}
