#include "std_e/logging/log.hpp"
#include "std_e/logging/printer_mpi.hpp"

namespace {

bool init_maia_default_loggers() {
  // ensures that these loggers are present for maia to run even if no logging configuration file was found
  std_e::add_logger_if_absent("maia"             ,std::make_unique<std_e::mpi_stdout_printer>());
  std_e::add_logger_if_absent("maia.rank_0"      ,std::make_unique<std_e::mpi_rank_0_stdout_printer>());
  std_e::add_logger_if_absent("maia perf level 0",std::make_unique<std_e::mpi_rank_0_stdout_printer>());
  std_e::add_logger_if_absent("maia perf level 1",std::make_unique<std_e::mpi_stdout_printer>());
  std_e::add_logger_if_absent("maia perf level 2",std::make_unique<std_e::mpi_stdout_printer>());
  return true;
}

// constant here just to trigger initialization
const bool _ = init_maia_default_loggers();

}
