#include <pybind11/pybind11.h>

#include "std_e/logging/log.hpp"
#include "std_e/logging/printer_common.hpp"
#include "std_e/logging/printer_mpi.hpp"

namespace py = pybind11;

void register_logging_module(py::module_& parent) {

  py::module_ m = parent.def_submodule("logging");

  m.def("log", [](const std::string& logger, const std::string& s){
                 std_e::log(logger, s + '\n'); // contrary to C++, the Python convention is
               }                               // to insert '\n' automatically
  );

  m.def("turn_on", &std_e::turn_on);
  m.def("turn_off", &std_e::turn_off);

  m.def("add_logger", [](const std::string& logger_name){ std_e::add_logger(logger_name); });

  m.def("add_stdout_printer           ", &std_e::add_stdout_printer           );
  m.def("add_file_printer             ", &std_e::add_file_printer             );
  m.def("add_mpi_stdout_printer       ", &std_e::add_mpi_stdout_printer       );
  m.def("add_mpi_rank_0_stdout_printer", &std_e::add_mpi_rank_0_stdout_printer);
  m.def("add_mpi_file_printer         ", &std_e::add_mpi_file_printer         );
}
