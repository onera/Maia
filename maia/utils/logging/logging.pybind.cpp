#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "std_e/logging/log.hpp"
#include "std_e/logging/build_printer_from_strings.hpp"
#include "std_e/base/msg_exception.hpp"

namespace py = pybind11;


class python_printer : public std_e::printer {
  public:
    python_printer() = default;

    python_printer(python_printer&& o)
      : p(o.p)
    {
      o.p = py::handle{};
    }
    python_printer& operator=(python_printer&& o) {
      p = o.p;
      o.p = py::handle{};
      return *this;
    }
    python_printer(const python_printer& o) = delete;
    python_printer& operator=(const python_printer& o) = delete;
  
    python_printer(py::handle p)
      : p(p)
    {
      Py_XINCREF(p.ptr());
    }

    ~python_printer()
    {
      // During normal ending of the program, the interpreter is deinitialized before the std_e::all_loggers global variable.
      // When the Python interpreter is finalized, all its memory is freed
      // (see https://docs.python.org/3/c-api/init.html#c.Py_FinalizeEx).
      // Hence, the PyObject of `p.ptr()` has already been freed: we don't need to (and mustn't) decrement the object counter
      if (Py_IsInitialized()) {
        Py_XDECREF(p.ptr());
      }
    }
    
    auto log(const std::string& msg) -> void override {
      if (Py_IsInitialized()) {
        p.attr("log")(msg);
      } else {
        throw std_e::msg_exception(
          "python_printer: trying to log whereas the Python interpreter is not initialized"
        );
      }
    }
  private:
    py::handle p;
};


void register_logging_module(py::module_& parent) {

  py::module_ m = parent.def_submodule("logging");

  m.def("log", [](const std::string& logger, const std::string& s){
                 std_e::log(logger, s + '\n'); // contrary to C++, the Python convention is
               }                               // to insert '\n' automatically
  );

  m.def(
    "add_logger",
    [](const std::string& logger_name, bool replace){ std_e::add_logger(std_e::logger{logger_name}, replace); },
    py::arg("logger_name"), py::kw_only(), py::arg("replace") = py::bool_(true)
  );

  m.def(
    "_add_printer_obj_to_logger",
    [](const std::string& logger_name, py::handle p){ std_e::add_printer_to_logger(logger_name, python_printer{p}); }
  );
  m.def(
    "_add_printer_type_to_logger",
    [](const std::string& logger_name, const std::string& printer_type, py::list args){
      std_e::add_printer(
        logger_name, 
        printer_type, 
        args.cast<std::vector<std::string>>()
      );
    }
  );

  m.def("turn_on", &std_e::turn_on);
  m.def("turn_off", &std_e::turn_off);
}
