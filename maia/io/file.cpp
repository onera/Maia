#if __cplusplus > 201703L
#include "maia/io/file.hpp"
#include "maia/utils/parallel/mpi4py.hpp"


#include "pybind11/embed.h"
namespace py = pybind11;
#include "cpp_cgns/interop/pycgns_converter.hpp"
#include "std_e/utils/embed_python.hpp"
#include "std_e/future/ranges.hpp"


namespace maia {


auto
file_to_dist_tree(const std::string& file_name, MPI_Comm comm) -> cgns::tree {
  std_e::throw_if_no_python_interpreter(__func__);
  auto m = py::module_::import("maia.io.cgns_io_tree");

  auto mpi4py_comm = comm_to_mpi4py_comm(comm);
  py::object py_tree = m .attr("file_to_dist_tree")(file_name,mpi4py_comm);
  return cgns::to_cpp_tree_copy(py_tree);
}


} // maia
#endif // C++>17
