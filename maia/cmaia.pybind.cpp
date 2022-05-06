#include <pybind11/pybind11.h>

#include "maia/pytree/pytree.pybind.hpp"
#include "maia/utils/utils.pybind.hpp"
#include "maia/algo/tree_algo.pybind.hpp"
#include "maia/algo/dist/dist_algo.pybind.hpp"
#include "maia/algo/part/part_algo.pybind.hpp"


namespace py = pybind11;

PYBIND11_MODULE(cmaia, m) {

    #if __cplusplus > 201703L
    m.attr("cpp20_enabled") = py::bool_(1);
    #else
    m.attr("cpp20_enabled") = py::bool_(0);
    #endif

    register_pytree_module(m);
    register_utils_module(m);

    register_tree_algo_module(m);
    register_dist_algo_module(m);
    register_part_algo_module(m);

}
