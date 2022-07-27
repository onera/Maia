#pragma once


#include <string>
#include "cpp_cgns/cgns.hpp"
#include <mpi.h>


namespace maia {


auto file_to_dist_tree(const std::string& file_name, MPI_Comm comm) -> cgns::tree;


} // maia
