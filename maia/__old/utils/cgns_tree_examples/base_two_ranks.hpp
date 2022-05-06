#pragma once


#include "cpp_cgns/cgns.hpp"


namespace example {

auto
create_base_two_ranks(int mpi_rank) -> cgns::tree;

} // example
