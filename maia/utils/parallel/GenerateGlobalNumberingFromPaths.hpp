#pragma once

#include <vector>
#include <string>
#include <mpi.h>

std::vector<int> generate_global_numbering(/* TODO const */ std::vector<std::string>& partPaths,
                                           MPI_Comm                  comm);
