#pragma once

#include <vector>
#include <string>
#include <mpi.h>
#include "pdm.h"

std::vector<PDM_g_num_t> generate_global_numbering(/* TODO const */ std::vector<std::string>& partPaths,
                                                   MPI_Comm                  comm);
