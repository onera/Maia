#pragma once


#include "cpp_cgns/cgns.hpp"
#include "cpp_cgns/sids/creation.hpp"


auto create_unstructured_base(cgns::factory F) -> cgns::tree;
