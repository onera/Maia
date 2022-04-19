#pragma once


#include <string>
#include "cpp_cgns/cgns.hpp"


namespace maia {


auto
to_node(const std::string& yaml_str) -> cgns::tree;
auto
to_nodes(const std::string& yaml_str) -> std::vector<cgns::tree>;
auto
to_cgns_tree(const std::string& yaml_str) -> cgns::tree;


} // maia
