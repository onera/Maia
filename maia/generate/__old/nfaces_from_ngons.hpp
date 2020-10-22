#pragma once


namespace cgns {

// Fwd decl
struct tree;
class factory;


auto
add_nfaces(tree& b, const factory& F) -> void;


} // cgns
