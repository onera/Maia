#include "std_e/unit_test/doctest.hpp"

#include "maia/utils/yaml/parse_yaml_cgns.hpp"

#include "cpp_cgns/tree_manip.hpp"

using namespace maia;
using namespace cgns;

TEST_CASE("parse_yaml_cgns") {
  std::string yaml_tree =
    "Base CGNSBase_t I4 [3,3]:\n"
    "  Zone0 Zone_t I4 [[ 8,1,0]]:\n"
    "  Zone1 Zone_t I4 [[12,2,0]]:\n";
  tree t = to_node(yaml_tree);

  CHECK(t.name == "Base");
  CHECK(t.label == "CGNSBase_t");
  CHECK(view_as_span<I4>(t.value) == std::vector{3,3});

  CHECK(get_child_by_name(t,"Zone0").name == "Zone0");
  CHECK(view_as_span<I4>(get_child_by_name(t,"Zone0").value) == std::vector{8,1,0});
  CHECK(get_child_by_name(t,"Zone1").name == "Zone1");
}
