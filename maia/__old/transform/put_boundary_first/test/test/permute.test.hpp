#if __cplusplus > 201703L
#include "std_e/unit_test/doctest.hpp"

#include "maia/__old/transform/put_boundary_first/permute.hpp"
#include "cpp_cgns/sids/creation.hpp"

using namespace cgns;
using std::vector;


TEST_CASE("boundary_ngons_at_beginning") {
  md_array<I4,2> parent_elts =
    { {1, 4},
      {0, 8},
      {0, 0},
      {3, 1} };
  auto pe_view = make_view(parent_elts);
  vector<I4> my_permutation = {1,2, 0,3};

  maia::permute_parent_elements(pe_view,my_permutation);

  md_array<I4,2> expected_parent_elts =
    { {0, 8},
      {0, 0},
      {1, 4},
      {3, 1} };
  CHECK( parent_elts == expected_parent_elts );
}
#endif // C++>17
