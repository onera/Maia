#include "maia/transform/fsdm_distribution.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "cpp_cgns/sids/sids.hpp"
#include "cpp_cgns/tree_manip.hpp"
#include "maia/utils/parallel/distribution.hpp"
#include "std_e/buffer/buffer_vector.hpp"


namespace cgns {

auto add_fsdm_distribution(tree& b, MPI_Comm comm) -> void {
  STD_E_ASSERT(b.label=="CGNSBase_t");
  auto zs = get_children_by_label(b,"Zone_t");
  if (zs.size()!=1) {
    throw cgns_exception("add_fsdm_distribution (as FSDM) expects only one zone per process");
  }
  tree& z = zs[0];

  auto fsdm_dist_node = new_UserDefinedData("FSDM_elt_distributions");

  int n_rank = std_e::nb_ranks(comm);

  auto n_vtx = VertexSize_U<I4>(z);
  auto n_ghost_node = 0;
  if (cgns::has_node(z,"GridCoordinates/FSDM#n_ghost")) {
    n_ghost_node = get_node_value_by_matching<I4>(z,"GridCoordinates/FSDM#n_ghost")[0];
  }
  I4 n_owned_vtx = n_vtx - n_ghost_node;
  auto vtx_distri = distribution_from_dsizes(n_owned_vtx, comm);
  std_e::buffer_vector<I4> vtx_distri_mem(n_rank+1);
  std::copy(begin(vtx_distri),end(vtx_distri),begin(vtx_distri_mem));
  tree vtx_dist = new_DataArray("NODE",std::move(vtx_distri_mem));
  emplace_child(fsdm_dist_node,std::move(vtx_dist));

  auto elt_sections = get_children_by_label(z,"Elements_t");
  for (tree& elt_section : elt_sections) {
    auto elt_range = ElementRange<I4>(elt_section);
    I4 n_owned_elt = elt_range[1] - elt_range[0] + 1;

    auto elt_distri = distribution_from_dsizes(n_owned_elt, comm);
    std_e::buffer_vector<I4> elt_distri_mem(n_rank+1);
    std::copy(begin(elt_distri),end(elt_distri),begin(elt_distri_mem));

    I4 elt_type = ElementType<I4>(elt_section);
    auto elt_name = cgns::to_string((ElementType_t)elt_type);
    tree elt_dist = new_DataArray(elt_name,std::move(elt_distri_mem));

    emplace_child(fsdm_dist_node,std::move(elt_dist));
  }

  emplace_child(b,std::move(fsdm_dist_node));
}

} // cgns
