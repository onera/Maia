#include "maia/transform/fsdm_distribution.hpp"
#include "cpp_cgns/sids/sids.hpp"
#include "cpp_cgns/array_utils.hpp"
#include "maia/utils/parallel/distribution.hpp"
#include "std_e/log.hpp" // TODO


namespace cgns {

auto add_fsdm_distribution(tree& b, factory F, MPI_Comm comm) -> void {
  auto zs = get_children_by_label(b,"Zone_t");
  if (zs.size()!=1) {
    throw cgns_exception("add_fsdm_distribution (as FSDM) expects only one zone per process");
  }
  tree& z = zs[0];

  auto fsdm_dist_node = F.newUserDefinedData("FSDM_elt_distributions");

  int n_rank = std_e::nb_ranks(comm);

  auto n_vtx = VertexSize_U<I4>(z);
  auto n_ghost_node = get_node_value_by_matching<I4>(z,"GridCoordinates/FSDM#n_ghost")[0];
  I4 n_owned_vtx = n_vtx - n_ghost_node;
  auto vtx_distri = distribution_from_dsizes(n_owned_vtx, comm);
  auto vtx_distri_mem = make_cgns_vector<I4>(n_rank+1,F.alloc());
  std::copy(begin(vtx_distri),end(vtx_distri),begin(vtx_distri_mem));
  tree vtx_dist = F.newDataArray("NODE",{"I4",{n_rank+1},vtx_distri_mem.data()});
  emplace_child(fsdm_dist_node,std::move(vtx_dist));

  auto elt_sections = get_children_by_label(z,"Elements_t");
  for (tree& elt_section : elt_sections) {
      auto elt_range = ElementRange<I4>(elt_section);
      I4 n_owned_elt = elt_range[1] - elt_range[0] + 1;

      auto elt_distri = distribution_from_dsizes(n_owned_elt, comm);
      auto elt_distri_mem = make_cgns_vector<I4>(n_rank+1,F.alloc());
      std::copy(begin(elt_distri),end(elt_distri),begin(elt_distri_mem));

      I4 elt_type = ElementType<I4>(elt_section);
      auto elt_name = cgns::to_string((ElementType_t)elt_type);
      tree elt_dist = F.newDataArray(elt_name,{"I4",{n_rank+1},elt_distri_mem.data()});

      emplace_child(fsdm_dist_node,std::move(elt_dist));
  }

  emplace_child(b,std::move(fsdm_dist_node));
}

} // cgns
