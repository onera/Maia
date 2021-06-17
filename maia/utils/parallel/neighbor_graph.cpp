#include "maia/utils/parallel/neighbor_graph.hpp"
#include "cpp_cgns/node_manip.hpp"
#include "cpp_cgns/tree_manip.hpp"
#include "std_e/utils/vector.hpp"
#include "std_e/parallel/mpi.hpp"
#include "std_e/utils/to_string.hpp"
#include "maia/utils/parallel/exchange/spread_then_collect.hpp"


namespace cgns {

auto
name_of_zones(const tree& b) -> std::vector<std::string> {
  STD_E_ASSERT(b.label=="CGNSBase_t");
  auto zones = get_children_by_label(b,"Zone_t");

  std::vector<std::string> z_names;
  for (const tree& z : zones) {
    z_names.push_back(z.name);
  }

  return z_names;
}

auto
name_of_mentionned_zones(const tree& b) -> std::vector<std::string> {
  STD_E_ASSERT(b.label=="CGNSBase_t");
  auto zones = get_children_by_label(b,"Zone_t");

  std::vector<std::string> z_names;
  for (const tree& z : zones) {
    z_names.push_back(z.name);
    if (has_child_of_name(z,"ZoneGridConnectivity")) {
      const tree& zgc = get_child_by_name(z,"ZoneGridConnectivity");
      auto gcs = get_children_by_label(zgc,"GridConnectivity_t");
      for (const tree& gc : gcs) {
        std::string opp_zone_name = to_string(value(gc));
        z_names.push_back(opp_zone_name);
      }
    }
  }

  std_e::sort_unique(z_names);
  return z_names;
}

auto
create_connectivity_infos(tree& b) -> std::vector<connectivity_info> {
  STD_E_ASSERT(b.label=="CGNSBase_t");
  auto zones = get_children_by_label(b,"Zone_t");

  std::vector<connectivity_info> cis;
  for (tree& z : zones) {
    if (has_child_of_name(z,"ZoneGridConnectivity")) {
      tree& zgc = get_child_by_name(z,"ZoneGridConnectivity");
      auto gcs = get_children_by_label(zgc,"GridConnectivity_t");
      for (tree& gc : gcs) {
        std::string opp_zone_name = to_string(value(gc));
        cis.push_back({z.name,opp_zone_name,&gc});
      }
    }
  }

  return cis;
}

auto
paths_of_all_mentionned_zones(const tree& b) -> cgns_paths {
  auto all_z_names = std_e::concatenate(name_of_zones(b),name_of_mentionned_zones(b));
  std_e::sort_unique(all_z_names);

  cgns_paths z_paths;
  for (const auto& z_name : all_z_names) {
    z_paths.push_back("/"+b.name+"/"+z_name);
  }
  return z_paths;
}

auto
compute_zone_infos(const tree& b, MPI_Comm comm) -> zone_infos {
  auto paths = paths_of_all_mentionned_zones(b);
  label_registry zone_reg(paths,comm);

  auto owned_zone_names = name_of_zones(b);
  int nb_owned_zones = owned_zone_names.size();
  std::vector<PDM_g_num_t> owned_zone_ids(nb_owned_zones);
  for (int i=0; i<nb_owned_zones; ++i) {
    owned_zone_ids[i] = get_global_id_from_path(zone_reg,"/"+b.name+"/"+owned_zone_names[i]); // TODO cgns_registry starts at 1
  }

  auto neighbor_zone_names = name_of_mentionned_zones(b);
  int nb_neighbor_zones = neighbor_zone_names.size();
  std::vector<PDM_g_num_t> neighbor_zone_ids_long(nb_neighbor_zones);
  for (int i=0; i<nb_neighbor_zones; ++i) {
    neighbor_zone_ids_long[i] = get_global_id_from_path(zone_reg,"/"+b.name+"/"+neighbor_zone_names[i]); // TODO cgns_registry starts at 1
  }

  std::vector<PDM_g_num_t> proc_of_owned_zones(nb_owned_zones,std_e::rank(comm));

  std_e::interval_vector<PDM_g_num_t> zone_reg_long(begin(zone_reg.distribution()), end(zone_reg.distribution()));
  auto proc_of_neighbor_zones_long = spread_then_collect(
    comm, zone_reg_long,
    owned_zone_ids, proc_of_owned_zones,
    neighbor_zone_ids_long
  );

  std::vector<int> proc_of_neighbor_zones(begin(proc_of_neighbor_zones_long), end(proc_of_neighbor_zones_long));
  std::vector<int> neighbor_zone_ids(begin(neighbor_zone_ids_long), end(neighbor_zone_ids_long));


  return {std::move(neighbor_zone_names),std::move(neighbor_zone_ids),std::move(proc_of_neighbor_zones)};
}


auto
donor_zones_ranks(const zone_infos& zis, const std::vector<connectivity_info>& cis) -> std::vector<int> {
  std::vector<int> ranks = std_e::transform(cis,[&zis](const auto& ci){ return find_donor_proc(ci,zis); });
  std_e::sort_unique(ranks);
  return ranks;
}


} // cgns
