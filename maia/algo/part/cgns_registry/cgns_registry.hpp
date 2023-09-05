#pragma once

#include <array>
#include "pdm.h"

#include "maia/pytree/cgns_keywords/cgns_keywords.hpp"
#include "maia/algo/part/cgns_registry/distributed_registry.hpp"


using cgns_path = std::string;
using cgns_paths = std::vector<cgns_path>;
using label_registry = distributed_registry<cgns_path>;

using cgns_paths_by_label = std::array<cgns_paths,CGNS::nb_cgns_labels>;

inline
void add_path(cgns_paths_by_label& paths, const cgns_path& path, CGNS::Label label) {
  paths[label].push_back(std::move(path));
}
inline
void add_path(cgns_paths_by_label& paths, const std::string& path, const std::string& label_str) {
  auto label = std_e::to_enum<CGNS::Label>(label_str);
  add_path(paths,path,label);
}


// ===========================================================================
class cgns_registry {
  public:
    cgns_registry() = default;
    cgns_registry(const cgns_paths_by_label& paths, MPI_Comm comm);

    const label_registry& at(int label) const {
      return registries_by_label[label];
    }

    const std::vector<std::string>& paths(int label) const {
      return registries_by_label[label].entities();
    }

    const std::vector<PDM_g_num_t>& global_ids(int label) const {
      return registries_by_label[label].ids();
    }

    const distribution_vector<PDM_g_num_t>& distribution(int label) const {
      return registries_by_label[label].distribution();
    }
  private:
    std::array<label_registry,CGNS::nb_cgns_labels> registries_by_label;
};

// ===========================================================================
int get_global_id_from_path(const label_registry& reg, std::string path);
std::string get_path_from_global_id(const label_registry& reg, PDM_g_num_t g_id);

// ===========================================================================
int get_global_id_from_path_and_type(const cgns_registry& cgns_reg, std::string path, CGNS::Label label);
std::string get_path_from_global_id_and_type(const cgns_registry& cgns_reg, PDM_g_num_t g_id, CGNS::Label label);

int get_global_id_from_path_and_type(const cgns_registry& cgns_reg, std::string path, std::string cgns_label_str);
std::string get_path_from_global_id_and_type(const cgns_registry& cgns_reg, PDM_g_num_t g_id, std::string cgns_label_str);

// ===========================================================================
std::string to_string(const cgns_registry& cgns_reg);
