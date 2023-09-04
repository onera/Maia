#pragma once


#include "std_e/data_structure/table.hpp"
#include "std_e/parallel/mpi.hpp"
#include "maia/utils/parallel/distribution.hpp"
#include "maia/algo/part/cgns_registry/GenerateGlobalNumberingFromPaths.hpp"
#include <iostream>
// TODO unit tests
// TODO replace int by PDM_g_id_t ?

template<class T> auto // TODO RENAME
create_sorted_id_table(std::vector<T> entities, MPI_Comm comm) -> std_e::table<PDM_g_num_t,T> {
  std::vector<PDM_g_num_t> integer_ids = generate_global_numbering(entities,comm);
  std_e::table<PDM_g_num_t,T> id_table(std::move(integer_ids),std::move(entities));
  std_e::sort(id_table);
  return id_table;
}

inline auto
generate_distribution(const std::vector<PDM_g_num_t>& sorted_local_ids, MPI_Comm comm) -> distribution_vector<PDM_g_num_t> {
  // precondition: Union(sorted_local_ids) over comm == [0,global_nb_elts)
  PDM_g_num_t local_max_id = 0;
  if (sorted_local_ids.size() > 0) {
    local_max_id = sorted_local_ids.back();
  }
  PDM_g_num_t global_nb_elts = std_e::max_global(local_max_id,comm);
  return uniform_distribution(std_e::n_rank(comm),global_nb_elts+1);
}


// Generate a global numbering and distribution from entities
template<class T>
class distributed_registry { // TODO RENAME partitionned_registry
  // Class invariants
  //   - entities and ids are both sorted by increasing ids
  //   - Union(ids) on comm == [0,global_nb_entities)
  //   - Sum(nb_entities()) on comm <= global_nb_entities (entities can be on more than one rank)
  public:
    distributed_registry() = default;
    distributed_registry(std::vector<T> entities, MPI_Comm comm)
      : id_table(create_sorted_id_table(std::move(entities),comm))
      , distrib(generate_distribution(ids(),comm))
    {}

    int nb_entities() const {
      return id_table.size();
    }
    const auto& ids() const {
      return std_e::range<0>(id_table);
    }
    const auto& entities() const {
      return std_e::range<1>(id_table);
    }
    const auto& distribution() const {
      return distrib;
    }

    auto
    find_entity_from_id(PDM_g_num_t id) const -> const T& {
      std::cout << __PRETTY_FUNCTION__ << " ----> " << id << std::endl;
      return find_associate(id_table,id);
    }
    auto
    find_id_from_entity(const T& e) const -> PDM_g_num_t {
      // std::cout << __PRETTY_FUNCTION__ << std::endl;
      return find_associate(id_table,e);
    }
  private:
    std_e::table<PDM_g_num_t,T> id_table;
    distribution_vector<PDM_g_num_t> distrib;
};
