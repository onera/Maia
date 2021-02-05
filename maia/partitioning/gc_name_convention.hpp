#pragma once

#include <string>
#include "std_e/future/contract.hpp"
#include "std_e/utils/string.hpp"

namespace maia {


inline auto
proc_of_zone(const std::string& zone_name) -> int {
  auto tokens = std_e::split(zone_name,'.');
  // Maia convention: zone_name == "name.P{0}.N{1}"
  // 0: zone proc
  // 1: partition on this proc
  int n_tok = tokens.size();
  auto tok = tokens[n_tok-2];
  STD_E_ASSERT(tok[0] == 'P');
  auto proc_str = tok.substr(1);
  return std::stoi(proc_str);
}

inline auto
proc_and_opp_proc_of_grid_connectivity(const std::string& grid_connectivity_name) -> std::pair<int,int> {
  auto tokens = std_e::split(grid_connectivity_name,'.');
  // Maia convention: grid_connectivity_name == "JN.P{0}.N{1}.LT.P{2}.N{3}"
  // 0: current zone proc
  // 1: partition on this proc
  // 2: opposite zone proc
  // 3: partition on opposite proc
  STD_E_ASSERT(tokens[0] == "JN");
  STD_E_ASSERT(tokens[3] == "LT");
  auto tok = tokens[1];
  STD_E_ASSERT(tok[0] == 'P');
  auto proc_str = tok.substr(1);
  auto tok_opp = tokens[4];
  STD_E_ASSERT(tok_opp[0] == 'P');
  auto opp_proc_str = tok_opp.substr(1);
  return std::make_pair(
    std::stoi(proc_str),
    std::stoi(opp_proc_str)
  );
}

inline auto
opp_proc_of_grid_connectivity(const std::string& grid_connectivity_name) -> int {
  return proc_and_opp_proc_of_grid_connectivity(grid_connectivity_name).second;
}


} // maia
