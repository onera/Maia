#pragma once

#include <string>
#include "std_e/logging/time_logger.hpp"
#include "std_e/logging/time_and_mem_logger.hpp"

// TODO DEL (use maia_perf_log_lvl_0)
[[nodiscard]] auto
maia_time_log(const std::string& msg) -> std_e::time_logger;


// NOTE: maia_perf_log_lvl_0 (contrary to _lvl_1/2)
//   - only prints out to rank 0
//   - also gives the memory delta
auto maia_perf_log_lvl_0(const std::string& msg) -> std_e::time_and_mem_logger;

auto maia_perf_log_lvl_1(const std::string& msg) -> std_e::time_logger;
auto maia_perf_log_lvl_2(const std::string& msg) -> std_e::time_logger;
