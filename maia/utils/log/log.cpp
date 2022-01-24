#include "maia/utils/log/log.hpp"

auto
maia_time_log(const std::string& msg) -> std_e::time_logger {
  return std_e::time_logger(&std_e::get_logger("maia"),msg);
}

auto
maia_perf_log_lvl_0(const std::string& msg) -> std_e::time_and_mem_logger {
  return std_e::time_and_mem_logger(&std_e::get_logger("maia perf level 0"),msg);
}
auto
maia_perf_log_lvl_1(const std::string& msg) -> std_e::time_logger {
  return std_e::time_logger(&std_e::get_logger("maia perf level 1"),"  "+msg);
}
auto
maia_perf_log_lvl_2(const std::string& msg) -> std_e::time_logger {
  return std_e::time_logger(&std_e::get_logger("maia perf level 2"),"    "+msg);
}
