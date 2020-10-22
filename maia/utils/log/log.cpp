#include "maia/utils/log/log.hpp"

auto
maia_time_log(const std::string& msg) -> std_e::time_logger {
  return std_e::time_logger(&std_e::get_logger("maia"),msg);
}
