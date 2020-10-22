#pragma once

#include <string>
#include "std_e/logging/time_logger.hpp"

[[nodiscard]] auto
maia_time_log(const std::string& msg) -> std_e::time_logger;
