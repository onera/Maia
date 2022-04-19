#pragma once

#include <string>

namespace maia {
  const std::string this_file_path = __FILE__;
  const auto mesh_dir = this_file_path.substr(0, this_file_path.rfind("/"))+"/../../share/meshes/";
}
