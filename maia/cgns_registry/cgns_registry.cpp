#include <iostream>
#include <cassert>

#include "maia/cgns_registry/cgns_keywords.hpp"
#include "maia/cgns_registry/cgns_registry.hpp"


// ===========================================================================
cgns_registry::cgns_registry(const cgns_paths_by_label& paths_by_label, MPI_Comm& comm) {
  for (int i=0; i < CGNS::nb_cgns_labels; ++i){
    // std::cout << to_string(static_cast<CGNS::Label::kind>(i)) <<std::endl;
    registries_by_label[i] = label_registry(paths_by_label[i],comm);
  }
}

// ===========================================================================
std::string to_string(const label_registry& reg) {
  std::string s;
  int n_entry = reg.nb_entities();
  for(int i = 0; i < n_entry; ++i ){
    s += "\t CGNSPath : " + reg.entities()[i] + " | global_id = " + std::to_string(reg.ids()[i]) + "\n";
  }
  int n_rank_p1 = reg.distribution().size();
  s += "Distrib : ";
  for(int i = 0; i < n_rank_p1; ++i){
    s += std::to_string(reg.distribution()[i]) + " ";
  }
  s += "\n";
  return s;
}

// ===========================================================================
std::string to_string(const cgns_registry& cgns_reg)
{
  std::string s;
  for (int i = 0; i < CGNS::nb_cgns_labels; ++i){
    int n_entry = cgns_reg.at(i).nb_entities();
    if (n_entry > 0) {
      s += " -------------------------------------------- \n";
      s += " ### CGNSLabel : " +to_string(static_cast<CGNS::Label::kind>(i))+"\n";
      s += to_string(cgns_reg.at(i));
    }
  }
  return s;
}

// ===========================================================================
int get_global_id_from_path(const label_registry& reg, std::string path) {
  return reg.find_id_from_entity(path);
}
std::string get_path_from_global_id(const label_registry& reg, int g_id) {
  return reg.find_entity_from_id(g_id);
}

// ===========================================================================
int get_global_id_from_path_and_type(const cgns_registry& cgns_reg, std::string path, CGNS::Label::kind label){
  return get_global_id_from_path(cgns_reg.at(label),path);
}

// ===========================================================================
std::string get_path_from_global_id_and_type(const cgns_registry& cgns_reg, int g_id, CGNS::Label::kind label){
  return get_path_from_global_id(cgns_reg.at(label),g_id);
}

// ===========================================================================
int get_global_id_from_path_and_type(const cgns_registry& cgns_reg, std::string path, std::string cgns_label_str){
  auto label = std_e::to_enum<CGNS::Label::kind>(cgns_label_str);
  return get_global_id_from_path_and_type(cgns_reg,path,label);
}

// ===========================================================================
std::string get_path_from_global_id_and_type(const cgns_registry& cgns_reg, int g_id, std::string cgns_label_str){
  auto label = std_e::to_enum<CGNS::Label::kind>(cgns_label_str);
  return get_path_from_global_id_and_type(cgns_reg,g_id,label);
}
