#include <iostream>
#include <vector>
#include <algorithm>
#include <pybind11/pybind11.h>
#include "maia/sids/cgns_keywords.hpp"

namespace py = pybind11;

template <typename Type>
void pybind_auto_enum(py::enum_<Type>& t){
  int nb_enum = std_e::enum_size<Type>;
  for(int i = 0; i < nb_enum; ++i){
    Type e_id = static_cast<Type>(i);
    auto name = to_string(e_id);
    t.value(name.c_str(), e_id);
  }
}

PYBIND11_MODULE(cgns_keywords, m) {
  m.doc() = "cgns_keywords module for CGNS Label and associated Value";

  // Label
  // +++++
  auto enum_cgns_label = py::enum_<CGNS::Label>(m, "Label", py::arithmetic(), "CGNS Label");
  pybind_auto_enum(enum_cgns_label);
  m.attr("nb_cgns_labels") = std_e::enum_size<CGNS::Label>;

  // Value
  // +++++

  // Units
  // =====
  // Mass
  auto enum_mass_units = py::enum_<CGNS::MassUnits>(m, "MassUnits", py::arithmetic(), "CGNS Units for MassUnits");
  pybind_auto_enum(enum_mass_units);
  // Length
  auto enum_length_units = py::enum_<CGNS::LengthUnits>(m, "LengthUnits", py::arithmetic(), "CGNS Units for LengthUnits");
  pybind_auto_enum(enum_length_units);
  // Time
  auto enum_time_units = py::enum_<CGNS::TimeUnits>(m, "TimeUnits", py::arithmetic(), "CGNS Units for TimeUnits");
  pybind_auto_enum(enum_time_units);
  // Temperature
  auto enum_temperature_units = py::enum_<CGNS::TemperatureUnits>(m, "TemperatureUnits", py::arithmetic(), "CGNS Value for TemperatureUnits");
  pybind_auto_enum(enum_temperature_units);
  // Angle
  auto enum_angle_units = py::enum_<CGNS::AngleUnits>(m, "AngleUnits", py::arithmetic(), "CGNS Units for AngleUnits");
  pybind_auto_enum(enum_angle_units);
  // ElectricCurrent
  auto enum_electric_current_units = py::enum_<CGNS::ElectricCurrentUnits>(m, "ElectricCurrentUnits", py::arithmetic(), "CGNS Units for ElectricCurrentUnits");
  pybind_auto_enum(enum_electric_current_units);
  // SubstanceAmount
  auto enum_substance_amount_units = py::enum_<CGNS::SubstanceAmountUnits>(m, "SubstanceAmountUnits", py::arithmetic(), "CGNS Units for SubstanceAmountUnits");
  pybind_auto_enum(enum_substance_amount_units);
  // LuminousIntensity
  auto enum_luminous_intensity_units = py::enum_<CGNS::LuminousIntensityUnits>(m, "LuminousIntensityUnits", py::arithmetic(), "CGNS Units for LuminousIntensityUnits");
  pybind_auto_enum(enum_luminous_intensity_units);

  // Class
  // =====
  // Data Class
  auto enum_data_class = py::enum_<CGNS::DataClass>(m, "DataClass", py::arithmetic(), "CGNS Class for DataClass");
  pybind_auto_enum(enum_data_class);

  // Values
  // ======
  // GridLocation
  auto enum_grid_location = py::enum_<CGNS::GridLocation>(m, "GridLocation", py::arithmetic(), "CGNS Value for GridLocation");
  pybind_auto_enum(enum_grid_location);
  // ChemicalKineticsModel
  auto enum_chemical_kinetic_model = py::enum_<CGNS::ChemicalKineticsModel>(m, "ChemicalKineticsModel", py::arithmetic(), "CGNS Value for ChemicalKineticsModel");
  pybind_auto_enum(enum_chemical_kinetic_model);
  // EMConductivityModel
  auto enum_em_conductivity_model = py::enum_<CGNS::EMConductivityModel>(m, "EMConductivityModel", py::arithmetic(), "CGNS Value for EMConductivityModel");
  pybind_auto_enum(enum_em_conductivity_model);
  // EMElectricFieldModel
  auto enum_em_electric_field_model = py::enum_<CGNS::EMElectricFieldModel>(m, "EMElectricFieldModel", py::arithmetic(), "CGNS Value for EMElectricFieldModel");
  pybind_auto_enum(enum_em_electric_field_model);
  // GasModel
  auto enum_gas_model = py::enum_<CGNS::GasModel>(m, "GasModel", py::arithmetic(), "CGNS Value for GasModel");
  pybind_auto_enum(enum_gas_model);
  // ThermalConductivityModel
  auto enum_thermal_conductivity_model = py::enum_<CGNS::ThermalConductivityModel>(m, "ThermalConductivityModel", py::arithmetic(), "CGNS Value for ThermalConductivityModel");
  pybind_auto_enum(enum_thermal_conductivity_model);
  // ThermalRelaxationModel
  auto enum_thermal_relaxation_model = py::enum_<CGNS::ThermalRelaxationModel>(m, "ThermalRelaxationModel", py::arithmetic(), "CGNS Value for ThermalRelaxationModel");
  pybind_auto_enum(enum_thermal_relaxation_model);
  // TurbulentClosure
  auto enum_turbulent_closure = py::enum_<CGNS::TurbulentClosure>(m, "TurbulentClosure", py::arithmetic(), "CGNS Value for TurbulentClosure");
  pybind_auto_enum(enum_turbulent_closure);
  // TurbulenceModel
  auto enum_turbulent_model = py::enum_<CGNS::TurbulenceModel>(m, "TurbulenceModel", py::arithmetic(), "CGNS Value for TurbulenceModel");
  pybind_auto_enum(enum_turbulent_model);
  // TransitionModel
  auto enum_transition_model = py::enum_<CGNS::TransitionModel>(m, "TransitionModel", py::arithmetic(), "CGNS Value for TransitionModel");
  pybind_auto_enum(enum_transition_model);
  // ViscosityModel
  auto enum_viscosity_model = py::enum_<CGNS::ViscosityModel>(m, "ViscosityModel", py::arithmetic(), "CGNS Value for ViscosityModel");
  pybind_auto_enum(enum_viscosity_model);

  // Types
  // =====
  // BCData Types
  auto enum_bc_data_types = py::enum_<CGNS::BCDataType>(m, "BCDataType", py::arithmetic(), "CGNS Types for BCDataType");
  pybind_auto_enum(enum_bc_data_types);
  // GridConnectivity Types
  auto enum_grid_connectivity_type = py::enum_<CGNS::GridConnectivityType>(m, "GridConnectivityType", py::arithmetic(), "CGNS Types for GridConnectivityType");
  pybind_auto_enum(enum_grid_connectivity_type);
  // Periodic Types
  auto enum_periodic_type = py::enum_<CGNS::PeriodicType>(m, "PeriodicType", py::arithmetic(), "CGNS Types for PeriodicType");
  pybind_auto_enum(enum_periodic_type);
  // Point Set Types
  auto enum_point_set_type = py::enum_<CGNS::PointSetType>(m, "PointSetType", py::arithmetic(), "CGNS Types for PointSetType");
  pybind_auto_enum(enum_point_set_type);
  // Governing Equations and Physical Models Types
  auto enum_governing_equations_type = py::enum_<CGNS::GoverningEquationsType>(m, "GoverningEquationsType", py::arithmetic(), "CGNS Types for GoverningEquationsType");
  pybind_auto_enum(enum_governing_equations_type);
  // Model Types
  auto enum_model_type = py::enum_<CGNS::ModelType>(m, "ModelType", py::arithmetic(), "CGNS Types for ModelType");
  pybind_auto_enum(enum_model_type);
  // GasModel Types
  auto enum_gas_model_type = py::enum_<CGNS::GasModelType>(m, "GasModelType", py::arithmetic(), "CGNS Types for GasModelType");
  pybind_auto_enum(enum_gas_model_type);
  // ViscosityModel Types
  auto enum_viscosity_model_type = py::enum_<CGNS::ViscosityModelType>(m, "ViscosityModelType", py::arithmetic(), "CGNS Types for ViscosityModelType");
  pybind_auto_enum(enum_viscosity_model_type);
  // Boundary Condition Types
  auto enum_bc_type = py::enum_<CGNS::BCType>(m, "BCType", py::arithmetic(), "CGNS Types for BCType");
  pybind_auto_enum(enum_bc_type);
  // Data Types
  auto enum_data_type = py::enum_<CGNS::DataType>(m, "DataType", py::arithmetic(), "CGNS Types for DataType");
  pybind_auto_enum(enum_data_type);
  // Element Types
  auto enum_element_type = py::enum_<CGNS::ElementType>(m, "ElementType", py::arithmetic(), "CGNS Types for ElementType");
  pybind_auto_enum(enum_element_type);
  // Zone Types
  auto enum_zone_type = py::enum_<CGNS::ZoneType>(m, "ZoneType", py::arithmetic(), "CGNS Types for ZoneType");
  pybind_auto_enum(enum_zone_type);
  // Rigid Grid Motion Types
  auto enum_rigid_grid_motion_type = py::enum_<CGNS::RigidGridMotionType>(m, "RigidGridMotionType", py::arithmetic(), "CGNS Types for RigidGridMotionType");
  pybind_auto_enum(enum_rigid_grid_motion_type);
  // Arbitrary Grid Motion Types
  auto enum_arbitrary_grid_motion_type = py::enum_<CGNS::ArbitraryGridMotionType>(m, "ArbitraryGridMotionType", py::arithmetic(), "CGNS Types for ArbitraryGridMotionType");
  pybind_auto_enum(enum_arbitrary_grid_motion_type);
  // Simulation Types
  auto enum_simulation_type = py::enum_<CGNS::SimulationType>(m, "SimulationType", py::arithmetic(), "CGNS Types for SimulationType");
  pybind_auto_enum(enum_simulation_type);
  // BC Property Types
  auto enum_bc_property_type = py::enum_<CGNS::WallFunctionType>(m, "WallFunctionType", py::arithmetic(), "CGNS Types for WallFunctionType");
  pybind_auto_enum(enum_bc_property_type);
  // Average Interface Types
  auto enum_average_interface_type = py::enum_<CGNS::AverageInterfaceType>(m, "AverageInterfaceType", py::arithmetic(), "CGNS Types for AverageInterfaceType");
  pybind_auto_enum(enum_average_interface_type);

  // // Name
  // // ----
  // py::module::import("cmaia.cgns_registry.cgns_name");
}
