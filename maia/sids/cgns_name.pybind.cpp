#include <iostream>
#include <vector>
#include <algorithm>
#include <pybind11/pybind11.h>
#include "maia/sids/cgns_keywords.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cgns_name, m) {
  m.doc() = "cgns_keywords module for CGNS Name";

  // Name
  // ++++
  // Coordinate system
  // -----------------
  m.attr("GridCoordinates")        = CGNS::Name::GridCoordinates;
  m.attr("CoordinateNames")        = CGNS::Name::CoordinateNames;
  m.attr("CoordinateX")            = CGNS::Name::CoordinateX;
  m.attr("CoordinateY")            = CGNS::Name::CoordinateY;
  m.attr("CoordinateZ")            = CGNS::Name::CoordinateZ;
  m.attr("CoordinateR")            = CGNS::Name::CoordinateR;
  m.attr("CoordinateTheta")        = CGNS::Name::CoordinateTheta;
  m.attr("CoordinatePhi")          = CGNS::Name::CoordinatePhi;
  m.attr("CoordinateNormal")       = CGNS::Name::CoordinateNormal;
  m.attr("CoordinateTangential")   = CGNS::Name::CoordinateTangential;
  m.attr("CoordinateXi")           = CGNS::Name::CoordinateXi;
  m.attr("CoordinateEta")          = CGNS::Name::CoordinateEta;
  m.attr("CoordinateZeta")         = CGNS::Name::CoordinateZeta;
  m.attr("CoordinateTransform")    = CGNS::Name::CoordinateTransform;
  m.attr("InterpolantsDonor")      = CGNS::Name::InterpolantsDonor;
  m.attr("ElementConnectivity")    = CGNS::Name::ElementConnectivity;
  m.attr("ParentData")             = CGNS::Name::ParentData;
  m.attr("ParentElements")         = CGNS::Name::ParentElements;
  m.attr("ParentElementsPosition") = CGNS::Name::ParentElementsPosition;
  m.attr("ElementSizeBoundary")    = CGNS::Name::ElementSizeBoundary;

  // FlowSolution Quantities
  // -----------------------
  // Patterns
  m.attr("VectorX_p")                       = CGNS::Name::VectorX_p;
  m.attr("VectorY_p")                       = CGNS::Name::VectorY_p;
  m.attr("VectorZ_p")                       = CGNS::Name::VectorZ_p;
  m.attr("VectorTheta_p")                   = CGNS::Name::VectorTheta_p;
  m.attr("VectorPhi_p")                     = CGNS::Name::VectorPhi_p;
  m.attr("VectorMagnitude_p")               = CGNS::Name::VectorMagnitude_p;
  m.attr("VectorNormal_p")                  = CGNS::Name::VectorNormal_p;
  m.attr("VectorTangential_p")              = CGNS::Name::VectorTangential_p;
  m.attr("Potential")                       = CGNS::Name::Potential;
  m.attr("StreamFunction")                  = CGNS::Name::StreamFunction;
  m.attr("Density")                         = CGNS::Name::Density;
  m.attr("Pressure")                        = CGNS::Name::Pressure;
  m.attr("Temperature")                     = CGNS::Name::Temperature;
  m.attr("EnergyInternal")                  = CGNS::Name::EnergyInternal;
  m.attr("Enthalpy")                        = CGNS::Name::Enthalpy;
  m.attr("Entropy")                         = CGNS::Name::Entropy;
  m.attr("EntropyApprox")                   = CGNS::Name::EntropyApprox;
  m.attr("DensityStagnation")               = CGNS::Name::DensityStagnation;
  m.attr("PressureStagnation")              = CGNS::Name::PressureStagnation;
  m.attr("TemperatureStagnation")           = CGNS::Name::TemperatureStagnation;
  m.attr("EnergyStagnation")                = CGNS::Name::EnergyStagnation;
  m.attr("EnthalpyStagnation")              = CGNS::Name::EnthalpyStagnation;
  m.attr("EnergyStagnationDensity")         = CGNS::Name::EnergyStagnationDensity;
  m.attr("VelocityX")                       = CGNS::Name::VelocityX;
  m.attr("VelocityY")                       = CGNS::Name::VelocityY;
  m.attr("VelocityZ")                       = CGNS::Name::VelocityZ;
  m.attr("VelocityR")                       = CGNS::Name::VelocityR;
  m.attr("VelocityTheta")                   = CGNS::Name::VelocityTheta;
  m.attr("VelocityPhi")                     = CGNS::Name::VelocityPhi;
  m.attr("VelocityMagnitude")               = CGNS::Name::VelocityMagnitude;
  m.attr("VelocityNormal")                  = CGNS::Name::VelocityNormal;
  m.attr("VelocityTangential")              = CGNS::Name::VelocityTangential;
  m.attr("VelocitySound")                   = CGNS::Name::VelocitySound;
  m.attr("VelocitySoundStagnation")         = CGNS::Name::VelocitySoundStagnation;
  m.attr("MomentumX")                       = CGNS::Name::MomentumX;
  m.attr("MomentumY")                       = CGNS::Name::MomentumY;
  m.attr("MomentumZ")                       = CGNS::Name::MomentumZ;
  m.attr("MomentumMagnitude")               = CGNS::Name::MomentumMagnitude;
  m.attr("RotatingVelocityX")               = CGNS::Name::RotatingVelocityX;
  m.attr("RotatingVelocityY")               = CGNS::Name::RotatingVelocityY;
  m.attr("RotatingVelocityZ")               = CGNS::Name::RotatingVelocityZ;
  m.attr("RotatingMomentumX")               = CGNS::Name::RotatingMomentumX;
  m.attr("RotatingMomentumY")               = CGNS::Name::RotatingMomentumY;
  m.attr("RotatingMomentumZ")               = CGNS::Name::RotatingMomentumZ;
  m.attr("RotatingVelocityMagnitude")       = CGNS::Name::RotatingVelocityMagnitude;
  m.attr("RotatingPressureStagnation")      = CGNS::Name::RotatingPressureStagnation;
  m.attr("RotatingEnergyStagnation")        = CGNS::Name::RotatingEnergyStagnation;
  m.attr("RotatingEnergyStagnationDensity") = CGNS::Name::RotatingEnergyStagnationDensity;
  m.attr("RotatingEnthalpyStagnation")      = CGNS::Name::RotatingEnthalpyStagnation;
  m.attr("EnergyKinetic")                   = CGNS::Name::EnergyKinetic;
  m.attr("PressureDynamic")                 = CGNS::Name::PressureDynamic;
  m.attr("SoundIntensityDB")                = CGNS::Name::SoundIntensityDB;
  m.attr("SoundIntensity")                  = CGNS::Name::SoundIntensity;
  m.attr("VorticityX")                      = CGNS::Name::VorticityX;
  m.attr("VorticityY")                      = CGNS::Name::VorticityY;
  m.attr("VorticityZ")                      = CGNS::Name::VorticityZ;
  m.attr("VorticityMagnitude")              = CGNS::Name::VorticityMagnitude;
  m.attr("SkinFrictionX")                   = CGNS::Name::SkinFrictionX;
  m.attr("SkinFrictionY")                   = CGNS::Name::SkinFrictionY;
  m.attr("SkinFrictionZ")                   = CGNS::Name::SkinFrictionZ;
  m.attr("SkinFrictionMagnitude")           = CGNS::Name::SkinFrictionMagnitude;
  m.attr("VelocityAngleX")                  = CGNS::Name::VelocityAngleX;
  m.attr("VelocityAngleY")                  = CGNS::Name::VelocityAngleY;
  m.attr("VelocityAngleZ")                  = CGNS::Name::VelocityAngleZ;
  m.attr("VelocityUnitVectorX")             = CGNS::Name::VelocityUnitVectorX;
  m.attr("VelocityUnitVectorY")             = CGNS::Name::VelocityUnitVectorY;
  m.attr("VelocityUnitVectorZ")             = CGNS::Name::VelocityUnitVectorZ;
  m.attr("MassFlow")                        = CGNS::Name::MassFlow;
  m.attr("ViscosityKinematic")              = CGNS::Name::ViscosityKinematic;
  m.attr("ViscosityMolecular")              = CGNS::Name::ViscosityMolecular;
  m.attr("ViscosityEddyDynamic")            = CGNS::Name::ViscosityEddyDynamic;
  m.attr("ViscosityEddy")                   = CGNS::Name::ViscosityEddy;
  m.attr("ThermalConductivity")             = CGNS::Name::ThermalConductivity;
  m.attr("ThermalConductivityReference")    = CGNS::Name::ThermalConductivityReference;
  m.attr("SpecificHeatPressure")            = CGNS::Name::SpecificHeatPressure;
  m.attr("SpecificHeatVolume")              = CGNS::Name::SpecificHeatVolume;
  m.attr("ReynoldsStressXX")                = CGNS::Name::ReynoldsStressXX;
  m.attr("ReynoldsStressXY")                = CGNS::Name::ReynoldsStressXY;
  m.attr("ReynoldsStressXZ")                = CGNS::Name::ReynoldsStressXZ;
  m.attr("ReynoldsStressYY")                = CGNS::Name::ReynoldsStressYY;
  m.attr("ReynoldsStressYZ")                = CGNS::Name::ReynoldsStressYZ;
  m.attr("ReynoldsStressZZ")                = CGNS::Name::ReynoldsStressZZ;
  m.attr("LengthReference")                 = CGNS::Name::LengthReference;
  m.attr("MolecularWeight")                 = CGNS::Name::MolecularWeight;
  m.attr("MolecularWeight_p")               = CGNS::Name::MolecularWeight_p;
  m.attr("HeatOfFormation")                 = CGNS::Name::HeatOfFormation;
  m.attr("HeatOfFormation_p")               = CGNS::Name::HeatOfFormation_p;
  m.attr("FuelAirRatio")                    = CGNS::Name::FuelAirRatio;
  m.attr("ReferenceTemperatureHOF")         = CGNS::Name::ReferenceTemperatureHOF;
  m.attr("MassFraction")                    = CGNS::Name::MassFraction;
  m.attr("MassFraction_p")                  = CGNS::Name::MassFraction_p;
  m.attr("LaminarViscosity")                = CGNS::Name::LaminarViscosity;
  m.attr("LaminarViscosity_p")              = CGNS::Name::LaminarViscosity_p;
  m.attr("ThermalConductivity_p")           = CGNS::Name::ThermalConductivity_p;
  m.attr("EnthalpyEnergyRatio")             = CGNS::Name::EnthalpyEnergyRatio;
  m.attr("CompressibilityFactor")           = CGNS::Name::CompressibilityFactor;
  m.attr("VibrationalElectronEnergy")       = CGNS::Name::VibrationalElectronEnergy;
  m.attr("VibrationalElectronTemperature")  = CGNS::Name::VibrationalElectronTemperature;
  m.attr("SpeciesDensity")                  = CGNS::Name::SpeciesDensity;
  m.attr("SpeciesDensity_p")                = CGNS::Name::SpeciesDensity_p;
  m.attr("MoleFraction")                    = CGNS::Name::MoleFraction;
  m.attr("MoleFraction_p")                  = CGNS::Name::MoleFraction_p;
  m.attr("ElectricFieldX")                  = CGNS::Name::ElectricFieldX;
  m.attr("ElectricFieldY")                  = CGNS::Name::ElectricFieldY;
  m.attr("ElectricFieldZ")                  = CGNS::Name::ElectricFieldZ;
  m.attr("MagneticFieldX")                  = CGNS::Name::MagneticFieldX;
  m.attr("MagneticFieldY")                  = CGNS::Name::MagneticFieldY;
  m.attr("MagneticFieldZ")                  = CGNS::Name::MagneticFieldZ;
  m.attr("CurrentDensityX")                 = CGNS::Name::CurrentDensityX;
  m.attr("CurrentDensityY")                 = CGNS::Name::CurrentDensityY;
  m.attr("CurrentDensityZ")                 = CGNS::Name::CurrentDensityZ;
  m.attr("LorentzForceX")                   = CGNS::Name::LorentzForceX;
  m.attr("LorentzForceY")                   = CGNS::Name::LorentzForceY;
  m.attr("LorentzForceZ")                   = CGNS::Name::LorentzForceZ;
  m.attr("ElectricConductivity")            = CGNS::Name::ElectricConductivity;
  m.attr("JouleHeating")                    = CGNS::Name::JouleHeating;

  // Typical Turbulence Models
  // --------------------------
  m.attr("TurbulentDistance")               = CGNS::Name::TurbulentDistance;
  m.attr("TurbulentEnergyKinetic")          = CGNS::Name::TurbulentEnergyKinetic;
  m.attr("TurbulentDissipation")            = CGNS::Name::TurbulentDissipation;
  m.attr("TurbulentDissipationRate")        = CGNS::Name::TurbulentDissipationRate;
  m.attr("TurbulentBBReynolds")             = CGNS::Name::TurbulentBBReynolds;
  m.attr("TurbulentSANuTilde")              = CGNS::Name::TurbulentSANuTilde;
  m.attr("TurbulentDistanceIndex")          = CGNS::Name::TurbulentDistanceIndex;
  m.attr("TurbulentEnergyKineticDensity")   = CGNS::Name::TurbulentEnergyKineticDensity;
  m.attr("TurbulentDissipationDensity")     = CGNS::Name::TurbulentDissipationDensity;
  m.attr("TurbulentSANuTildeDensity")       = CGNS::Name::TurbulentSANuTildeDensity;

  // Nondimensional Parameters
  // --------------------------
  m.attr("Mach")                         = CGNS::Name::Mach;
  m.attr("Mach_Velocity")                = CGNS::Name::Mach_Velocity;
  m.attr("Mach_VelocitySound")           = CGNS::Name::Mach_VelocitySound;
  m.attr("Reynolds")                     = CGNS::Name::Reynolds;
  m.attr("Reynolds_Velocity")            = CGNS::Name::Reynolds_Velocity;
  m.attr("Reynolds_Length")              = CGNS::Name::Reynolds_Length;
  m.attr("Reynolds_ViscosityKinematic")  = CGNS::Name::Reynolds_ViscosityKinematic;
  m.attr("Prandtl")                      = CGNS::Name::Prandtl;
  m.attr("Prandtl_ThermalConductivity")  = CGNS::Name::Prandtl_ThermalConductivity;
  m.attr("Prandtl_ViscosityMolecular")   = CGNS::Name::Prandtl_ViscosityMolecular;
  m.attr("Prandtl_SpecificHeatPressure") = CGNS::Name::Prandtl_SpecificHeatPressure;
  m.attr("PrandtlTurbulent")             = CGNS::Name::PrandtlTurbulent;
  m.attr("CoefPressure")                 = CGNS::Name::CoefPressure;
  m.attr("CoefSkinFrictionX")            = CGNS::Name::CoefSkinFrictionX;
  m.attr("CoefSkinFrictionY")            = CGNS::Name::CoefSkinFrictionY;
  m.attr("CoefSkinFrictionZ")            = CGNS::Name::CoefSkinFrictionZ;
  m.attr("Coef_PressureDynamic")         = CGNS::Name::Coef_PressureDynamic;
  m.attr("Coef_PressureReference")       = CGNS::Name::Coef_PressureReference;

  // Characteristics and Riemann invariant
  // --------------------------------------
  m.attr("Vorticity")                    = CGNS::Name::Vorticity;
  m.attr("Acoustic")                     = CGNS::Name::Acoustic;
  m.attr("RiemannInvariantPlus")         = CGNS::Name::RiemannInvariantPlus;
  m.attr("RiemannInvariantMinus")        = CGNS::Name::RiemannInvariantMinus;
  m.attr("CharacteristicEntropy")        = CGNS::Name::CharacteristicEntropy;
  m.attr("CharacteristicVorticity1")     = CGNS::Name::CharacteristicVorticity1;
  m.attr("CharacteristicVorticity2")     = CGNS::Name::CharacteristicVorticity2;
  m.attr("CharacteristicAcousticPlus")   = CGNS::Name::CharacteristicAcousticPlus;
  m.attr("CharacteristicAcousticMinus")  = CGNS::Name::CharacteristicAcousticMinus;

  // Forces and Moments
  // -------------------
  m.attr("ForceX")          = CGNS::Name::ForceX;
  m.attr("ForceY")          = CGNS::Name::ForceY;
  m.attr("ForceZ")          = CGNS::Name::ForceZ;
  m.attr("ForceR")          = CGNS::Name::ForceR;
  m.attr("ForceTheta")      = CGNS::Name::ForceTheta;
  m.attr("ForcePhi")        = CGNS::Name::ForcePhi;
  m.attr("Lift")            = CGNS::Name::Lift;
  m.attr("Drag")            = CGNS::Name::Drag;
  m.attr("MomentX")         = CGNS::Name::MomentX;
  m.attr("MomentY")         = CGNS::Name::MomentY;
  m.attr("MomentZ")         = CGNS::Name::MomentZ;
  m.attr("MomentR")         = CGNS::Name::MomentR;
  m.attr("MomentTheta")     = CGNS::Name::MomentTheta;
  m.attr("MomentPhi")       = CGNS::Name::MomentPhi;
  m.attr("MomentXi")        = CGNS::Name::MomentXi;
  m.attr("MomentEta")       = CGNS::Name::MomentEta;
  m.attr("MomentZeta")      = CGNS::Name::MomentZeta;
  m.attr("Moment_CenterX")  = CGNS::Name::Moment_CenterX;
  m.attr("Moment_CenterY")  = CGNS::Name::Moment_CenterY;
  m.attr("Moment_CenterZ")  = CGNS::Name::Moment_CenterZ;
  m.attr("CoefLift")        = CGNS::Name::CoefLift;
  m.attr("CoefDrag")        = CGNS::Name::CoefDrag;
  m.attr("CoefMomentX")     = CGNS::Name::CoefMomentX;
  m.attr("CoefMomentY")     = CGNS::Name::CoefMomentY;
  m.attr("CoefMomentZ")     = CGNS::Name::CoefMomentZ;
  m.attr("CoefMomentR")     = CGNS::Name::CoefMomentR;
  m.attr("CoefMomentTheta") = CGNS::Name::CoefMomentTheta;
  m.attr("CoefMomentPhi")   = CGNS::Name::CoefMomentPhi;
  m.attr("CoefMomentXi")    = CGNS::Name::CoefMomentXi;
  m.attr("CoefMomentEta")   = CGNS::Name::CoefMomentEta;
  m.attr("CoefMomentZeta")  = CGNS::Name::CoefMomentZeta;
  m.attr("Coef_Area")       = CGNS::Name::Coef_Area;
  m.attr("Coef_Length")     = CGNS::Name::Coef_Length;

  // Time dependent flow
  // --------------------
  m.attr("TimeValues")                   = CGNS::Name::TimeValues;
  m.attr("IterationValues")              = CGNS::Name::IterationValues;
  m.attr("NumberOfZones")                = CGNS::Name::NumberOfZones;
  m.attr("NumberOfFamilies")             = CGNS::Name::NumberOfFamilies;
  m.attr("NumberOfSteps")                = CGNS::Name::NumberOfSteps;
  m.attr("DataConversion")               = CGNS::Name::DataConversion;
  m.attr("ZonePointers")                 = CGNS::Name::ZonePointers;
  m.attr("FamilyPointers")               = CGNS::Name::FamilyPointers;
  m.attr("RigidGridMotionPointers")      = CGNS::Name::RigidGridMotionPointers;
  m.attr("ArbitraryGridMotionPointers")  = CGNS::Name::ArbitraryGridMotionPointers;
  m.attr("GridCoordinatesPointers")      = CGNS::Name::GridCoordinatesPointers;
  m.attr("FlowSolutionPointers")         = CGNS::Name::FlowSolutionPointers;
  m.attr("ZoneGridConnectivityPointers") = CGNS::Name::ZoneGridConnectivityPointers;
  m.attr("ZoneSubRegionPointers")        = CGNS::Name::ZoneSubRegionPointers;
  m.attr("OriginLocation")               = CGNS::Name::OriginLocation;
  m.attr("RigidRotationAngle")           = CGNS::Name::RigidRotationAngle;
  m.attr("RigidVelocity")                = CGNS::Name::RigidVelocity;
  m.attr("RigidRotationRate")            = CGNS::Name::RigidRotationRate;
  m.attr("GridVelocityX")                = CGNS::Name::GridVelocityX;
  m.attr("GridVelocityY")                = CGNS::Name::GridVelocityY;
  m.attr("GridVelocityZ")                = CGNS::Name::GridVelocityZ;
  m.attr("GridVelocityR")                = CGNS::Name::GridVelocityR;
  m.attr("GridVelocityTheta")            = CGNS::Name::GridVelocityTheta;
  m.attr("GridVelocityPhi")              = CGNS::Name::GridVelocityPhi;
  m.attr("GridVelocityXi")               = CGNS::Name::GridVelocityXi;
  m.attr("GridVelocityEta")              = CGNS::Name::GridVelocityEta;
  m.attr("GridVelocityZeta")             = CGNS::Name::GridVelocityZeta;

  // Miscellanous
  // -------------
  m.attr("CGNSLibraryVersion")         = CGNS::Name::CGNSLibraryVersion;
  m.attr("CellDimension")              = CGNS::Name::CellDimension;
  m.attr("IndexDimension")             = CGNS::Name::IndexDimension;
  m.attr("PhysicalDimension")          = CGNS::Name::PhysicalDimension;
  m.attr("VertexSize")                 = CGNS::Name::VertexSize;
  m.attr("CellSize")                   = CGNS::Name::CellSize;
  m.attr("VertexSizeBoundary")         = CGNS::Name::VertexSizeBoundary;
  m.attr("ElementsSize")               = CGNS::Name::ElementsSize;
  m.attr("ZoneDonorName")              = CGNS::Name::ZoneDonorName;
  m.attr("BCRegionName")               = CGNS::Name::BCRegionName;
  m.attr("GridConnectivityRegionName") = CGNS::Name::GridConnectivityRegionName;
  m.attr("SurfaceArea")                = CGNS::Name::SurfaceArea;
  m.attr("RegionName")                 = CGNS::Name::RegionName;
  m.attr("Axisymmetry")                = CGNS::Name::Axisymmetry;
  m.attr("AxisymmetryReferencePoint")  = CGNS::Name::AxisymmetryReferencePoint;
  m.attr("AxisymmetryAxisVector")      = CGNS::Name::AxisymmetryAxisVector;
  m.attr("AxisymmetryAngle")           = CGNS::Name::AxisymmetryAngle;
  m.attr("ZoneConvergenceHistory")     = CGNS::Name::ZoneConvergenceHistory;
  m.attr("GlobalConvergenceHistory")   = CGNS::Name::GlobalConvergenceHistory;
  m.attr("NormDefinitions")            = CGNS::Name::NormDefinitions;
  m.attr("DimensionalExponents")       = CGNS::Name::DimensionalExponents;
  m.attr("DiscreteData")               = CGNS::Name::DiscreteData;
  m.attr("FamilyBC")                   = CGNS::Name::FamilyBC;
  m.attr("FamilyName")                 = CGNS::Name::FamilyName;
  m.attr("AdditionalFamilyName")       = CGNS::Name::AdditionalFamilyName;
  m.attr("Family")                     = CGNS::Name::Family;
  m.attr("FlowEquationSet")            = CGNS::Name::FlowEquationSet;
  m.attr("GasModel")                   = CGNS::Name::GasModel;
  m.attr("GeometryReference")          = CGNS::Name::GeometryReference;
  m.attr("Gravity")                    = CGNS::Name::Gravity;
  m.attr("GravityVector")              = CGNS::Name::GravityVector;
  m.attr("GridConnectivityProperty")   = CGNS::Name::GridConnectivityProperty;
  m.attr("InwardNormalList")           = CGNS::Name::InwardNormalList;
  m.attr("InwardNormalIndex")          = CGNS::Name::InwardNormalIndex;
  m.attr("Ordinal")                    = CGNS::Name::Ordinal;
  m.attr("Transform")                  = CGNS::Name::Transform;
  m.attr("OversetHoles")               = CGNS::Name::OversetHoles;
  m.attr("Periodic")                   = CGNS::Name::Periodic;
  m.attr("ReferenceState")             = CGNS::Name::ReferenceState;
  m.attr("RigidGridMotion")            = CGNS::Name::RigidGridMotion;
  m.attr("Rind")                       = CGNS::Name::Rind;
  m.attr("RotatingCoordinates")        = CGNS::Name::RotatingCoordinates;
  m.attr("RotationRateVector")         = CGNS::Name::RotationRateVector;
  m.attr("GoverningEquations")         = CGNS::Name::GoverningEquations;
  m.attr("BCTypeSimple")               = CGNS::Name::BCTypeSimple;
  m.attr("BCTypeCompound")             = CGNS::Name::BCTypeCompound;
  m.attr("ElementRangeList")           = CGNS::Name::ElementRangeList;
}
