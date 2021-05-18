#pragma once

#include <string>
#include "std_e/utils/enum.hpp"

namespace CGNS {

//Enum section
//############

//The strings defined below are type names used for node labels
//#############################################################

// Types as strings
// -----------------
STD_E_ENUM(Label,
  CGNSTree_t,
  CGNSBase_t,
  Zone_t,
  ZoneType_t,
  GridCoordinates_t,
  GridLocation_t,
  ZoneBC_t,
  BC_t,
  BCData_t,
  BCDataSet_t,
  ZoneGridConnectivity_t,
  GridConnectivity1to1_t,
  GridConnectivity_t,
  Family_t,
  FamilyName_t,
  AdditionalFamilyName_t,
  AdditionalExponents_t,
  AdditionalUnits_t,
  ArbitraryGridMotion_t,
  Area_t,
  AverageInterface_t,
  Axisymmetry_t,
  BCProperty_t,
  BCTypeSimple_t,
  BCTypeCompound_ts,
  BaseIterativeData_t,
  CGNSLibraryVersion_t,
  ChemicalKineticsModel_t,
  ConvergenceHistory_t,
  DataArray_t,
  DataClass_t,
  DataConversion_t,
  Descriptor_t,
  DimensionalExponents_t,
  DimensionalUnits_t,
  DiscreteData_t,
  Elements_t,
  FamilyBC_t,
  FamilyBCDataSet_t,
  FlowEquationSet_t,
  FlowSolution_t,
  GasModel_t,
  GasModelType_t,
  GeometryEntity_t,
  GeometryFile_t,
  GeometryFormat_t,
  GeometryReference_t,
  GoverningEquations_t,
  Gravity_t,
  GridConnectivityProperty_t,
  GridConnectivityType_t,
  IndexArray_t,
  IndexRange_t,
  IntegralData_t,
  InwardNormalList_t,
  Ordinal_t,
  OversetHoles_t,
  Periodic_t,
  ReferenceState_t,
  RigidGridMotion_t,
  Rind_t,
  RotatingCoordinates_t,
  SimulationType_t,
  ThermalConductivityModel_t,
  ThermalRelaxationModel_t,
  TurbulenceClosure_t,
  TurbulenceModel_t,
  UserDefinedData_t,
  ViscosityModel_t,
  ViscosityModelType_t,
  WallFunction_t,
  ZoneIterativeData_t,
  ZoneSubRegion_t,
  UserDefined_t,
  BulkRegionFamily_t,
  BndConditionFamily_t,
  BndConnectionFamily_t,
  Invalid_t
);

constexpr int nb_cgns_labels = std_e::enum_size<Label>;

//  Units
// ======

// Mass
// -----
STD_E_ENUM_CLASS(MassUnits,
  Null,
  UserDefined,
  Kilogram,
  Gram,
  Slug,
  PoundMass,
  maxMassUnits
);

// Length
// -------
STD_E_ENUM_CLASS(LengthUnits,
  Null,
  UserDefined,
  Meter,
  Centimeter,
  Millimeter,
  Foot,
  Inch,
  maxLengthUnits
);

// Time
// ----
STD_E_ENUM_CLASS(TimeUnits,
  Null,
  UserDefined,
  Second,
  maxTimeUnits
);

// Temperature
// ------------
STD_E_ENUM_CLASS(TemperatureUnits,
  Null,
  UserDefined,
  Kelvin,
  Celsius,
  Rankine,
  Fahrenheit,
  maxTemperatureUnits
);

// Angle
// ------
STD_E_ENUM_CLASS(AngleUnits,
  Null,
  UserDefined,
  Degree,
  Radian,
  maxAngleUnits
);

// ElectricCurrent
// ----------------
STD_E_ENUM_CLASS(ElectricCurrentUnits,
  Null,
  UserDefined,
  Ampere,
  Abampere,
  Statampere,
  Edison,
  auCurrent,
  maxElectricCurrentUnits
);

// SubstanceAmount
// ----------------
STD_E_ENUM_CLASS(SubstanceAmountUnits,
  Null,
  UserDefined,
  Mole,
  Entities,
  StandardCubicFoot,
  StandardCubicMeter,
  maxSubstanceAmountUnits
);

// LuminousIntensity
// ------------------
STD_E_ENUM_CLASS(LuminousIntensityUnits,
  Null,
  UserDefined,
  Candela,
  Candle,
  Carcel,
  Hefner,
  Violle,
  maxLuminousIntensityUnits
);


//  Class
// ======

// Data Class
// -----------
STD_E_ENUM_CLASS(DataClass,
  Null,
  UserDefined,
  Dimensional,
  NormalizedByDimensional,
  NormalizedByUnknownDimensional,
  NondimensionalParameter,
  DimensionlessConstant,
  maxDataClass
);


//  Values
// =======

// GridLocation
// ------------
STD_E_ENUM_CLASS(GridLocation,
  Null,
  UserDefined,
  Vertex,
  CellCenter,
  FaceCenter,
  IFaceCenter,
  JFaceCenter,
  KFaceCenter,
  EdgeCenter,
  maxGridLocation
);

// ChemicalKineticsModel
// ---------------------
STD_E_ENUM_CLASS(ChemicalKineticsModel,
  Null,
  UserDefined,
  Frozen,
  ChemicalEquilibCurveFit,
  ChemicalEquilibMinimization,
  ChemicalNonequilib,
  maxChemicalKineticsModel
);

// EMConductivityModel
// -------------------
STD_E_ENUM_CLASS(EMConductivityModel,
  Null,
  UserDefined,
  Constant,
  Frozen,
  Equilibrium_LinRessler,
  Chemistry_LinRessler,
  maxEMConductivityModel
);

// EMElectricFieldModel
// --------------------
STD_E_ENUM_CLASS(EMElectricFieldModel,
  Null,
  UserDefined,
  Voltage,
  Interpolated,
  Constant,
  Frozen,
  maxEMElectricFieldModel
);

// EMMagneticFieldModel
// --------------------
STD_E_ENUM_CLASS(EMMagneticFieldModel,
  Null,
  UserDefined,
  Interpolated,
  Constant,
  Frozen,
  maxEMMagneticFieldModel
);

// GasModel
// --------
STD_E_ENUM_CLASS(GasModel,
  Null,
  UserDefined,
  Ideal,
  VanderWaals,
  CaloricallyPerfect,
  ThermallyPerfect,
  ConstantDensity,
  RedlichKwong,
  maxGasModel
);

// ThermalConductivityModel
// ------------------------
STD_E_ENUM_CLASS(ThermalConductivityModel,
  Null,
  UserDefined,
  PowerLaw,
  SutherlandLaw,
  ConstantPrandtl,
  maxThermalConductivityModel
);

// ThermalRelaxationModel
// ----------------------
STD_E_ENUM_CLASS(ThermalRelaxationModel,
  Null,
  UserDefined,
  Frozen,
  ThermalEquilib,
  ThermalNonequilib,
  maxThermalRelaxationModel
);

// TurbulentClosure
// ----------------
STD_E_ENUM_CLASS(TurbulentClosure,
  Null,
  UserDefined,
  EddyViscosity,
  ReynoldsStress,
  ReynoldsStressAlgebraic,
  maxTurbulentClosure
);

// TurbulenceModel
// ---------------
STD_E_ENUM_CLASS(TurbulenceModel,
  Null,
  UserDefined,
  Algebraic_BaldwinLomax,
  Algebraic_CebeciSmith,
  HalfEquation_JohnsonKing,
  OneEquation_BaldwinBarth,
  OneEquation_SpalartAllmaras,
  TwoEquation_JonesLaunder,
  TwoEquation_MenterSST,
  TwoEquation_Wilcox,
  maxTurbulenceModel
);

// TransitionModel
// ---------------
STD_E_ENUM_CLASS(TransitionModel,
  Null,
  UserDefined,
  TwoEquation_LangtryMenter,
  maxTransitionModel
);

// ViscosityModel
// --------------
STD_E_ENUM_CLASS(ViscosityModel,
  Null,
  UserDefined,
  Constant,
  PowerLaw,
  SutherlandLaw,
  maxViscosityModel
);


//  Types
// ======

// BCData Types
// ------------
STD_E_ENUM_CLASS(BCDataType,
  Null,
  UserDefined,
  Dirichlet,
  Neumann,
  maxBCDataType
);

// Grid Connectivity Types
// ------------------------
STD_E_ENUM_CLASS(GridConnectivityType,
  Null,
  UserDefined,
  Overset,
  Abutting,
  Abutting1to1,
  maxGridConnectivityType
);


// Periodic Types
// ------------------
STD_E_ENUM_CLASS(PeriodicType,
  Translation,
  Rotation,
  maxPeriodicType
);

// Point Set Types
// ----------------
STD_E_ENUM_CLASS(PointSetType,
  Null,
  UserDefined,
  PointList,
  PointListDonor,
  PointRange,
  PointRangeDonor,
  ElementRange,
  ElementList,
  CellListDonor,
  maxPointSetType
);

// Governing Equations and Physical Models Types
// ----------------------------------------------
STD_E_ENUM_CLASS(GoverningEquationsType,
  Null,
  UserDefined,
  FullPotential,
  Euler,
  NSLaminar,
  NSTurbulent,
  NSLaminarIncompressible,
  NSTurbulentIncompressible,
  maxGoverningEquationsType
);

// Model Types
// -----------
STD_E_ENUM_CLASS(ModelType,
  Null,
  UserDefined,
  Ideal, VanderWaals,
  Constant,
  PowerLaw, SutherlandLaw,
  ConstantPrandtl,
  EddyViscosity, ReynoldsStress, ReynoldsStressAlgebraic,
  Algebraic_BaldwinLomax, Algebraic_CebeciSmith,
  HalfEquation_JohnsonKing, OneEquation_BaldwinBarth,
  OneEquation_SpalartAllmaras, TwoEquation_JonesLaunder,
  TwoEquation_MenterSST, TwoEquation_Wilcox,
  CaloricallyPerfect, ThermallyPerfect,
  ConstantDensity, RedlichKwong,
  Frozen, ThermalEquilib, ThermalNonequilib,
  ChemicalEquilibCurveFit, ChemicalEquilibMinimization,
  ChemicalNonequilib,
  EMElectricField, EMMagneticField, EMConductivity,
  Voltage, Interpolated, Equilibrium_LinRessler, Chemistry_LinRessler,
  maxModelType
);

// GasModel Types
// --------------
STD_E_ENUM_CLASS(GasModelType,
  Null,
  UserDefined,
  IdealGasConstant,
  SpecificHeatRatio,
  SpecificHeatVolume,
  SpecificHeatPressure,
  maxGasModelType
);

// ViscosityModel Types
// --------------------
STD_E_ENUM_CLASS(ViscosityModelType,
  Null,
  UserDefined,
  PowerLawExponent,
  SutherlandLawConstant,
  TemperatureReference,
  ViscosityMolecularReference,
  maxViscosityModelType
);

// Boundary Condition Types
// -------------------------
STD_E_ENUM_CLASS(BCType,
  Null,
  UserDefined,
  BCAxisymmetricWedge,
  BCDegenerateLine,
  BCDegeneratePoint,
  BCDirichlet,
  BCExtrapolate,
  BCFarfield,
  BCGeneral,
  BCInflow,
  BCInflowSubsonic,
  BCInflowSupersonic,
  BCNeumann,
  BCOutflow,
  BCOutflowSubsonic,
  BCOutflowSupersonic,
  BCSymmetryPlane,
  BCSymmetryPolar,
  BCTunnelInflow,
  BCTunnelOutflow,
  BCWall,
  BCWallInviscid,
  BCWallViscous,
  BCWallViscousHeatFlux,
  BCWallViscousIsothermal,
  FamilySpecified,
  maxBCType
);

// Data types : Can not add data types and stay forward compatible
// ----------------------------------------------------------------
STD_E_ENUM_CLASS(DataType,
  Null,
  UserDefined,
  Integer,
  RealSingle,
  RealDouble,
  Character,
  LongInteger,
  ComplexSingle,
  ComplexDouble,
  maxDataType
);

// Element Types
// -------------
STD_E_ENUM_CLASS(ElementType,
  Null,
  UserDefined,
  NODE,
  BAR_2,
  BAR_3,
  TRI_3,
  TRI_6,
  QUAD_4,
  QUAD_8,
  QUAD_9,
  TETRA_4,
  TETRA_10,
  PYRA_5,
  PYRA_14,
  PENTA_6,
  PENTA_15,
  PENTA_18,
  HEXA_8,
  HEXA_20,
  HEXA_27,
  MIXED,
  PYRA_13,
  NGON_n,
  NFACE_n,
  BAR_4,
  TRI_9,
  TRI_10,
  QUAD_12,
  QUAD_16,
  TETRA_16,
  TETRA_20,
  PYRA_21,
  PYRA_29,
  PYRA_30,
  PENTA_24,
  PENTA_38,
  PENTA_40,
  HEXA_32,
  HEXA_56,
  HEXA_64,
  BAR_5,
  TRI_12,
  TRI_15,
  QUAD_P4_16,
  QUAD_25,
  TETRA_22,
  TETRA_34,
  TETRA_35,
  PYRA_P4_29,
  PYRA_50,
  PYRA_55,
  PENTA_33,
  PENTA_66,
  PENTA_75,
  HEXA_44,
  HEXA_98,
  HEXA_125,
  maxElementType
);

// Zone Types
// ----------
STD_E_ENUM_CLASS(ZoneType,
  Null,
  UserDefined,
  Structured,
  Unstructured,
  maxZoneType
);

// Rigid Grid Motion Types
// -----------------------
STD_E_ENUM_CLASS(RigidGridMotionType,
  Null,
  UserDefined,
  ConstantRate,
  VariableRate,
  maxRigidGridMotionType
);

// Arbitrary Grid Motion Types
// ---------------------------
STD_E_ENUM_CLASS(ArbitraryGridMotionType,
  Null,
  UserDefined,
  NonDeformingGrid,
  DeformingGrid,
  maxArbitraryGridMotionType
);

// Simulation Types
// ----------------
STD_E_ENUM_CLASS(SimulationType,
  Null,
  UserDefined,
  TimeAccurate,
  NonTimeAccurate,
  maxSimulationType
);

// BC Property Types
// -----------------
STD_E_ENUM_CLASS(WallFunctionType,
  Null,
  UserDefined,
  Generic,
  maxWallFunctionType
);

// Average Interface Types
// -----------------------
STD_E_ENUM_CLASS(AverageInterfaceType,
  Null,
  UserDefined,
  AverageAll,
  AverageCircumferential,
  AverageRadial,
  AverageI,
  AverageJ,
  AverageK,
  maxAverageInterfaceType
);

namespace Name {

//The strings defined below are node names or node name patterns
//##############################################################

// Coordinate system
// -----------------
extern const char* GridCoordinates;
extern const char* CoordinateNames;
extern const char* CoordinateX;
extern const char* CoordinateY;
extern const char* CoordinateZ;
extern const char* CoordinateR;
extern const char* CoordinateTheta;
extern const char* CoordinatePhi;
extern const char* CoordinateNormal;
extern const char* CoordinateTangential;
extern const char* CoordinateXi;
extern const char* CoordinateEta;
extern const char* CoordinateZeta;
extern const char* CoordinateTransform;
extern const char* InterpolantsDonor;
extern const char* ElementConnectivity;
extern const char* ParentData;
extern const char* ParentElements;
extern const char* ParentElementsPosition;
extern const char* ElementSizeBoundary;

// FlowSolution Quantities
// -----------------------
// Patterns
extern const char* VectorX_p;
extern const char* VectorY_p;
extern const char* VectorZ_p;
extern const char* VectorTheta_p;
extern const char* VectorPhi_p;
extern const char* VectorMagnitude_p;
extern const char* VectorNormal_p;
extern const char* VectorTangential_p;
extern const char* Potential;
extern const char* StreamFunction;
extern const char* Density;
extern const char* Pressure;
extern const char* Temperature;
extern const char* EnergyInternal;
extern const char* Enthalpy;
extern const char* Entropy;
extern const char* EntropyApprox;
extern const char* DensityStagnation;
extern const char* PressureStagnation;
extern const char* TemperatureStagnation;
extern const char* EnergyStagnation;
extern const char* EnthalpyStagnation;
extern const char* EnergyStagnationDensity;
extern const char* VelocityX;
extern const char* VelocityY;
extern const char* VelocityZ;
extern const char* VelocityR;
extern const char* VelocityTheta;
extern const char* VelocityPhi;
extern const char* VelocityMagnitude;
extern const char* VelocityNormal;
extern const char* VelocityTangential;
extern const char* VelocitySound;
extern const char* VelocitySoundStagnation;
extern const char* MomentumX;
extern const char* MomentumY;
extern const char* MomentumZ;
extern const char* MomentumMagnitude;
extern const char* RotatingVelocityX;
extern const char* RotatingVelocityY;
extern const char* RotatingVelocityZ;
extern const char* RotatingMomentumX;
extern const char* RotatingMomentumY;
extern const char* RotatingMomentumZ;
extern const char* RotatingVelocityMagnitude;
extern const char* RotatingPressureStagnation;
extern const char* RotatingEnergyStagnation;
extern const char* RotatingEnergyStagnationDensity;
extern const char* RotatingEnthalpyStagnation;
extern const char* EnergyKinetic;
extern const char* PressureDynamic;
extern const char* SoundIntensityDB;
extern const char* SoundIntensity;
extern const char* VorticityX;
extern const char* VorticityY;
extern const char* VorticityZ;
extern const char* VorticityMagnitude;
extern const char* SkinFrictionX;
extern const char* SkinFrictionY;
extern const char* SkinFrictionZ;
extern const char* SkinFrictionMagnitude;
extern const char* VelocityAngleX;
extern const char* VelocityAngleY;
extern const char* VelocityAngleZ;
extern const char* VelocityUnitVectorX;
extern const char* VelocityUnitVectorY;
extern const char* VelocityUnitVectorZ;
extern const char* MassFlow;
extern const char* ViscosityKinematic;
extern const char* ViscosityMolecular;
extern const char* ViscosityEddyDynamic;
extern const char* ViscosityEddy;
extern const char* ThermalConductivity;
extern const char* ThermalConductivityReference;
extern const char* SpecificHeatPressure;
extern const char* SpecificHeatVolume;
extern const char* ReynoldsStressXX;
extern const char* ReynoldsStressXY;
extern const char* ReynoldsStressXZ;
extern const char* ReynoldsStressYY;
extern const char* ReynoldsStressYZ;
extern const char* ReynoldsStressZZ;
extern const char* LengthReference;
extern const char* MolecularWeight;
extern const char* MolecularWeight_p;
extern const char* HeatOfFormation;
extern const char* HeatOfFormation_p;
extern const char* FuelAirRatio;
extern const char* ReferenceTemperatureHOF;
extern const char* MassFraction;
extern const char* MassFraction_p;
extern const char* LaminarViscosity;
extern const char* LaminarViscosity_p;
extern const char* ThermalConductivity_p;
extern const char* EnthalpyEnergyRatio;
extern const char* CompressibilityFactor;
extern const char* VibrationalElectronEnergy;
extern const char* VibrationalElectronTemperature;
extern const char* SpeciesDensity;
extern const char* SpeciesDensity_p;
extern const char* MoleFraction;
extern const char* MoleFraction_p;
extern const char* ElectricFieldX;
extern const char* ElectricFieldY;
extern const char* ElectricFieldZ;
extern const char* MagneticFieldX;
extern const char* MagneticFieldY;
extern const char* MagneticFieldZ;
extern const char* CurrentDensityX;
extern const char* CurrentDensityY;
extern const char* CurrentDensityZ;
extern const char* LorentzForceX;
extern const char* LorentzForceY;
extern const char* LorentzForceZ;
extern const char* ElectricConductivity;
extern const char* JouleHeating;

// Typical Turbulence Models
// --------------------------
extern const char* TurbulentDistance;
extern const char* TurbulentEnergyKinetic;
extern const char* TurbulentDissipation;
extern const char* TurbulentDissipationRate;
extern const char* TurbulentBBReynolds;
extern const char* TurbulentSANuTilde;
extern const char* TurbulentDistanceIndex;
extern const char* TurbulentEnergyKineticDensity;
extern const char* TurbulentDissipationDensity;
extern const char* TurbulentSANuTildeDensity;

// Nondimensional Parameters
// --------------------------
extern const char* Mach;
extern const char* Mach_Velocity;
extern const char* Mach_VelocitySound;
extern const char* Reynolds;
extern const char* Reynolds_Velocity;
extern const char* Reynolds_Length;
extern const char* Reynolds_ViscosityKinematic;
extern const char* Prandtl;
extern const char* Prandtl_ThermalConductivity;
extern const char* Prandtl_ViscosityMolecular;
extern const char* Prandtl_SpecificHeatPressure;
extern const char* PrandtlTurbulent;
extern const char* CoefPressure;
extern const char* CoefSkinFrictionX;
extern const char* CoefSkinFrictionY;
extern const char* CoefSkinFrictionZ;
extern const char* Coef_PressureDynamic;
extern const char* Coef_PressureReference;

// Characteristics and Riemann invariant
// --------------------------------------
extern const char* Vorticity;
extern const char* Acoustic;
extern const char* RiemannInvariantPlus;
extern const char* RiemannInvariantMinus;
extern const char* CharacteristicEntropy;
extern const char* CharacteristicVorticity1;
extern const char* CharacteristicVorticity2;
extern const char* CharacteristicAcousticPlus;
extern const char* CharacteristicAcousticMinus;

// Forces and Moments
// -------------------
extern const char* ForceX;
extern const char* ForceY;
extern const char* ForceZ;
extern const char* ForceR;
extern const char* ForceTheta;
extern const char* ForcePhi;
extern const char* Lift;
extern const char* Drag;
extern const char* MomentX;
extern const char* MomentY;
extern const char* MomentZ;
extern const char* MomentR;
extern const char* MomentTheta;
extern const char* MomentPhi;
extern const char* MomentXi;
extern const char* MomentEta;
extern const char* MomentZeta;
extern const char* Moment_CenterX;
extern const char* Moment_CenterY;
extern const char* Moment_CenterZ;
extern const char* CoefLift;
extern const char* CoefDrag;
extern const char* CoefMomentX;
extern const char* CoefMomentY;
extern const char* CoefMomentZ;
extern const char* CoefMomentR;
extern const char* CoefMomentTheta;
extern const char* CoefMomentPhi;
extern const char* CoefMomentXi;
extern const char* CoefMomentEta;
extern const char* CoefMomentZeta;
extern const char* Coef_Area;
extern const char* Coef_Length;

// Time dependent flow
// --------------------
extern const char* TimeValues;
extern const char* IterationValues;
extern const char* NumberOfZones;
extern const char* NumberOfFamilies;
extern const char* NumberOfSteps;
extern const char* DataConversion;
extern const char* ZonePointers;
extern const char* FamilyPointers;
extern const char* RigidGridMotionPointers;
extern const char* ArbitraryGridMotionPointers;
extern const char* GridCoordinatesPointers;
extern const char* FlowSolutionPointers;
extern const char* ZoneGridConnectivityPointers;
extern const char* ZoneSubRegionPointers;
extern const char* OriginLocation;
extern const char* RigidRotationAngle;
extern const char* RigidVelocity;
extern const char* RigidRotationRate;
extern const char* GridVelocityX;
extern const char* GridVelocityY;
extern const char* GridVelocityZ;
extern const char* GridVelocityR;
extern const char* GridVelocityTheta;
extern const char* GridVelocityPhi;
extern const char* GridVelocityXi;
extern const char* GridVelocityEta;
extern const char* GridVelocityZeta;

// Miscellanous
// -------------
extern const char* CGNSLibraryVersion;
extern const char* CellDimension;
extern const char* IndexDimension;
extern const char* PhysicalDimension;
extern const char* VertexSize;
extern const char* CellSize;
extern const char* VertexSizeBoundary;
extern const char* ElementsSize;
extern const char* ZoneDonorName;
extern const char* BCRegionName;
extern const char* GridConnectivityRegionName;
extern const char* SurfaceArea;
extern const char* RegionName;
extern const char* Axisymmetry;
extern const char* AxisymmetryReferencePoint;
extern const char* AxisymmetryAxisVector;
extern const char* AxisymmetryAngle;
extern const char* ZoneConvergenceHistory;
extern const char* GlobalConvergenceHistory;
extern const char* NormDefinitions;
extern const char* DimensionalExponents;
extern const char* DiscreteData;
extern const char* FamilyBC;
extern const char* FamilyName;
extern const char* AdditionalFamilyName;
extern const char* Family;
extern const char* FlowEquationSet;
extern const char* GasModel;
extern const char* GeometryReference;
extern const char* Gravity;
extern const char* GravityVector;
extern const char* GridConnectivityProperty;
extern const char* InwardNormalList;
extern const char* InwardNormalIndex;
extern const char* Ordinal;
extern const char* Transform;
extern const char* OversetHoles;
extern const char* Periodic;
extern const char* ReferenceState;
extern const char* RigidGridMotion;
extern const char* Rind;
extern const char* RotatingCoordinates;
extern const char* RotationRateVector;
extern const char* GoverningEquations;
extern const char* BCTypeSimple;
extern const char* BCTypeCompound;
extern const char* ElementRangeList;

} /// End of Name namespace

} /// End of CGNS namespace
