from enum import Enum

##Enum section
##############

##The strings defined below are type names used for node labels
###############################################################

## Types as strings
## -----------------
Label = Enum('Label', [
  'CGNSTree_t',
  'CGNSBase_t',
  'Zone_t',
  'ZoneType_t',
  'GridCoordinates_t',
  'GridLocation_t',
  'ZoneBC_t',
  'BC_t',
  'BCData_t',
  'BCDataSet_t',
  'ZoneGridConnectivity_t',
  'GridConnectivity1to1_t',
  'GridConnectivity_t',
  'Family_t',
  'FamilyName_t',
  'AdditionalFamilyName_t',
  'AdditionalExponents_t',
  'AdditionalUnits_t',
  'ArbitraryGridMotion_t',
  'Area_t',
  'AverageInterface_t',
  'AverageInterfaceType_t',
  'Axisymmetry_t',
  'BCProperty_t',
  'BCTypeSimple_t',
  'BCTypeCompound_ts',
  'BaseIterativeData_t',
  'CGNSLibraryVersion_t',
  'ChemicalKineticsModel_t',
  'ConvergenceHistory_t',
  'DataArray_t',
  'DataClass_t',
  'DataConversion_t',
  'Descriptor_t',
  'DimensionalExponents_t',
  'DimensionalUnits_t',
  'DiscreteData_t',
  'Elements_t',
  'FamilyBC_t',
  'FamilyBCDataSet_t',
  'FlowEquationSet_t',
  'FlowSolution_t',
  'GasModel_t',
  'GasModelType_t',
  'GeometryEntity_t',
  'GeometryFile_t',
  'GeometryFormat_t',
  'GeometryReference_t',
  'GoverningEquations_t',
  'Gravity_t',
  'GridConnectivityProperty_t',
  'GridConnectivityType_t',
  'IndexArray_t',
  'IndexRange_t',
  'IntegralData_t',
  'InwardNormalList_t',
  'Ordinal_t',
  'OversetHoles_t',
  'ParticleBreakupModel_t',
  'ParticleBreakupModelType_t',
  'ParticleCollisionModel_t',
  'ParticleCollisionModelType_t',
  'ParticleCoordinates_t',
  'ParticleForceModel_t',
  'ParticleForceModelType_t',
  'ParticleGoverningEquations_t',
  'ParticleGoverningEquationsType_t',
  'ParticleEquationSet_t',
  'ParticleIterativeData_t',
  'ParticleModelType_t',
  'ParticlePhaseChangeModel_t',
  'ParticlePhaseChangeModelType_t',
  'ParticleSolution_t',
  'ParticleWallInteractionModel_t',
  'ParticleWallInteractionModelType_t',
  'ParticleZone_t',
  'Periodic_t',
  'ReferenceState_t',
  'RigidGridMotion_t',
  'Rind_t',
  'RotatingCoordinates_t',
  'SimulationType_t',
  'ThermalConductivityModel_t',
  'ThermalRelaxationModel_t',
  'TurbulenceClosure_t',
  'TurbulenceModel_t',
  'UserDefinedData_t',
  'ViscosityModel_t',
  'ViscosityModelType_t',
  'WallFunction_t',
  'ZoneIterativeData_t',
  'ZoneSubRegion_t',
  'UserDefined_t',
  'BulkRegionFamily_t',
  'BndConditionFamily_t',
  'BndConnectionFamily_t',
  'Invalid_t'
  ], start=0)

nb_cgns_labels = len(Label)

##  Units
## ======

## Mass
## -----
MassUnits = Enum('MassUnits', [
  'Null',
  'UserDefined',
  'Kilogram',
  'Gram',
  'Slug',
  'PoundMass',
  'maxMassUnits'
  ], start=0)

## Length
## -------
LengthUnits = Enum('LengthUnits', [
  'Null',
  'UserDefined',
  'Meter',
  'Centimeter',
  'Millimeter',
  'Foot',
  'Inch',
  'maxLengthUnits'
  ], start=0)

## Time
## ----
TimeUnits = Enum('TimeUnits', [
  'Null',
  'UserDefined',
  'Second',
  'maxTimeUnits'
  ], start=0)

## Temperature
## ------------
TemperatureUnits = Enum('TemperatureUnits', [
  'Null',
  'UserDefined',
  'Kelvin',
  'Celsius',
  'Rankine',
  'Fahrenheit',
  'maxTemperatureUnits'
  ], start=0)

## Angle
## ------
AngleUnits = Enum('AngleUnits', [
  'Null',
  'UserDefined',
  'Degree',
  'Radian',
  'maxAngleUnits'
  ], start=0)

## ElectricCurrent
## ----------------
ElectricCurrentUnits = Enum('ElectricCurrentUnits', [
  'Null',
  'UserDefined',
  'Ampere',
  'Abampere',
  'Statampere',
  'Edison',
  'auCurrent',
  'maxElectricCurrentUnits'
  ], start=0)

## SubstanceAmount
## ----------------
SubstanceAmountUnits = Enum('SubstanceAmountUnits', [
  'Null',
  'UserDefined',
  'Mole',
  'Entities',
  'StandardCubicFoot',
  'StandardCubicMeter',
  'maxSubstanceAmountUnits'
  ], start=0)

## LuminousIntensity
## ------------------
LuminousIntensityUnits = Enum('LuminousIntensityUnits', [
  'Null',
  'UserDefined',
  'Candela',
  'Candle',
  'Carcel',
  'Hefner',
  'Violle',
  'maxLuminousIntensityUnits'
  ], start=0)


##  Class
## ======

## Data Class
## -----------
DataClass = Enum('DataClass', [
  'Null',
  'UserDefined',
  'Dimensional',
  'NormalizedByDimensional',
  'NormalizedByUnknownDimensional',
  'NondimensionalParameter',
  'DimensionlessConstant',
  'maxDataClass'
  ], start=0)


##  Values
## =======

## GridLocation
## ------------
GridLocation = Enum('GridLocation', [
  'Null',
  'UserDefined',
  'Vertex',
  'CellCenter',
  'FaceCenter',
  'IFaceCenter',
  'JFaceCenter',
  'KFaceCenter',
  'EdgeCenter',
  ## Only for SoNICS
  'DualFacetCenter',
  'maxGridLocation'
  ], start=0)

## ChemicalKineticsModel
## ---------------------
ChemicalKineticsModel = Enum('ChemicalKineticsModel', [
  'Null',
  'UserDefined',
  'Frozen',
  'ChemicalEquilibCurveFit',
  'ChemicalEquilibMinimization',
  'ChemicalNonequilib',
  'maxChemicalKineticsModel'
  ], start=0)

## EMConductivityModel
## -------------------
EMConductivityModel = Enum('EMConductivityModel', [
  'Null',
  'UserDefined',
  'Constant',
  'Frozen',
  'Equilibrium_LinRessler',
  'Chemistry_LinRessler',
  'maxEMConductivityModel'
  ], start=0)

## EMElectricFieldModel
## --------------------
EMElectricFieldModel = Enum('EMElectricFieldModel', [
  'Null',
  'UserDefined',
  'Voltage',
  'Interpolated',
  'Constant',
  'Frozen',
  'maxEMElectricFieldModel'
  ], start=0)

## EMMagneticFieldModel
## --------------------
EMMagneticFieldModel = Enum('EMMagneticFieldModel', [
  'Null',
  'UserDefined',
  'Interpolated',
  'Constant',
  'Frozen',
  'maxEMMagneticFieldModel'
  ], start=0)

## GasModel
## --------
GasModel = Enum('GasModel', [
  'Null',
  'UserDefined',
  'Ideal',
  'VanderWaals',
  'CaloricallyPerfect',
  'ThermallyPerfect',
  'ConstantDensity',
  'RedlichKwong',
  'maxGasModel'
  ], start=0)

## ThermalConductivityModel
## ------------------------
ThermalConductivityModel = Enum('ThermalConductivityModel', [
  'Null',
  'UserDefined',
  'PowerLaw',
  'SutherlandLaw',
  'ConstantPrandtl',
  'maxThermalConductivityModel'
  ], start=0)

## ThermalRelaxationModel
## ----------------------
ThermalRelaxationModel = Enum('ThermalRelaxationModel', [
  'Null',
  'UserDefined',
  'Frozen',
  'ThermalEquilib',
  'ThermalNonequilib',
  'maxThermalRelaxationModel'
  ], start=0)

## TurbulentClosure
## ----------------
TurbulentClosure = Enum('TurbulentClosure', [
  'Null',
  'UserDefined',
  'EddyViscosity',
  'ReynoldsStress',
  'ReynoldsStressAlgebraic',
  'maxTurbulentClosure'
  ], start=0)

## TurbulenceModel
## ---------------
TurbulenceModel = Enum('TurbulenceModel', [
  'Null',
  'UserDefined',
  'Algebraic_BaldwinLomax',
  'Algebraic_CebeciSmith',
  'HalfEquation_JohnsonKing',
  'OneEquation_BaldwinBarth',
  'OneEquation_SpalartAllmaras',
  'TwoEquation_JonesLaunder',
  'TwoEquation_MenterSST',
  'TwoEquation_Wilcox',
  'maxTurbulenceModel'
  ], start=0)

## TransitionModel
## ---------------
TransitionModel = Enum('TransitionModel', [
  'Null',
  'UserDefined',
  'TwoEquation_LangtryMenter',
  'maxTransitionModel'
  ], start=0)

## ViscosityModel
## --------------
ViscosityModel = Enum('ViscosityModel', [
  'Null',
  'UserDefined',
  'Constant',
  'PowerLaw',
  'SutherlandLaw',
  'maxViscosityModel'
  ], start=0)


##  Types
## ======

## BCData Types
## ------------
BCDataType = Enum('BCDataType', [
  'Null',
  'UserDefined',
  'Dirichlet',
  'Neumann',
  'maxBCDataType'
  ], start=0)

## Grid Connectivity Types
## ------------------------
GridConnectivityType = Enum('GridConnectivityType', [
  'Null',
  'UserDefined',
  'Overset',
  'Abutting',
  'Abutting1to1',
  'maxGridConnectivityType'
  ], start=0)


## Periodic Types
## ------------------
PeriodicType = Enum('PeriodicType', [
  'Translation',
  'Rotation',
  'maxPeriodicType'
  ], start=0)

## Point Set Types
## ----------------
PointSetType = Enum('PointSetType', [
  'Null',
  'UserDefined',
  'PointList',
  'PointListDonor',
  'PointRange',
  'PointRangeDonor',
  'ElementRange',
  'ElementList',
  'CellListDonor',
  'maxPointSetType'
  ], start=0)

## Governing Equations and Physical Models Types
## ----------------------------------------------
GoverningEquationsType = Enum('GoverningEquationsType', [
  'Null',
  'UserDefined',
  'FullPotential',
  'Euler',
  'NSLaminar',
  'NSTurbulent',
  'NSLaminarIncompressible',
  'NSTurbulentIncompressible',
  'maxGoverningEquationsType'
  ], start=0)

## Model Types
## -----------
ModelType = Enum('ModelType', [
  'Null',
  'UserDefined',
  'Ideal', 'VanderWaals',
  'Constant',
  'PowerLaw', 'SutherlandLaw',
  'ConstantPrandtl',
  'EddyViscosity', 'ReynoldsStress', 'ReynoldsStressAlgebraic',
  'Algebraic_BaldwinLomax', 'Algebraic_CebeciSmith',
  'HalfEquation_JohnsonKing', 'OneEquation_BaldwinBarth',
  'OneEquation_SpalartAllmaras', 'TwoEquation_JonesLaunder',
  'TwoEquation_MenterSST', 'TwoEquation_Wilcox',
  'CaloricallyPerfect', 'ThermallyPerfect',
  'ConstantDensity', 'RedlichKwong',
  'Frozen', 'ThermalEquilib', 'ThermalNonequilib',
  'ChemicalEquilibCurveFit', 'ChemicalEquilibMinimization',
  'ChemicalNonequilib',
  'EMElectricField', 'EMMagneticField', 'EMConductivity',
  'Voltage', 'Interpolated', 'Equilibrium_LinRessler', 'Chemistry_LinRessler',
  'maxModelType'
  ], start=0)

## GasModel Types
## --------------
GasModelType = Enum('GasModelType', [
  'Null',
  'UserDefined',
  'IdealGasConstant',
  'SpecificHeatRatio',
  'SpecificHeatVolume',
  'SpecificHeatPressure',
  'maxGasModelType'
  ], start=0)

## ViscosityModel Types
## --------------------
ViscosityModelType = Enum('ViscosityModelType', [
  'Null',
  'UserDefined',
  'PowerLawExponent',
  'SutherlandLawConstant',
  'TemperatureReference',
  'ViscosityMolecularReference',
  'maxViscosityModelType'
  ], start=0)

## Boundary Condition Types
## -------------------------
BCType = Enum('BCType', [
  'Null',
  'UserDefined',
  'BCAxisymmetricWedge',
  'BCDegenerateLine',
  'BCDegeneratePoint',
  'BCDirichlet',
  'BCExtrapolate',
  'BCFarfield',
  'BCGeneral',
  'BCInflow',
  'BCInflowSubsonic',
  'BCInflowSupersonic',
  'BCNeumann',
  'BCOutflow',
  'BCOutflowSubsonic',
  'BCOutflowSupersonic',
  'BCSymmetryPlane',
  'BCSymmetryPolar',
  'BCTunnelInflow',
  'BCTunnelOutflow',
  'BCWall',
  'BCWallInviscid',
  'BCWallViscous',
  'BCWallViscousHeatFlux',
  'BCWallViscousIsothermal',
  'FamilySpecified',
  'maxBCType'
  ], start=0)

## Data types : Can not add data types and stay forward compatible
## ----------------------------------------------------------------
DataType = Enum('DataType', [
  'Null',
  'UserDefined',
  'Integer',
  'RealSingle',
  'RealDouble',
  'Character',
  'LongInteger',
  'ComplexSingle',
  'ComplexDouble',
  'maxDataType'
  ], start=0)

## Element Types
## -------------
ElementType = Enum('ElementType', [
  'Null',
  'UserDefined',
  'NODE',
  'BAR_2',
  'BAR_3',
  'TRI_3',
  'TRI_6',
  'QUAD_4',
  'QUAD_8',
  'QUAD_9',
  'TETRA_4',
  'TETRA_10',
  'PYRA_5',
  'PYRA_14',
  'PENTA_6',
  'PENTA_15',
  'PENTA_18',
  'HEXA_8',
  'HEXA_20',
  'HEXA_27',
  'MIXED',
  'PYRA_13',
  'NGON_n',
  'NFACE_n',
  'BAR_4',
  'TRI_9',
  'TRI_10',
  'QUAD_12',
  'QUAD_16',
  'TETRA_16',
  'TETRA_20',
  'PYRA_21',
  'PYRA_29',
  'PYRA_30',
  'PENTA_24',
  'PENTA_38',
  'PENTA_40',
  'HEXA_32',
  'HEXA_56',
  'HEXA_64',
  'BAR_5',
  'TRI_12',
  'TRI_15',
  'QUAD_P4_16',
  'QUAD_25',
  'TETRA_22',
  'TETRA_34',
  'TETRA_35',
  'PYRA_P4_29',
  'PYRA_50',
  'PYRA_55',
  'PENTA_33',
  'PENTA_66',
  'PENTA_75',
  'HEXA_44',
  'HEXA_98',
  'HEXA_125',
  'maxElementType'
  ], start=0)

## Zone Types
## ----------
ZoneType = Enum('ZoneType', [
  'Null',
  'UserDefined',
  'Structured',
  'Unstructured',
  'maxZoneType'
  ], start=0)

## Rigid Grid Motion Types
## -----------------------
RigidGridMotionType = Enum('RigidGridMotionType', [
  'Null',
  'UserDefined',
  'ConstantRate',
  'VariableRate',
  'maxRigidGridMotionType'
  ], start=0)

## Arbitrary Grid Motion Types
## ---------------------------
ArbitraryGridMotionType = Enum('ArbitraryGridMotionType', [
  'Null',
  'UserDefined',
  'NonDeformingGrid',
  'DeformingGrid',
  'maxArbitraryGridMotionType'
  ], start=0)

## Simulation Types
## ----------------
SimulationType = Enum('SimulationType', [
  'Null',
  'UserDefined',
  'TimeAccurate',
  'NonTimeAccurate',
  'maxSimulationType'
  ], start=0)

## BC Property Types
## -----------------
WallFunctionType = Enum('WallFunctionType', [
  'Null',
  'UserDefined',
  'Generic',
  'maxWallFunctionType'
  ], start=0)

## Average Interface Types
## -----------------------
AverageInterfaceType = Enum('AverageInterfaceType', [
  'Null',
  'UserDefined',
  'AverageAll',
  'AverageCircumferential',
  'AverageRadial',
  'AverageI',
  'AverageJ',
  'AverageK',
  'maxAverageInterfaceType'
  ], start=0)

