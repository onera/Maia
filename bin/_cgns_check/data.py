#Specific check for :
#AdditionalUnits_t
#DimensionalUnits
#IndexArray / IndexRange
# Rind : 2 x IdxDim
# ZoneSubRegion (value optional)
#DataArray


# Pas de dtype pour DataArray, IndexArray
# Pas de shape pour ['DataArray_t', 'IndexArray_t', 'IndexRange_t', 'Rind_t', 'Zone_t']

ArbitraryGridMotionType = {'ArbitraryGridMotionTypeNull', 'ArbitraryGridMotionTypeUserDefined',
                           'NonDeformingGrid', 'DeformingGrid'}
AreaType = {'AreaTypeNull', 'AreaTypeUserDefined', 'BleedArea', 'CaptureArea'}
AverageInterfaceType = {'AverageInterfaceTypeNull', 'AverageInterfaceTypeUserDefined', 'AverageAll',
                        'AverageCircumferential', 'AverageRadial', 'AverageI', 'AverageJ', 'AverageK'}
BCTypeSimple = {'BCTypeNull', 'BCTypeUserDefined', 'BCAxisymmetricWedge', 'BCDegenerateLine',
                'BCDegeneratePoint', 'BCDirichlet', 'BCExtrapolate', 'BCGeneral', 'BCInflowSubsonic',
                'BCInflowSupersonic', 'BCNeumann', 'BCOutflowSubsonic', 'BCOutflowSupersonic', 'BCSymmetryPlane',
                'BCSymmetryPolar', 'BCTunnelInflow', 'BCTunnelOutflow', 'BCWall', 'BCWallInviscid',
                'BCWallViscous', 'BCWallViscousHeatFlux', 'BCWallViscousIsothermal', 'FamilySpecified'}
BCTypeCompound = {'BCTypeNull', 'BCTypeUserDefined', 'BCInflow', 'BCOutflow', 'BCFarfield'}
BCType = BCTypeSimple | BCTypeCompound
ChemicalKineticsModelType = {'ModelTypeNull', 'ModelTypeUserDefined', 'Frozen', 'ChemicalEquilibCurveFit',
                             'ChemicalEquilibMinimization', 'ChemicalNonequilib'}
DataClass = {'DataClassNull', 'DataClassUserDefined', 'Dimensional', 'NormalizedByDimensional',
             'NormalizedByUnknownDimensional', 'NondimensionalParameter', 'DimensionlessConstant'}
EMConductivityModelType = {'ModelTypeNull', 'ModelTypeUserDefined', 'Constant', 'Frozen',
                           'Equilibrium_LinRessler', 'Chemistry_LinRessler'}
EMElectricFieldModelType = {'ModelTypeNull', 'ModelTypeUserDefined', 'Constant',
                            'Frozen', 'Interpolated', 'Voltage'}
EMMagneticFieldModelType = { 'ModelTypeNull', 'ModelTypeUserDefined', 'Constant', 'Frozen', 'Interpolated'}
GasModel = {'ModelTypeNull', 'ModelTypeUserDefined', 'Ideal', 'VanderWaals', 'CaloricallyPerfect',
            'ThermallyPerfect', 'ConstantDensity', 'RedlichKwong'}
GeometryFormat = {'GeometryFormatNull', 'GeometryFormatUserDefined', 'NASA-IGES', 'SDRC',
                  'STEP-AP203', 'STEP-AP242', 'Unigraphics', 'ProEngineer', 'ICEM-CFD'}
GoverningEquationsType = {'GoverningEquationsTypeNull', 'GoverningEquationsTypeUserDefined',
                          'FullPotential', 'Euler', 'NSLaminar', 'NSTurbulent', 'NSLaminarIncompressible',
                          'NSTurbulentIncompressible', 'LatticeBoltzmann'}
GridConnectivityType = {'GridConnectivityTypeNull', 'GridConnectivityTypeUserDefined',
                        'Overset', 'Abutting', 'Abutting1to1'}
GridLocation = {'GridLocationNull', 'GridLocationUserDefined', 'Vertex', 'CellCenter', 'FaceCenter',
                'IFaceCenter', 'JFaceCenter', 'KFaceCenter', 'EdgeCenter'}
GridLocation |= {'IEdgeCenter', 'JEdgeCenter', 'KEdgeCenter'} # Not in SIDS
RigidGridMotionType = {'RigidGridMotionTypeNull', 'RigidGridMotionTypeUserDefined', 'ConstantRate', 'VariableRate'}
SimulationType = {'SimulationTypeNull', 'SimulationTypeUserDefined', 'TimeAccurate', 'NonTimeAccurate'}
ThermalConductivityModelType = {'ModelTypeNull', 'ModelTypeUserDefined', 'ConstantPrandtl',
                                'PowerLaw', 'SutherlandLaw'}
ThermalRelaxationModelType = {'ModelTypeNull', 'ModelTypeUserDefined', 'Frozen', 'ThermalEquilib', 'ThermalNonequilib'}
TurbulenceClosureType = {'ModelTypeNull', 'ModelTypeUserDefined', 'EddyViscosity', 'ReynoldsStress',
                         'ReynoldsStressAlgebraic'}
TurbulenceModelType = {'ModelTypeNull', 'ModelTypeUserDefined', 'Algebraic_BaldwinLomax', 'Algebraic_CebeciSmith',
                       'HalfEquation_JohnsonKing', 'OneEquation_BaldwinBarth', 'OneEquation_SpalartAllmaras',
                       'TwoEquation_JonesLaunder', 'TwoEquation_MenterSST', 'TwoEquation_Wilcox'}
ViscosityModelType = {'ModelTypeNull', 'ModelTypeUserDefined', 'Constant', 'PowerLaw', 'SutherlandLaw'}
WallFunctionType = {'WallFunctionTypeNull', 'WallFunctionTypeUserDefined', 'Generic'}
ZoneType = {'ZoneTypeNull', 'ZoneTypeUserDefined', 'Structured', 'Unstructured'}

# Units
AngleUnits = {'AngleUnitsNull', 'AngleUnitsUserDefined', 'Degree', 'Radian'}
ElectricCurrentUnits   = {'ElectricCurrentUnitsNull', 'ElectricCurrentUnitsUserDefined', 'Ampere',
                          'Abampere', 'Statampere', 'Edison', 'auCurrent'}
LengthUnits = {'LengthUnitsNull', 'LengthUnitsUserDefined', 'Meter', 'Centimeter',
               'Millimeter', 'Foot', 'Inch'}
LuminousIntensityUnits = {'LuminousIntensityUnitsNull', 'LuminousIntensityUnitsUserDefined', 'Candela',
                          'Candle', 'Carcel', 'Hefner', 'Violle'}
MassUnits = {'MassUnitsNull', 'MassUnitsUserDefined', 'Kilogram', 'Gram', 'Slug', 'PoundMass'}
SubstanceAmountUnits= {'SubstanceAmountUnitsNull', 'SubstanceAmountUnitsUserDefined', 'Mole', 'Entities',
                       'StandardCubicFoot', 'StandardCubicMeter'}
TemperatureUnits = {'TemperatureUnitsNull', 'TemperatureUnitsUserDefined', 'Kelvin', 'Celsius',
                    'Rankine', 'Fahrenheit'}
TimeUnits = {'TimeUnitsNull', 'TimeUnitsUserDefined', 'Second'}

ALL_ENUMS = [ArbitraryGridMotionType, AreaType, AverageInterfaceType, BCTypeSimple, BCTypeCompound, BCType, ChemicalKineticsModelType,
             DataClass, EMConductivityModelType, EMElectricFieldModelType, EMMagneticFieldModelType, GasModel, GeometryFormat, GoverningEquationsType, GridConnectivityType, GridLocation,
             RigidGridMotionType, SimulationType, ThermalConductivityModelType, ThermalRelaxationModelType,
             TurbulenceClosureType, TurbulenceModelType, ViscosityModelType, WallFunctionType, ZoneType, AngleUnits, ElectricCurrentUnits,
             LengthUnits, LuminousIntensityUnits, MassUnits, SubstanceAmountUnits, TemperatureUnits, TimeUnits]
UNITS_ENUM = [MassUnits, LengthUnits, TimeUnits, TemperatureUnits, AngleUnits,
              ElectricCurrentUnits, SubstanceAmountUnits, LuminousIntensityUnits]
UNITS_NAME = ['Mass', 'Length', 'Time', 'Temperature', 'Angle',
              'ElectricCurrent', 'SubstanceAmount', 'LuminousIntensity']

LABEL_PROPS = {
    'AdditionalExponents_t': {
        "TYPE" : 'R',
        "SHAPE" : (3,)
    },
    'AdditionalFamilyName_t': {
        "TYPE" : 'C1',
    },
    'AdditionalUnits_t': {
        "TYPE" : 'C1',
        "SHAPE" : (32,3)
    },
    'ArbitraryGridMotion_t': {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : ArbitraryGridMotionType,
        "ALLOWED_CHILDREN": [
            ('DataArray_t', '*'),
            ('Descriptor_t', '*'),
            ('GridLocation_t', '?', 'GridLocation'),
            ('Rind_t', '?', 'Rind'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),

        ],
        "DOC": 'https://cgns.org/standard/SIDS/time.html#arbitrary-grid-motion-structure-definition-arbitrarygridmotion-t',
    },
    'AreaType_t': {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : AreaType,
    },
    'Area_t': {
        "TYPE" : 'MT',
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('AreaType_t', 1, 'AreaType'),
            ('DataArray_t', 2, ['SurfaceArea', 'RegionName']),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/boundary.html#area-structure-definition-area-t',
    },
    'AverageInterfaceType_t': {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : AverageInterfaceType,
    },
    'AverageInterface_t': {
        "TYPE" : 'MT',
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('AverageInterfaceType_t', 1, 'AverageInterfaceType'),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/multizone.html#average-interface-structure-definition-averageinterface-t',
    },
    'Axisymmetry_t': {
        "TYPE" : 'MT',
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', '*', ['AxisymmetryReferencePoint', 'AxisymmetryAxisVector', 'AxisymmetryAngle', 'CoordinateNames']),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ],
        "EXACTLY_ONE_OF": [
            ('AxisymmetryReferencePoint',),
            ('AxisymmetryAxisVector',),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/grid.html#axisymmetry-structure-definition-axisymmetry-t',
    },
    'BCDataSet_t': {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : BCTypeSimple,
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('BCData_t', '<=2', ['DirichletData', 'NeumannData']),
            ('GridLocation_t', '?', 'GridLocation'),
            ('IndexRange_t', '?', 'PointRange'),
            ('IndexArray_t', '?', 'PointList'),
            ('ReferenceState_t', '?', 'ReferenceState'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ],
        "AT_MOST_ONE_OF": [
            ('PointList', 'PointRange'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/boundary.html#boundary-condition-data-set-structure-definition-bcdataset-t',
    },
    'BCData_t': {
        "TYPE" : 'MT',
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/boundary.html#boundary-condition-data-structure-definition-bcdata-t',
    },
    'BCProperty_t': {
        "TYPE" : 'MT',
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('WallFunction_t', '?', 'WallFunction'),
            ('Area_t', '?', 'Area'),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/boundary.html#boundary-condition-property-structure-definition-bcproperty-t',
    },
    'BC_t': {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : BCType,
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('GridLocation_t', '?', 'GridLocation'),
            ('IndexRange_t', '?', 'PointRange'),
            ('IndexArray_t', '<=2', ['PointList', 'InwardNormalList']),
            ('"int[IndexDimension]"', '?', 'InwardNormalIndex'),
            ('BCDataSet_t', '*'),
            ('BCProperty_t', '?', 'BCProperty'),
            ('FamilyName_t', '?', 'FamilyName'),
            ('AdditionalFamilyName_t', '*'),
            ('ReferenceState_t', '?', 'ReferenceState'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
            ('Ordinal_t', '?', 'Ordinal'),
        ],
        "EXACTLY_ONE_OF": [
            ('PointList', 'PointRange'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/boundary.html#boundary-condition-structure-definition-bc-t',
    },
    'BaseIterativeData_t': {
        "TYPE" : 'I4',
        "SHAPE" : (1,),
        "ALLOWED_CHILDREN": [
            ('DataArray_t', '*', ['TimeValues', 'IterationValues', 'NumberOfZones', 'NumberOfFamilies', 'ZonePointers', 'FamilyPointers', '*']),
            ('Descriptor_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ],
        "AT_LEAST_ONE_OF": [
            ('TimeValues', 'IterationValues'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/time.html#base-iterative-data-structure-definition-baseiterativedata-t',
    },
    'CGNSBase_t': {
        "TYPE" : 'I4',
        "SHAPE" : (2,),
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'), 
            ('BaseIterativeData_t', '?'), 
            ('Zone_t', '*'), 
            ('ReferenceState_t', '?', 'ReferenceState'),
            ('Axisymmetry_t', '?', 'Axisymmetry'),
            ('RotatingCoordinates_t', '?', 'RotatingCoordinates'),
            ('Gravity_t', '?', 'Gravity'),
            ('SimulationType_t', '?', 'SimulationType'), 
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('FlowEquationSet_t', '?', 'FlowEquationSet'), 
            ('ConvergenceHistory_t', '?', 'GlobalConvergenceHistory'),
            ('IntegralData_t', '*'),
            ('Family_t', '*'),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/hierarchy.html#cgns-entry-level-structure-definition-cgnsbase-t',
    },
    'CGNSLibraryVersion_t': {
        "TYPE" : 'R4',
        "SHAPE" : (1,),
    },
    'CGNSTree_t': {
        "TYPE" : 'MT',
        "ALLOWED_CHILDREN": [
            ('CGNSBase_t', '*'),
            ('CGNSLibraryVersion_t', 1, 'CGNSLibraryVersion'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/hierarchy.html#hierarchical-structures',
    },
    'ChemicalKineticsModel_t': {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : ChemicalKineticsModelType,
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/equation.html#chemical-kinetics-model-structure-definition-chemicalkineticsmodel-t',
    },
    'ConvergenceHistory_t': {
        "TYPE" : 'I4',
        "SHAPE" : (1,),
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/misc.html#convergence-history-structure-definition-convergencehistory-t',
    },
    'DataArray_t': {
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('DimensionalExponents_t', '?', 'DimensionalExponents'),
            ('DataConversion_t', '?', 'DataConversion'),

        ],
        "DOC": 'https://cgns.org/standard/SIDS/array.html#definition-dataarray-t',
    },
    'DataClass_t' : {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : DataClass,
        "DOC": 'https://cgns.org/standard/SIDS/block.html#definition-dataclass-t',
    }, 
    'DataConversion_t': {
        "TYPE" : 'R',
        "SHAPE" : (2,),
    },
    'Descriptor_t': { # Structure terminale, 1 valeur C1
        "TYPE" : 'C1',
        "DOC": 'https://cgns.org/standard/SIDS/block.html#definition-descriptor-t',
    },
    'DimensionalExponents_t': {
        "TYPE" : 'R',
        "SHAPE" : (5,),
        "ALLOWED_CHILDREN": [
            ('AdditionalExponents_t', '?', 'AdditionalExponents_t'),
        ], 
        "DOC": 'https://cgns.org/standard/SIDS/block.html#definition-dimensionalexponents-t',
    },
    'DimensionalUnits_t': {
        "TYPE" : 'C1',
        "SHAPE" : (32,5),
        "ALLOWED_CHILDREN": [
            ('AdditionalUnits_t', '?', 'AdditionalUnits'),
        ], 
        "DOC": 'https://cgns.org/standard/SIDS/block.html#definition-dimensionalunits-t',
    },
    'DiscreteData_t': {
        "TYPE" : 'MT',
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('GridLocation_t', '?', 'GridLocation'),
            ('IndexRange_t', '?', 'PointRange'),
            ('IndexArray_t', '?', 'PointList'),
            ('Rind_t', '?', 'Rind'),
            ('DataArray_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ], 
        "AT_MOST_ONE_OF": [
            ('PointList', 'PointRange'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/misc.html#discrete-data-structure-definition-discretedata-t',
    },
    'EMConductivityModel_t': {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : EMConductivityModelType,
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/equation.html#electromagnetics-conductivity-model-structure-definition-emconductivitymodel-t',
    },
    'EMElectricFieldModel_t': {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : EMElectricFieldModelType,
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/equation.html#electromagnetics-electric-field-model-structure-definition-emelectricfieldmodel-t',
    },
    'EMMagneticFieldModel_t': {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : EMMagneticFieldModelType,
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/equation.html#electromagnetics-magnetic-field-model-structure-definition-emmagneticfieldmodel-t',
    },
    'Elements_t': {
        "TYPE" : 'I4', 
        "SHAPE" : (2,),
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('Rind_t', '?', 'Rind'),
            ('IndexRange_t', 1, 'ElementRange'),
            ('DataArray_t', '<=4', ['ElementStartOffset', 'ElementConnectivity', 'ParentElements', 'ParentElementsPosition']),
            ('UserDefinedData_t', '*'),
        ],
        "EXACTLY_ONE_OF": [
            ('ElementConnectivity',), 
        ],
        "DOC": 'https://cgns.org/standard/SIDS/grid.html#elements-structure-definition-elements-t',
    },
    'FamilyBCDataSet_t': {
        "TYPE" : 'C1', 
        "ALLOWED_VALUE" : BCTypeSimple,
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('BCData_t', '<=2', ['DirichletData', 'NeumannData']),
            ('ReferenceState_t', '?', 'ReferenceState'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/misc.html#family-boundary-condition-data-set-structure-definition-familybcdataset-t',
    },
    'FamilyBC_t': {
        "TYPE" : 'C1', 
        "ALLOWED_VALUE" : BCType,
        "ALLOWED_CHILDREN": [
            ('FamilyBCDataSet_t', '?'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/misc.html#family-boundary-condition-structure-definition-familybc-t',
    },
    'FamilyName_t': {
        "TYPE" : 'C1', 
    },
    'Family_t': {
        "TYPE" : 'MT', 
        "ALLOWED_CHILDREN": [
            ('FamilyBC_t', '?', 'FamilyBC'), #Conflict on name
            ('GeometryReference_t', '*'),
            ('RotatingCoordinates_t', '?', 'RotatingCoordinates'),
            ('Family_t', '*'),
            ('FamilyName_t', '*'),
            ('Descriptor_t', '*'),
            ('UserDefinedData_t', '*'),
            ('Ordinal_t', '?', 'Ordinal'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/misc.html#family-data-structure-definition-family-t',
    },
    'FlowEquationSet_t': {
        "TYPE" : 'MT', 
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('"int"', '?', 'EquationDimension'),
            ('GoverningEquations_t', '?', 'GoverningEquations'),
            ('GasModel_t', '?', 'GasModel'),
            ('ViscosityModel_t', '?', 'ViscosityModel'),
            ('ThermalConductivityModel_t', '?', 'ThermalConductivityModel'),
            ('TurbulenceClosure_t', '?', 'TurbulenceClosure'),
            ('TurbulenceModel_t', '?', 'TurbulenceModel'),
            ('ThermalRelaxationModel_t', '?', 'ThermalRelaxationModel'),
            ('ChemicalKineticsModel_t', '?', 'ChemicalKineticsModel'),
            ('EMElectricFieldModel_t', '?', 'EMElectricFieldModel'),
            ('EMMagneticFieldModel_t', '?', 'EMMagneticFieldModel'),
            ('EMConductivityModel_t', '?', 'EMConductivityModel'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/equation.html#flow-equation-set-structure-definition-flowequationset-t',
    },
    'FlowSolution_t': {
        "TYPE" : 'MT', 
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('GridLocation_t', '?', 'GridLocation'),
            ('IndexRange_t', '?', 'PointRange'),
            ('IndexArray_t', '?', 'PointList'),
            ('Rind_t', '?', 'Rind'),
            ('DataArray_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ], 
        "AT_MOST_ONE_OF": [
            ('PointList', 'PointRange'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/grid.html#flow-solution-structure-definition-flowsolution-t',
    },
    'GasModel_t': {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : GasModel,
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ], 
        "DOC": 'https://cgns.org/standard/SIDS/equation.html#thermodynamic-gas-model-structure-definition-gasmodel-t',
    },
    'GeometryEntity_t': {
        "TYPE" : 'MT',
    },
    'GeometryFile_t': {
        "TYPE" : 'C1',
    },
    'GeometryFormat_t': {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : GeometryFormat,
    },
    'GeometryReference_t': {
        "TYPE" : 'MT',
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('GeometryFormat_t', 1, 'GeometryFormat'),
            ('GeometryFile_t', 1, 'GeometryFile'),
            ('GeometryEntity_t', '*'),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/misc.html#geometry-reference-structure-definition-geometryreference-t',
    },
    'GoverningEquations_t': {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : GoverningEquationsType,
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('"int[1+...+IndexDimension]"', '?', 'DiffusionModel'),
            ('UserDefinedData_t', '*'),
        ], 
        "DOC": 'https://cgns.org/standard/SIDS/equation.html#governing-equations-structure-definition-governingequations-t',
    },
    'Gravity_t': {
        "TYPE" : 'MT',
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', 1, 'GravityVector'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ], 
        "DOC": 'https://cgns.org/standard/SIDS/misc.html#gravity-data-structure-definition-gravity-t',
    },
    'GridConnectivity1to1_t': {
        "TYPE" : 'C1',
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('"int[IndexDimension]"', '?', 'Transform'),
            ('IndexRange_t', 2, ['PointRange', 'PointRangeDonor']),
            ('GridConnectivityProperty_t', '?', 'GridConnectivityProperty'),
            ('UserDefinedData_t', '*'),
            ('Ordinal_t', '?', 'Ordinal'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/multizone.html#to-1-interface-connectivity-structure-definition-gridconnectivity1to1-t',
    },
    'GridConnectivityProperty_t': {
        "TYPE" : 'MT',
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('Periodic_t', '?', 'Periodic'),
            ('AverageInterface_t', '?', 'AverageInterface'),
            ('UserDefinedData_t', '*'),
        ], 
        "DOC": 'https://cgns.org/standard/SIDS/multizone.html#grid-connectivity-property-structure-definition-gridconnectivityproperty-t',
    },
    'GridConnectivityType_t': {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : GridConnectivityType,  
    },
    'GridConnectivity_t': {
        "TYPE" : 'C1',
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('GridConnectivityType_t', '?', 'GridConnectivityType'),
            ('GridLocation_t', '?', 'GridLocation'),
            ('IndexRange_t', '?', 'PointRange'),
            ('IndexArray_t', '<=2', ['PointList', 'PointListDonor', 'CellListDonor']),
            ('DataArray_t', '?', 'InterpolantsDonor'),
            ('GridConnectivityProperty_t', '?', 'GridConnectivityProperty'),
            ('UserDefinedData_t', '*'),
            ('Ordinal_t', '?', 'Ordinal'),
        ],
        "EXACTLY_ONE_OF": [
            ('PointList', 'PointRange'),
        ],
        "AT_MOST_ONE_OF": [
            ('PointListDonor', 'CellListDonor'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/multizone.html#general-interface-connectivity-structure-definition-gridconnectivity-t',
    },
    'GridCoordinates_t': {
        "TYPE" : 'MT',
        "ALLOWED_CHILDREN": [
            ('DataArray_t', '*'), # Attention, un tableau BoundingBox est facultatif
            ('Descriptor_t', '*'),
            ('Rind_t', '?', 'Rind'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/grid.html#grid-coordinates-structure-definition-gridcoordinates-t'
    },
    'GridLocation_t': {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : GridLocation,
    },
    'IndexArray_t': {}, # Terminal node (with data) I4/I8
    'IndexRange_t': {
        "TYPE" : 'I',
    }, # Terminal data node, I4/I8 or R4/R8
    'IntegralData_t': {
        "TYPE" : 'MT',
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ], 
        "DOC": 'https://cgns.org/standard/SIDS/misc.html#integral-data-structure-definition-integraldata-t',
    },
    'Ordinal_t': {
        "TYPE" : 'I4',
        "SHAPE" : (1,),
    },
    'OversetHoles_t': {
        "TYPE" : 'MT',
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('GridLocation_t', '?', 'GridLocation'),
            ('IndexRange_t', '*'),
            ('IndexArray_t', '?', 'PointList'),
            ('UserDefinedData_t', '*'),
        ], 
        "EXACTLY_ONE_OF": [
            ('PointList', 'IndexRange_t'), # Multiple PointRange allowed
        ],
        "DOC": 'https://cgns.org/standard/SIDS/multizone.html#overset-grid-holes-structure-definition-oversetholes-t',
    },
    'Periodic_t': {
        "TYPE" : 'MT',
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', 3, ['RotationCenter', 'RotationAngle', 'Translation']),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ], 
        "DOC": 'https://cgns.org/standard/SIDS/multizone.html#periodic-interface-structure-definition-periodic-t',
    },
    'ReferenceState_t': {
        "TYPE" : 'MT',
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ], 
        "DOC": 'https://cgns.org/standard/SIDS/misc.html#reference-state-structure-definition-referencestate-t',
    },
    'RigidGridMotion_t': {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : RigidGridMotionType,
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', '*', ['OriginLocation', 'OriginLocation', 'RigidVelocity', 'RigidRotationRate', '*']), # Custom also allowed
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ], 
        "EXACTLY_ONE_OF": [
            ('OriginLocation',), 
        ],
        "DOC": 'https://cgns.org/standard/SIDS/time.html#rigid-grid-motion-structure-definition-rigidgridmotion-t',
    },
    'Rind_t': {
        "TYPE" : 'I4',
    },
    'RotatingCoordinates_t': {
        "TYPE" : 'MT',
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', 2, ['RotationCenter', 'RotationRateVector']),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ], 
    },
    'SimulationType_t': {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : SimulationType,
    },
    'ThermalConductivityModel_t': {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : ThermalConductivityModelType,
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/equation.html#thermal-conductivity-model-structure-definition-thermalconductivitymodel-t',
    },
    'ThermalRelaxationModel_t': {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : ThermalRelaxationModelType,
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ], 
        "DOC": 'https://cgns.org/standard/SIDS/equation.html#thermal-relaxation-model-structure-definition-thermalrelaxationmodel-t',
    },
    'TurbulenceClosure_t': {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : TurbulenceClosureType,
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ], 
        "DOC": 'https://cgns.org/standard/SIDS/equation.html#turbulence-closure-structure-definition-turbulenceclosure-t',
    },
    'TurbulenceModel_t': {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : TurbulenceModelType,
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', '*'),
            ('"int[1+...+IndexDimension]"', '?', 'DiffusionModel'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ], 
        "DOC": 'https://cgns.org/standard/SIDS/equation.html#turbulence-model-structure-definition-turbulencemodel-t',
    },
    'UserDefinedData_t': {
        "TYPE" : 'MT',
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('GridLocation_t', '?', 'GridLocation'),
            ('IndexRange_t', '?', 'PointRange'),
            ('IndexArray_t', '?', 'PointList'),
            ('DataArray_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('FamilyName_t', '?', 'FamilyName'),
            ('AdditionalFamilyName_t', '*'),
            ('UserDefinedData_t', '*'),
            ('Ordinal_t', '?', 'Ordinal')
        ],
        "AT_MOST_ONE_OF": [
            ('PointList', 'PointRange'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/misc.html#user-defined-data-structure-definition-userdefineddata-t',
    },
    'ViscosityModel_t': {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : ViscosityModelType,
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/equation.html#molecular-viscosity-model-structure-definition-viscositymodel-t',
    },
    'WallFunctionType_t': {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : WallFunctionType,
    },
    'WallFunction_t': {
        "TYPE" : 'MT',
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('WallFunctionType_t', 1, 'WallFunctionType'),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/boundary.html#wall-function-structure-definition-wallfunction-t',
    },
    'ZoneBC_t': {
        "TYPE" : 'MT',
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('BC_t', '*'),
            ('ReferenceState_t', '?', 'ReferenceState'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/boundary.html#zonal-boundary-condition-structure-definition-zonebc-t',
    },
    'ZoneGridConnectivity_t': {
        "TYPE" : 'MT',
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('GridConnectivity1to1_t', '*'),
            ('GridConnectivity_t', '*'),
            ('OversetHoles_t', '*'),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/multizone.html#zonal-connectivity-structure-definition-zonegridconnectivity-t',
    },
    'ZoneIterativeData_t': {
        "TYPE" : 'MT',
        "ALLOWED_CHILDREN": [
            ('DataArray_t', '*', ['RigidGridMotionPointers', 'ArbitraryGridMotionPointers', 'GridCoordinatesPointers', 
                                  'FlowSolutionPointers', 'ZoneGridConnectivityPointers', 'ZoneSubRegionPointers', '*']),
            ('Descriptor_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),

        ],
        "DOC": 'https://cgns.org/standard/SIDS/time.html#zone-iterative-data-structure-definition-zoneiterativedata-t',
    },
    'ZoneSubRegion_t': {
        "TYPE" : 'I4',
        "SHAPE" : (1,),
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('GridLocation_t', '?', 'GridLocation'),
            ('IndexRange_t', '?', 'PointRange'),
            ('IndexArray_t', '?', 'PointList'),
            ('Rind_t', '?', 'Rind'),
            ('DataArray_t', '*'),
            ('FamilyName_t', '?', 'FamilyName'),
            ('AdditionalFamilyName_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ],
        "EXACTLY_ONE_OF": [
            ('PointList', 'PointRange', 'BCRegionName', 'GridConnectivityRegionName'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/grid.html#zone-subregion-structure-definition-zonesubregion-t',
    },
    'ZoneType_t': {
        "TYPE" : 'C1',
        "ALLOWED_VALUE" : ZoneType,
    },
    'Zone_t': {
        "TYPE" : 'I',
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('ZoneType_t', 1, 'ZoneType'),
            ('GridCoordinates_t', '*'), #Maybe first is imposed (GridCoordinates)
            ('Elements_t', '*'),
            ('RigidGridMotion_t', '*'),
            ('ArbitraryGridMotion_t', '*'),
            ('FamilyName_t', '?', 'FamilyName'),
            ('AdditionalFamilyName_t', '*'),
            ('FlowSolution_t', '*'),
            ('DiscreteData_t', '*'),
            ('IntegralData_t', '*'),
            ('ZoneGridConnectivity_t', '*'),
            ('ZoneSubRegion_t', '*'),
            ('ZoneBC_t', '?', 'ZoneBC'),
            ('ZoneIterativeData_t', '?'),
            ('ReferenceState_t', '?', 'ReferenceState'),
            ('RotatingCoordinates_t', '?', 'RotatingCoordinates'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('FlowEquationSet_t', '?', 'FlowEquationSet'),
            ('ConvergenceHistory_t', '?', 'ZoneConvergenceHistory'),
            ('UserDefinedData_t', '*'),
            ('Ordinal_t', '?', 'Ordinal'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/hierarchy.html#zone-structure-definition-zone-t',
    },
}




DATANAME_IDENTIFIERS = {
# A1 Coordinate Systems
"CoordinateX", "CoordinateY", "CoordinateZ", "CoordinateR", "CoordinateTheta", "CoordinatePhi",
"CoordinateNormal", "CoordinateTangential", "CoordinateXi", "CoordinateEta", "CoordinateZeta",
"CoordinateTransform", "InterpolantsDonor", "ElementConnectivity", "ParentData",
# A2 Flowfield Solution
"Potential", "StreamFunction", "Density", "Pressure", "Temperature", "EnergyInternal", "Enthalpy",
"Entropy", "EntropyApprox", "DensityStagnation", "PressureStagnation", "TemperatureStagnation",
"EnergyStagnation", "EnthalpyStagnation", "EnergyStagnationDensity", "VelocityX", "VelocityY",
"VelocityZ", "VelocityR", "VelocityTheta", "VelocityPhi", "VelocityMagnitude", "VelocityNormal",
"VelocityTangential", "VelocitySound", "VelocitySoundStagnation", "MomentumX", "MomentumY", "MomentumZ",
"MomentumMagnitude", "RotatingVelocityX", "RotatingVelocityY", "RotatingVelocityZ", "RotatingMomentumX",
"RotatingMomentumY", "RotatingMomentumZ", "RotatingVelocityMagnitude", "RotatingPressureStagnation",
"RotatingEnergyStagnation", "RotatingEnergyStagnationDensity", "RotatingEnthalpyStagnation", "EnergyKinetic",
"PressureDynamic", "SoundIntensityDB", "SoundIntensity", "VorticityX", "VorticityY", "VorticityZ",
"VorticityMagnitude", "SkinFrictionX", "SkinFrictionY", "SkinFrictionZ", "SkinFrictionMagnitude",
"VelocityAngleX", "VelocityAngleY", "VelocityAngleZ", "VelocityUnitVectorX", "VelocityUnitVectorY",
"VelocityUnitVectorZ", "MassFlow", "ViscosityKinematic", "ViscosityMolecular", "ViscosityEddyDynamic",
"ViscosityEddy", "ThermalConductivity", "PowerLawExponent", "SutherlandLawConstant", "TemperatureReference",
"ViscosityMolecularReference", "ThermalConductivityReference", "IdealGasConstant", "SpecificHeatPressure",
"SpecificHeatVolume", "ReynoldsStressXX", "ReynoldsStressXY", "ReynoldsStressXZ", "ReynoldsStressYY",
"ReynoldsStressYZ", "ReynoldsStressZZ", "MolecularWeightSymbol", "HeatOfFormationSymbol",
"FuelAirRatio", "ReferenceTemperatureHOF", "MassFractionSymbol", "LaminarViscositySymbol",
"ThermalConductivitySymbol", "EnthalpyEnergyRatio", "CompressibilityFactor", "VibrationalElectronEnergy",
"HeatOfFormation", "VibrationalElectronTemperature",
"SpeciesDensitySymbol", "MoleFractionSymbol", "Voltage", "ElectricFieldX", "ElectricFieldY",
"ElectricFieldZ", "MagneticFieldX", "MagneticFieldY", "MagneticFieldZ", "CurrentDensityX",
"CurrentDensityY", "CurrentDensityZ", "ElectricConductivity", "LorentzForceX", "LorentzForceY",
"LorentzForceZ", "JouleHeating", "LengthReference",
# A3 Turbulence Model Solution
"TurbulentDistance", "TurbulentEnergyKinetic", "TurbulentDissipation", "TurbulentDissipationRate",
"TurbulentBBReynolds", "TurbulentSANuTilde",
# A4 Nondimensional Parameters
"Mach", "Mach_Velocity", "Mach_VelocitySound", "RotatingMach", "Reynolds", "Reynolds_Velocity",
"Reynolds_Length", "Reynolds_ViscosityKinematic", "Prandtl", "Prandtl_ThermalConductivity",
"Prandtl_ViscosityMolecular", "Prandtl_SpecificHeatPressure", "PrandtlTurbulent", "SpecificHeatRatio",
"SpecificHeatRatio_Pressure", "SpecificHeatRatio_Volume", "CoefPressure", "CoefSkinFrictionX",
"CoefSkinFrictionY", "CoefSkinFrictionZ", "Coef_PressureDynamic", "Coef_PressureReference",
"Weber", "WeberDensity", "WeberVelocity", "WeberLength", "WeberSurfaceTension",
# A5 Characteristics and Riemann Invariants Based on 1D flow
"RiemannInvariantPlus", "RiemannInvariantMinus", "CharacteristicEntropy", "CharacteristicVorticity1",
"CharacteristicVorticity2", "CharacteristicAcousticPlus", "CharacteristicAcousticMinus",
# A6 Forces and Moments
"ForceX", "ForceY", "ForceZ", "ForceR", "ForceTheta", "ForcePhi", "Lift", "Drag", "MomentX",
"MomentY", "MomentZ", "MomentR", "MomentTheta", "MomentPhi", "MomentXi", "MomentEta", "MomentZeta",
"Moment_CenterX", "Moment_CenterY", "Moment_CenterZ", "CoefLift", "CoefDrag", "CoefMomentX",
"CoefMomentY", "CoefMomentZ", "CoefMomentR", "CoefMomentTheta", "CoefMomentPhi", "CoefMomentXi",
"CoefMomentEta", "CoefMomentZeta", "Coef_PressureDynamic", "Coef_Area", "Coef_Length",
# A7 Time-Dependant flows
"TimeValues", "IterationValues", "NumberOfZones", "NumberOfFamilies", "ZonePointers",
"FamilyPointers", "RigidGridMotionPointers", "ArbitraryGridMotionPointers", "GridCoordinatesPointers",
"FlowSolutionPointers", "OriginLocation", "RigidRotationAngle", "RigidVelocity", "RigidRotationRate",
"GridVelocityX", "GridVelocityY", "GridVelocityZ", "GridVelocityR", "GridVelocityTheta", "GridVelocityPhi",
"GridVelocityXi", "GridVelocityEta", "GridVelocityZeta",
}


# Build table of reserved names
for label in LABEL_PROPS.values():
    if "ALLOWED_CHILDREN" in label:
        reserved = {}
        for child in label["ALLOWED_CHILDREN"]:
            if len(child) > 2:
                reserved_lbl = child[0]
                reserved_name = child[2]
                if isinstance(reserved_name, str):
                    reserved[reserved_name] = reserved_lbl
                else: # List
                    for rsvd_name in reserved_name:
                        reserved[rsvd_name] = reserved_lbl

        label["RESERVED_NAMES"] = reserved
# Add particular cases
LABEL_PROPS['ArbitraryGridMotion_t']["RESERVED_NAMES"]['ArbitraryGridMotionType'] = None
LABEL_PROPS['Zone_t']["RESERVED_NAMES"]['GridCoordinates'] = 'GridCoordinates_t'
# Add empty entry for strange CGNS labels
for lbl in ['"int[IndexDimension]"', '"int"', '"int[1+...+IndexDimension]"']:
    LABEL_PROPS[lbl] = {}

ALL_LABELS = set(LABEL_PROPS.keys()) \
           | {T[0] for D in LABEL_PROPS.values() for T in D.get('ALLOWED_CHILDREN', [])}

# In CGNSLib custom enum XXXNull and XXXUserDefined actually maps to Null / UserDefined
for value_set in ALL_ENUMS:
    value_set |= {'Null', 'UserDefined'}



# Debug section
# List cardinal symbols
symbs = set()
for data in LABEL_PROPS.values():
    child_list = data.get("ALLOWED_CHILDREN", [])
    for lbl in child_list:
        symbs |= set([lbl[1]])


# Reserved names
resvd = set()
for data in LABEL_PROPS.values():
    reserved = data.get("RESERVED_NAMES", {})
    resvd |= set(reserved.keys())
