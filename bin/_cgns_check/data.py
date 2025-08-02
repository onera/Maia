LABEL_PROPS = {
    'AdditionalExponents_t': {},
    'AdditionalFamilyName_t': {},
    'AdditionalUnits_t': {},
    'ArbitraryGridMotion_t': {
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
    'AreaType_t': {}, # Terminal enum,   AreaTypeNull, AreaTypeUserDefined, BleedArea, CaptureArea
    'Area_t': {
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('AreaType_t', 1, 'AreaType'),
            ('DataArray_t', 2, ['SurfaceArea', 'RegionName']),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/boundary.html#area-structure-definition-area-t',
    },
    'AverageInterface_t': {
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('AverageInterfaceType_t', 1, 'AverageInterfaceType'),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/multizone.html#average-interface-structure-definition-averageinterface-t',
    },
    'Axisymmetry_t': {
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
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('WallFunction_t', '?', 'WallFunction'),
            ('Area_t', '?', 'Area'),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/boundary.html#boundary-condition-property-structure-definition-bcproperty-t',
    },
    'BC_t': {
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('GridLocation_t', '?', 'GridLocation'),
            ('IndexRange_t', '?', 'PointRange'),
            ('IndexArray_t', '?', 'PointList'),
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
            # Attention, InwardNormalList est aussi un IndexArray_t,
        ],
        "EXACTLY_ONE_OF": [
            ('PointList', 'PointRange'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/boundary.html#boundary-condition-structure-definition-bc-t',
    },
    'BaseIterativeData_t': { # 1 valeur I4
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
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'), 
            ('BaseIterativeData_t', '?'), 
            ('Zone_t', '*'), 
            ('ParticleZone_t', '*'), 
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
    'CGNSLibraryVersion_t': {}, # Structure terminale, 1 valeur R4
    'CGNSTree_t': {
        "ALLOWED_CHILDREN": [
            ('CGNSBase_t', '*'),
            ('CGNSLibraryVersion_t', 1, 'CGNSLibraryVersion'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/hierarchy.html#hierarchical-structures',
    },
    'ChemicalKineticsModel_t': { # 1 valeur C1
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/equation.html#chemical-kinetics-model-structure-definition-chemicalkineticsmodel-t',
    },
    'ConvergenceHistory_t': { # Value: nb iter 
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
    'DataClass_t' : {# Terminal, 1 valeur str enum
        "DOC": 'https://cgns.org/standard/SIDS/block.html#definition-dataclass-t',
    }, 
    'DataConversion_t': {}, # Structure terminale, 2 valeurs R4/R8
    'Descriptor_t': { # Structure terminale, 1 valeur C1
        "DOC": 'https://cgns.org/standard/SIDS/block.html#definition-descriptor-t',
    },
    'DimensionalExponents_t': { # Value : 5 float (R4/R8)
        "ALLOWED_CHILDREN": [
            ('AdditionalExponents_t', '?', 'AdditionalExponents_t'),
        ], 
        "DOC": 'https://cgns.org/standard/SIDS/block.html#definition-dimensionalexponents-t',
    },
    'DimensionalUnits_t': { # 32,5 str array
        "ALLOWED_CHILDREN": [
            ('AdditionalUnits_t', '?', 'AdditionalUnits'),
        ], 
        "DOC": 'https://cgns.org/standard/SIDS/block.html#definition-dimensionalunits-t',
    },
    'DiscreteData_t': {
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
    'Elements_t': {
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
        "ALLOWED_CHILDREN": [
            ('FamilyBCDataSet_t', '?'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/misc.html#family-boundary-condition-structure-definition-familybc-t',
    },
    'FamilyName_t': {},
    'Family_t': {
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
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('int', '?', 'EquationDimension'),
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
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ], 
        "DOC": 'https://cgns.org/standard/SIDS/equation.html#thermodynamic-gas-model-structure-definition-gasmodel-t',
    },
    'GeometryEntity_t': {}, # Terminal str C1,
    'GeometryFile_t': {}, #Terminal str C1
    'GeometryFormat_t': {}, #Terminal str C1
    'GeometryReference_t': {
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
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('int[1 + ... + IndexDimension]', '?', 'DiffusionModel'),
            ('UserDefinedData_t', '*'),
        ], 
        "DOC": 'https://cgns.org/standard/SIDS/equation.html#governing-equations-structure-definition-governingequations-t',
    },
    'Gravity_t': {
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
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('Periodic_t', '?', 'Periodic'),
            ('AverageInterface_t', '?', 'AverageInterface'),
            ('UserDefinedData_t', '*'),
        ], 
        "DOC": 'https://cgns.org/standard/SIDS/multizone.html#grid-connectivity-property-structure-definition-gridconnectivityproperty-t',
    },
    'GridConnectivityType_t': {}, # Terminal node (enum),
    'GridConnectivity_t': {
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
    'GridLocation_t': {},
    'IndexArray_t': {}, # Terminal node (with data) I4/I8
    'IndexRange_t': {}, # Terminal data node, I4/I8 or R4/R8
    'IntegralData_t': {
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ], 
        "DOC": 'https://cgns.org/standard/SIDS/misc.html#integral-data-structure-definition-integraldata-t',
    },
    'Ordinal_t': {}, # Terminal node I4 scalar data
    'OversetHoles_t': {
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
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ], 
        "DOC": 'https://cgns.org/standard/SIDS/misc.html#reference-state-structure-definition-referencestate-t',
    },
    'RigidGridMotion_t': { #Enum value
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
    'Rind_t': {}, # Terminal data node  2*IdxDim, I4
    'RotatingCoordinates_t': {
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', 2, ['RotationCenter', 'RotationRateVector']),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ], 
    },
    'SimulationType_t': {}, # Terminal enum node SimulationTypeNull, SimulationTypeUserDefined, TimeAccurate, NonTimeAccurate
    'ThermalConductivityModel_t': {
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
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', '*'),
            ('int[1 + ... + IndexDimension]', '?', 'DiffusionModel'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ], 
        "DOC": 'https://cgns.org/standard/SIDS/equation.html#turbulence-model-structure-definition-turbulencemodel-t',
    },
    'UserDefinedData_t': {
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
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('DataArray_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/equation.html#molecular-viscosity-model-structure-definition-viscositymodel-t',
    },
    'WallFunction_t': {
        "ALLOWED_CHILDREN": [
            ('Descriptor_t', '*'),
            ('WallFunctionType_t', 1, 'WallFunctionType'),
            ('UserDefinedData_t', '*'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/boundary.html#wall-function-structure-definition-wallfunction-t',
    },
    'ZoneBC_t': {
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
        "DOC": 'https://cgns.org/standard/SIDS/time.html#zone-iterative-data-structure-definition-zoneiterativedata-t',
    },
    'ZoneSubRegion_t': {
        "ALLOWED_CHILDREN": [
            ('DataArray_t', '*'),
            ('Descriptor_t', '*'),
            ('DataClass_t', '?', 'DataClass'),
            ('DimensionalUnits_t', '?', 'DimensionalUnits'),
            ('UserDefinedData_t', '*'),
        ],
        "EXACTLY_ONE_OF": [
            ('PointList', 'PointRange', 'BCRegionName', 'GridConnectivityRegionName'),
        ],
        "DOC": 'https://cgns.org/standard/SIDS/grid.html#zone-subregion-structure-definition-zonesubregion-t',
    },
    'ZoneType_t': {},
    'Zone_t': {
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

ALL_LABELS = set(LABEL_PROPS.keys()) \
           | {T[0] for D in LABEL_PROPS.values() for T in D.get('ALLOWED_CHILDREN', [])}