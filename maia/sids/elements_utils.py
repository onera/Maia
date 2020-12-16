
elements_traits = [
  ["ElementTypeNull"        ,  0  ],
  ["ElementTypeUserDefined" , -1  ],
  ["NODE"                   ,  1  ],
  ["BAR_2"                  ,  2  ],
  ["BAR_3"                  ,  3  ],
  ["TRI_3"                  ,  3  ],
  ["TRI_6"                  ,  6  ],
  ["QUAD_4"                 ,  4  ],
  ["QUAD_8"                 ,  8  ],
  ["QUAD_9"                 ,  9  ],
  ["TETRA_4"                ,  4  ],
  ["TETRA_10"               ,  10 ],
  ["PYRA_5"                 ,  5  ],
  ["PYRA_14"                ,  14 ],
  ["PENTA_6"                ,  6  ],
  ["PENTA_15"               ,  15 ],
  ["PENTA_18"               ,  18 ],
  ["HEXA_8"                 ,  8  ],
  ["HEXA_20"                ,  20 ],
  ["HEXA_27"                ,  27 ],
  ["MIXED"                  , -1  ],
  ["PYRA_13"                ,  13 ],
  ["NGON_n"                 , -1  ],
  ["NFACE_n"                , -1  ],
  ["BAR_4"                  ,  4  ],
  ["TRI_9"                  ,  9  ],
  ["TRI_10"                 ,  10 ],
  ["QUAD_12"                ,  12 ],
  ["QUAD_16"                ,  16 ],
  ["TETRA_16"               ,  16 ],
  ["TETRA_20"               ,  20 ],
  ["PYRA_21"                ,  21 ],
  ["PYRA_29"                ,  29 ],
  ["PYRA_30"                ,  30 ],
  ["PENTA_24"               ,  24 ],
  ["PENTA_38"               ,  38 ],
  ["PENTA_40"               ,  40 ],
  ["HEXA_32"                ,  32 ],
  ["HEXA_56"                ,  56 ],
  ["HEXA_64"                ,  64 ],
  ["BAR_5"                  ,  5  ],
  ["TRI_12"                 ,  12 ],
  ["TRI_15"                 ,  15 ],
  ["QUAD_P4_16"             ,  16 ],
  ["QUAD_25"                ,  25 ],
  ["TETRA_22"               ,  22 ],
  ["TETRA_34"               ,  34 ],
  ["TETRA_35"               ,  35 ],
  ["PYRA_P4_29"             ,  29 ],
  ["PYRA_50"                ,  50 ],
  ["PYRA_55"                ,  55 ],
  ["PENTA_33"               ,  33 ],
  ["PENTA_66"               ,  66 ],
  ["PENTA_75"               ,  75 ],
  ["HEXA_44"                ,  44 ],
  ["HEXA_98"                ,  98 ],
  ["HEXA_125"               ,  125]
]

def element_name (n):
  return elements_traits[n][0]

def element_number_of_nodes(n):
  return elements_traits[n][1]

def element_types_of_dimension(dim):
  if dim==0: return ["NODE"]

  if dim==1: return ["BAR_2", "BAR_3", "BAR_4", "BAR_5"]

  if dim==2: return [ \
    "TRI_3", "TRI_6", "TRI_9", "TRI_10", "TRI_12", "TRI_15, \
    QUAD_4", "QUAD_8", "QUAD_9", "QUAD_12", "QUAD_16", "QUAD_P4_16", "QUAD_25", "NGON_n" \
  ] # note: no MIXED

  if dim==3: return [ \
    "TETRA_4", "TETRA_10", "TETRA_16", "TETRA_20", "TETRA_22", "TETRA_34", "TETRA_35, \
    PYRA_5", "PYRA_13", "PYRA_14", "PYRA_21", "PYRA_29", "PYRA_30", "PYRA_P4_29", "PYRA_50", "PYRA_55, \
    PENTA_6", "PENTA_15", "PENTA_18", "PENTA_24", "PENTA_38", "PENTA_40", "PENTA_33", "PENTA_66", "PENTA_75, \
    HEXA_8", "HEXA_20", "HEXA_27", "HEXA_32", "HEXA_56", "HEXA_64", "HEXA_44", "HEXA_98", "HEXA_125", "NFACE_n" \
  ] # note: no MIXED

  return []


PDM_MESH_NODAL_POINT = 0
PDM_MESH_NODAL_BAR2 = 1
PDM_MESH_NODAL_TRIA3 = 2
PDM_MESH_NODAL_QUAD4 = 3
PDM_MESH_NODAL_POLY_2D = 4
PDM_MESH_NODAL_TETRA4 = 5
PDM_MESH_NODAL_PYRAMID5 = 6
PDM_MESH_NODAL_PRISM6 = 7
PDM_MESH_NODAL_HEXA8 = 8
PDM_MESH_NODAL_POLY_3D = 9
cgns_pdm_element_type = [
  ("NODE"    ,  PDM_MESH_NODAL_POINT    ),
  ("BAR_2"   ,  PDM_MESH_NODAL_BAR2     ),
  ("TRI_3"   ,  PDM_MESH_NODAL_TRIA3    ),
  ("QUAD_4"  ,  PDM_MESH_NODAL_QUAD4    ),
  ("TETRA_4" ,  PDM_MESH_NODAL_TETRA4   ),
  ("PYRA_5"  ,  PDM_MESH_NODAL_PYRAMID5 ),
  ("PENTA_6" ,  PDM_MESH_NODAL_PRISM6   ),
  ("HEXA_8"  ,  PDM_MESH_NODAL_HEXA8    ),
]

def cgns_elt_name_to_pdm_element_type(name):
  for cgns_name,pdm_name in cgns_pdm_element_type:
    if cgns_name==name: return pdm_name
  raise NameError("No PDM element associated to "+name)

