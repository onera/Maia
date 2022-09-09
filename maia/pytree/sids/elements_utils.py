elements_properties = [
#CGNS_Id, ElementName        ,  dim, nVtx,nEdge,nFace, refElt
    ("Null"                  , None, None, None, None,   None),
    ("UserDefined"           , None, None, None, None,   None),
    ("NODE"                  ,    0,    1,    1,    0,  "NODE"),
    ("BAR_2"                 ,    1,    2,    1,    0,   "BAR"),
    ("BAR_3"                 ,    1,    3,    1,    0,   "BAR"),
    ("TRI_3"                 ,    2,    3,    3,    1,   "TRI"),
    ("TRI_6"                 ,    2,    6,    3,    1,   "TRI"),
    ("QUAD_4"                ,    2,    4,    4,    1,  "QUAD"),
    ("QUAD_8"                ,    2,    8,    4,    1,  "QUAD"),
    ("QUAD_9"                ,    2,    9,    4,    1,  "QUAD"),
    ("TETRA_4"               ,    3,    4,    6,    4, "TETRA"),
    ("TETRA_10"              ,    3,   10,    6,    4, "TETRA"),
    ("PYRA_5"                ,    3,    5,    8,    5,  "PYRA"),
    ("PYRA_14"               ,    3,   14,    8,    5,  "PYRA"),
    ("PENTA_6"               ,    3,    6,    9,    5, "PENTA"),
    ("PENTA_15"              ,    3,   15,    9,    5, "PENTA"),
    ("PENTA_18"              ,    3,   18,    9,    5, "PENTA"),
    ("HEXA_8"                ,    3,    8,   12,    6,  "HEXA"),
    ("HEXA_20"               ,    3,   20,   12,    6,  "HEXA"),
    ("HEXA_27"               ,    3,   27,   12,    6,  "HEXA"),
    ("MIXED"                 , None, None, None, None,    None),
    ("PYRA_13"               ,    3,   13,    8,    5,  "PYRA"),
    ("NGON_n"                ,    2, None, None, None,    None),
    ("NFACE_n"               ,    3, None, None, None,    None),
    ("BAR_4"                 ,    1,    4,    1,    0,   "BAR"),
    ("TRI_9"                 ,    2,    9,    3,    1,   "TRI"),
    ("TRI_10"                ,    2,   10,    3,    1,   "TRI"),
    ("QUAD_12"               ,    2,   12,    4,    1,  "QUAD"),
    ("QUAD_16"               ,    2,   16,    4,    1,  "QUAD"),
    ("TETRA_16"              ,    3,   16,    6,    4, "TETRA"),
    ("TETRA_20"              ,    3,   20,    6,    4, "TETRA"),
    ("PYRA_21"               ,    3,   21,    8,    5,  "PYRA"),
    ("PYRA_29"               ,    3,   29,    8,    5,  "PYRA"),
    ("PYRA_30"               ,    3,   30,    8,    5,  "PYRA"),
    ("PENTA_24"              ,    3,   24,    9,    5, "PENTA"),
    ("PENTA_38"              ,    3,   38,    9,    5, "PENTA"),
    ("PENTA_40"              ,    3,   40,    9,    5, "PENTA"),
    ("HEXA_32"               ,    3,   32,   12,    6,  "HEXA"),
    ("HEXA_56"               ,    3,   56,   12,    6,  "HEXA"),
    ("HEXA_64"               ,    3,   64,   12,    6,  "HEXA"),
    ("BAR_5"                 ,    1,    5,    1,    0,   "BAR"),
    ("TRI_12"                ,    2,   12,    3,    1,   "TRI"),
    ("TRI_15"                ,    2,   15,    3,    1,   "TRI"),
    ("QUAD_P4_16"            ,    2,   16,    4,    1,  "QUAD"),
    ("QUAD_25"               ,    2,   25,    4,    1,  "QUAD"),
    ("TETRA_22"              ,    3,   22,    6,    4, "TETRA"),
    ("TETRA_34"              ,    3,   34,    6,    4, "TETRA"),
    ("TETRA_35"              ,    3,   35,    6,    4, "TETRA"),
    ("PYRA_P4_29"            ,    3,   29,    8,    5,  "PYRA"),
    ("PYRA_50"               ,    3,   50,    8,    5,  "PYRA"),
    ("PYRA_55"               ,    3,   55,    8,    5,  "PYRA"),
    ("PENTA_33"              ,    3,   33,    9,    5, "PENTA"),
    ("PENTA_66"              ,    3,   66,    9,    5, "PENTA"),
    ("PENTA_75"              ,    3,   75,    9,    5, "PENTA"),
    ("HEXA_44"               ,    3,   44,   12,    6,  "HEXA"),
    ("HEXA_98"               ,    3,   98,   12,    6,  "HEXA"),
    ("HEXA_125"              ,    3,  125,   12,    6,  "HEXA"),
    ]

def element_name(n):
  assert n < len(elements_properties)
  return elements_properties[n][0]

def cgns_name_to_id(name):
  return [EP[0] for EP in elements_properties].index(name)

def element_dim(n):
  assert n < len(elements_properties)
  return elements_properties[n][1]

def element_number_of_nodes(n):
  assert n < len(elements_properties)
  return elements_properties[n][2]
