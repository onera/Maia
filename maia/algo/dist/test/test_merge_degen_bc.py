import copy
import itertools
import mpi4py.MPI      as MPI
import numpy           as np

import pytest
import pytest_parallel

import maia
import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia.algo.part.point_cloud_utils as PCU

from maia                     import npy_pdm_gnum_dtype    as pdm_gnum_dtype
from maia.algo.dist           import merge_degen_bc        as MDB
from maia.algo.dist           import remove_element        as RME
from maia.algo.dist.merge_ids import merge_distributed_ids
from maia.transfer            import protocols             as EP
from maia.utils               import par_utils

import Pypdm.Pypdm as PDM

int_type = 4 if pdm_gnum_dtype==np.int32 else 8

yaml_ref = f'''
  CGNSLibraryVersion CGNSLibraryVersion_t 4.2:
  Base CGNSBase_t I4 [3, 3]:
    INLET Family_t:
    OUTLET Family_t:
    AXIS Family_t:
    FARFIELD Family_t:
    PER1 Family_t:
    PER2 Family_t:
    zone Zone_t I{int_type} [[105, 64, 0]]:
      ZoneType ZoneType_t 'Unstructured':
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t:
          R8 : [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0,
                1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0,
                2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0,
                3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0,
                4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        CoordinateY DataArray_t:
          R8 : [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 0.9807852804032304,
                0.9807852804032304, 0.9807852804032304, 0.9807852804032304, 0.9807852804032304, 1.9615705608064609, 1.9615705608064609,
                1.9615705608064609, 1.9615705608064609, 1.9615705608064609, 2.9423558412096913, 2.9423558412096913, 2.9423558412096913,
                2.9423558412096913, 2.9423558412096913, 3.9231411216129217, 3.9231411216129217, 3.9231411216129217, 3.9231411216129217,
                3.9231411216129217, 0.9238795325112867, 0.9238795325112867, 0.9238795325112867, 0.9238795325112867, 0.9238795325112867,
                1.8477590650225735, 1.8477590650225735, 1.8477590650225735, 1.8477590650225735, 1.8477590650225735, 2.77163859753386,
                2.77163859753386, 2.77163859753386, 2.77163859753386, 2.77163859753386, 3.695518130045147, 3.695518130045147,
                3.695518130045147, 3.695518130045147, 3.695518130045147, 0.8314696123025452, 0.8314696123025452, 0.8314696123025452,
                0.8314696123025452, 0.8314696123025452, 1.6629392246050905, 1.6629392246050905, 1.6629392246050905, 1.6629392246050905,
                1.6629392246050905, 2.4944088369076356, 2.4944088369076356, 2.4944088369076356, 2.4944088369076356, 2.4944088369076356,
                3.325878449210181, 3.325878449210181, 3.325878449210181, 3.325878449210181, 3.325878449210181, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.7071067811865476, 0.7071067811865476, 0.7071067811865476, 0.7071067811865476, 0.7071067811865476,
                1.4142135623730951, 1.4142135623730951, 1.4142135623730951, 1.4142135623730951, 1.4142135623730951, 2.121320343559643,
                2.121320343559643, 2.121320343559643, 2.121320343559643, 2.121320343559643, 2.8284271247461903, 2.8284271247461903,
                2.8284271247461903, 2.8284271247461903, 2.8284271247461903]
        CoordinateZ DataArray_t:
          R8 : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.19509032201612825,
                0.19509032201612825, 0.19509032201612825, 0.19509032201612825, 0.19509032201612825, 0.3901806440322565, 0.3901806440322565,
                0.3901806440322565, 0.3901806440322565, 0.3901806440322565, 0.5852709660483848, 0.5852709660483848, 0.5852709660483848,
                0.5852709660483848, 0.5852709660483848, 0.780361288064513, 0.780361288064513, 0.780361288064513, 0.780361288064513,
                0.780361288064513, 0.3826834323650898, 0.3826834323650898, 0.3826834323650898, 0.3826834323650898, 0.3826834323650898,
                0.7653668647301796, 0.7653668647301796, 0.7653668647301796, 0.7653668647301796, 0.7653668647301796, 1.1480502970952693,
                1.1480502970952693, 1.1480502970952693, 1.1480502970952693, 1.1480502970952693, 1.5307337294603591, 1.5307337294603591,
                1.5307337294603591, 1.5307337294603591, 1.5307337294603591, 0.5555702330196022, 0.5555702330196022, 0.5555702330196022,
                0.5555702330196022, 0.5555702330196022, 1.1111404660392044, 1.1111404660392044, 1.1111404660392044, 1.1111404660392044,
                1.1111404660392044, 1.6667106990588065, 1.6667106990588065, 1.6667106990588065, 1.6667106990588065, 1.6667106990588065,
                2.2222809320784087, 2.2222809320784087, 2.2222809320784087, 2.2222809320784087, 2.2222809320784087, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.7071067811865475, 0.7071067811865475, 0.7071067811865475, 0.7071067811865475, 0.7071067811865475,
                1.414213562373095, 1.414213562373095, 1.414213562373095, 1.414213562373095, 1.414213562373095, 2.1213203435596424,
                2.1213203435596424, 2.1213203435596424, 2.1213203435596424, 2.1213203435596424, 2.82842712474619, 2.82842712474619,
                2.82842712474619, 2.82842712474619, 2.82842712474619]
      ZoneBC ZoneBC_t:
        Zmin BC_t 'FamilySpecified':
          GridLocation GridLocation_t 'FaceCenter':
          FamilyName FamilyName_t 'PER1':
          PointList IndexArray_t I{int_type} [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 18, 21, 24, 27]]:
        Zmax BC_t 'FamilySpecified':
          GridLocation GridLocation_t 'FaceCenter':
          FamilyName FamilyName_t 'PER2':
          PointList IndexArray_t I{int_type} [[194, 197, 200, 203, 207, 210, 213, 215, 217, 218, 219, 220, 221, 222, 223, 224]]:
        Xmin BC_t 'FamilySpecified':
          GridLocation GridLocation_t 'FaceCenter':
          FamilyName FamilyName_t 'INLET':
          PointList IndexArray_t I{int_type} [[11, 23, 34, 46, 64, 76, 86, 98, 116, 128, 138, 150, 168, 178, 189, 201]]:
        Xmax BC_t 'FamilySpecified':
          GridLocation GridLocation_t 'FaceCenter':
          FamilyName FamilyName_t 'OUTLET':
          PointList IndexArray_t I{int_type} [[20, 32, 43, 56, 73, 84, 95, 108, 125, 136, 147, 160, 177, 188, 199, 212]]:
        Ymax BC_t 'FamilySpecified':
          GridLocation GridLocation_t 'FaceCenter':
          FamilyName FamilyName_t 'FARFIELD':
          PointList IndexArray_t I{int_type} [[52, 55, 58, 60, 104, 107, 110, 112, 156, 159, 162, 164, 208, 211, 214, 216]]:
      NGonElements Elements_t I4 [22, 0]:
        ElementRange IndexRange_t I{int_type} [1, 224]:
        ElementStartOffset DataArray_t:
          I{int_type} : [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 43, 47, 50, 54, 57, 61, 64, 68, 72, 75, 79, 83, 87, 91, 95, 99,
                103, 107, 111, 115, 119, 123, 127, 131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179, 183,
                187, 191, 195, 199, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239, 243, 247, 250, 254, 257, 260, 264,
                268, 271, 275, 279, 282, 286, 290, 294, 298, 302, 306, 310, 314, 318, 322, 326, 330, 334, 338, 342, 346,
                350, 354, 358, 362, 366, 370, 374, 378, 382, 386, 390, 394, 398, 402, 406, 410, 414, 418, 422, 426, 430,
                434, 438, 442, 446, 450, 453, 457, 460, 463, 467, 471, 474, 478, 482, 485, 489, 493, 497, 501, 505, 509,
                513, 517, 521, 525, 529, 533, 537, 541, 545, 549, 553, 557, 561, 565, 569, 573, 577, 581, 585, 589, 593,
                597, 601, 605, 609, 613, 617, 621, 625, 629, 633, 637, 641, 645, 649, 653, 656, 660, 663, 666, 670, 674,
                677, 681, 685, 688, 692, 696, 700, 704, 708, 712, 716, 720, 724, 728, 732, 736, 740, 744, 748, 752, 756,
                760, 764, 768, 772, 776, 780, 784, 788, 792, 796, 800, 804, 808, 812, 816, 820, 824, 828, 832, 836, 840,
                844, 848, 852, 856, 860, 864, 868, 872, 876]
        ElementConnectivity DataArray_t:
          I{int_type} : [1, 2, 82, 81, 2, 3, 83, 82, 84, 83, 3, 4, 4, 5, 85, 84, 6, 7, 2, 1, 3, 2, 7, 8, 8, 9, 4, 3, 5, 4, 9, 10,
                7, 6, 11, 12, 12, 13, 8, 7, 81, 21, 1, 13, 14, 9, 8, 2, 22, 82, 14, 15, 10, 9, 3, 23, 83, 21, 22, 2, 1, 4,
                24, 84, 16, 17, 12, 11, 22, 23, 3, 2, 5, 25, 85, 13, 12, 17, 18, 23, 24, 4, 3, 21, 26, 6, 1, 14, 13, 18,
                19, 24, 25, 5, 4, 7, 27, 22, 2, 15, 14, 19, 20, 8, 28, 23, 3, 26, 27, 7, 6, 9, 29, 24, 4, 27, 28, 8, 7, 10,
                30, 25, 5, 28, 29, 9, 8, 26, 31, 11, 6, 29, 30, 10, 9, 12, 32, 27, 7, 13, 33, 28, 8, 22, 21, 81, 82, 31,
                32, 12, 11, 14, 34, 29, 9, 23, 22, 82, 83, 32, 33, 13, 12, 15, 35, 30, 10, 24, 23, 83, 84, 33, 34, 14, 13,
                31, 36, 16, 11, 25, 24, 84, 85, 34, 35, 15, 14, 17, 37, 32, 12, 18, 38, 33, 13, 27, 26, 21, 22, 36, 37, 17,
                16, 19, 39, 34, 14, 28, 27, 22, 23, 37, 38, 18, 17, 20, 40, 35, 15, 29, 28, 23, 24, 38, 39, 19, 18, 30, 29,
                24, 25, 39, 40, 20, 19, 32, 31, 26, 27, 33, 32, 27, 28, 34, 33, 28, 29, 81, 41, 21, 35, 34, 29, 30, 22, 42,
                82, 23, 43, 83, 37, 36, 31, 32, 41, 42, 22, 21, 24, 44, 84, 38, 37, 32, 33, 42, 43, 23, 22, 25, 45, 85, 39,
                38, 33, 34, 43, 44, 24, 23, 41, 46, 26, 21, 40, 39, 34, 35, 44, 45, 25, 24, 27, 47, 42, 22, 28, 48, 43, 23,
                46, 47, 27, 26, 29, 49, 44, 24, 47, 48, 28, 27, 30, 50, 45, 25, 48, 49, 29, 28, 46, 51, 31, 26, 49, 50, 30,
                29, 32, 52, 47, 27, 33, 53, 48, 28, 42, 41, 81, 82, 51, 52, 32, 31, 34, 54, 49, 29, 43, 42, 82, 83, 52, 53,
                33, 32, 35, 55, 50, 30, 44, 43, 83, 84, 53, 54, 34, 33, 51, 56, 36, 31, 45, 44, 84, 85, 54, 55, 35, 34, 37,
                57, 52, 32, 38, 58, 53, 33, 47, 46, 41, 42, 56, 57, 37, 36, 39, 59, 54, 34, 48, 47, 42, 43, 57, 58, 38, 37,
                40, 60, 55, 35, 49, 48, 43, 44, 58, 59, 39, 38, 50, 49, 44, 45, 59, 60, 40, 39, 52, 51, 46, 47, 53, 52, 47,
                48, 54, 53, 48, 49, 81, 61, 41, 55, 54, 49, 50, 42, 62, 82, 43, 63, 83, 57, 56, 51, 52, 61, 62, 42, 41, 44,
                64, 84, 58, 57, 52, 53, 62, 63, 43, 42, 45, 65, 85, 59, 58, 53, 54, 63, 64, 44, 43, 61, 66, 46, 41, 60, 59,
                54, 55, 64, 65, 45, 44, 47, 67, 62, 42, 48, 68, 63, 43, 66, 67, 47, 46, 49, 69, 64, 44, 67, 68, 48, 47, 50,
                70, 65, 45, 68, 69, 49, 48, 66, 71, 51, 46, 69, 70, 50, 49, 52, 72, 67, 47, 53, 73, 68, 48, 62, 61, 81, 82,
                71, 72, 52, 51, 54, 74, 69, 49, 63, 62, 82, 83, 72, 73, 53, 52, 55, 75, 70, 50, 64, 63, 83, 84, 73, 74, 54,
                53, 56, 51, 71, 76, 65, 64, 84, 85, 74, 75, 55, 54, 57, 77, 72, 52, 58, 78, 73, 53, 67, 66, 61, 62, 76, 77,
                57, 56, 59, 79, 74, 54, 68, 67, 62, 63, 77, 78, 58, 57, 60, 80, 75, 55, 69, 68, 63, 64, 78, 79, 59, 58, 70,
                69, 64, 65, 79, 80, 60, 59, 72, 71, 66, 67, 73, 72, 67, 68, 74, 73, 68, 69, 61, 81, 86, 75, 74, 69, 70, 62,
                87, 82, 63, 88, 83, 77, 76, 71, 72, 86, 87, 62, 61, 64, 89, 84, 78, 77, 72, 73, 87, 88, 63, 62, 65, 90, 85,
                66, 61, 86, 91, 79, 78, 73, 74, 88, 89, 64, 63, 80, 79, 74, 75, 89, 90, 65, 64, 67, 92, 87, 62, 68, 93, 88,
                63, 91, 92, 67, 66, 69, 94, 89, 64, 92, 93, 68, 67, 70, 95, 90, 65, 71, 66, 91, 96, 93, 94, 69, 68, 94, 95,
                70, 69, 72, 97, 92, 67, 73, 98, 93, 68, 81, 82, 87, 86, 96, 97, 72, 71, 74, 99, 94, 69, 82, 83, 88, 87, 97,
                98, 73, 72, 75, 100, 95, 70, 83, 84, 89, 88, 76, 71, 96, 101, 98, 99, 74, 73, 84, 85, 90, 89, 99, 100, 75,
                74, 77, 102, 97, 72, 78, 103, 98, 73, 86, 87, 92, 91, 101, 102, 77, 76, 79, 104, 99, 74, 87, 88, 93, 92,
                102, 103, 78, 77, 80, 105, 100, 75, 88, 89, 94, 93, 103, 104, 79, 78, 89, 90, 95, 94, 104, 105, 80, 79, 91,
                92, 97, 96, 92, 93, 98, 97, 93, 94, 99, 98, 94, 95, 100, 99, 96, 97, 102, 101, 97, 98, 103, 102, 98, 99,
                104, 103, 99, 100, 105, 104]
        ParentElements DataArray_t:
          I{int_type} : [[225, 0], [226, 0], [227, 0], [228, 0], [229, 0], [230, 0], [231, 0], [232, 0], [233, 0], [234, 0], [225,
                0], [235, 0], [225, 226], [236, 0], [226, 227], [225, 229], [227, 228], [237, 0], [226, 230], [228, 0], [238,
                0], [227, 231], [229, 0], [239, 0], [228, 232], [229, 230], [240, 0], [230, 231], [229, 233], [231, 232],
                [230, 234], [232, 0], [231, 235], [233, 0], [232, 236], [233, 234], [234, 235], [225, 241], [233, 237], [235,
                236], [226, 242], [234, 238], [236, 0], [227, 243], [235, 239], [237, 0], [228, 244], [236, 240], [237, 238],
                [238, 239], [229, 245], [237, 0], [239, 240], [230, 246], [238, 0], [240, 0], [231, 247], [239, 0], [232,
                248], [240, 0], [233, 249], [234, 250], [235, 251], [241, 0], [236, 252], [241, 242], [242, 243], [237, 253],
                [241, 245], [243, 244], [238, 254], [242, 246], [244, 0], [239, 255], [243, 247], [245, 0], [240, 256], [244,
                248], [245, 246], [246, 247], [245, 249], [247, 248], [246, 250], [248, 0], [247, 251], [249, 0], [248, 252],
                [249, 250], [250, 251], [241, 257], [249, 253], [251, 252], [242, 258], [250, 254], [252, 0], [243, 259],
                [251, 255], [253, 0], [244, 260], [252, 256], [253, 254], [254, 255], [245, 261], [253, 0], [255, 256], [246,
                262], [254, 0], [256, 0], [247, 263], [255, 0], [248, 264], [256, 0], [249, 265], [250, 266], [251, 267],
                [257, 0], [252, 268], [257, 258], [258, 259], [253, 269], [257, 261], [259, 260], [254, 270], [258, 262],
                [260, 0], [255, 271], [259, 263], [261, 0], [256, 272], [260, 264], [261, 262], [262, 263], [261, 265], [263,
                264], [262, 266], [264, 0], [263, 267], [265, 0], [264, 268], [265, 266], [266, 267], [257, 273], [265, 269],
                [267, 268], [258, 274], [266, 270], [268, 0], [259, 275], [267, 271], [269, 0], [260, 276], [268, 272], [269,
                270], [270, 271], [261, 277], [269, 0], [271, 272], [262, 278], [270, 0], [272, 0], [263, 279], [271, 0],
                [264, 280], [272, 0], [265, 281], [266, 282], [267, 283], [273, 0], [268, 284], [273, 274], [274, 275], [269,
                285], [273, 277], [275, 276], [270, 286], [274, 278], [276, 0], [277, 0], [271, 287], [275, 279], [272, 288],
                [276, 280], [277, 278], [278, 279], [277, 281], [279, 280], [278, 282], [280, 0], [281, 0], [279, 283], [280,
                284], [281, 282], [282, 283], [273, 0], [281, 285], [283, 284], [274, 0], [282, 286], [284, 0], [275, 0],
                [285, 0], [283, 287], [276, 0], [284, 288], [285, 286], [286, 287], [277, 0], [285, 0], [287, 288], [278,
                0], [286, 0], [288, 0], [279, 0], [287, 0], [280, 0], [288, 0], [281, 0], [282, 0], [283, 0], [284, 0], [285,
                0], [286, 0], [287, 0], [288, 0]]
'''
      # ZSR_Data2 ZoneSubRegion_t:
      #   GridLocation GridLocation_t 'FaceCenter':
      #   PointList IndexArray_t I{int_type} [[52]]:
      #   Data2 DataArray_t I{int_type} [2]:

yaml_ref_zgc = f'''
      ZoneGridConnectivity ZoneGridConnectivity_t:
        Xmin_0 GridConnectivity_t 'Base/zone':
          GridConnectivityType GridConnectivityType_t 'Abutting1to1':
          GridLocation GridLocation_t 'FaceCenter':
          PointList IndexArray_t I{int_type} [[11, 23, 34, 46, 64, 76, 86, 98, 116, 128, 138, 150, 168, 178, 189, 201]]:
          PointListDonor IndexArray_t I{int_type} [[20, 32, 43, 56, 73, 84, 95, 108, 125, 136, 147, 160, 177, 188, 199, 212]]:
          GridConnectivityProperty GridConnectivityProperty_t:
            Periodic Periodic_t:
              RotationAngle DataArray_t R4 [0, 0, 0]:
              RotationCenter DataArray_t R4 [0, 0, 0]:
              Translation DataArray_t R4 [4, 0, 0]:
          GridConnectivityDonorName Descriptor_t 'Xmax_0':
          FamilyName FamilyName_t 'INLET':
        Xmax_0 GridConnectivity_t 'Base/zone':
          GridConnectivityType GridConnectivityType_t 'Abutting1to1':
          GridLocation GridLocation_t 'FaceCenter':
          PointList IndexArray_t I{int_type} [[20, 32, 43, 56, 73, 84, 95, 108, 125, 136, 147, 160, 177, 188, 199, 212]]:
          PointListDonor IndexArray_t I{int_type} [[11, 23, 34, 46, 64, 76, 86, 98, 116, 128, 138, 150, 168, 178, 189, 201]]:
          GridConnectivityProperty GridConnectivityProperty_t:
            Periodic Periodic_t:
              RotationAngle DataArray_t R4 [-0, -0, -0]:
              RotationCenter DataArray_t R4 [0, 0, 0]:
              Translation DataArray_t R4 [-4, -0, -0]:
          GridConnectivityDonorName Descriptor_t 'Xmin_0':
          FamilyName FamilyName_t 'OUTLET':
'''

@pytest_parallel.mark.parallel([1,2,7,11,23,59])
# @pytest_parallel.mark.parallel([1,2,3])
@pytest.mark.parametrize("ZSR", [False, True])
@pytest.mark.parametrize("JN", [False, True])
def test_merge_degen_faces(ZSR,JN,comm):
  #----------------------------
  # Ref yaml
  
  
  #----------------------------
  # Parameters
  nx = 5
  ny = 5
  nz = 5
  sector_angle = np.pi/4
  
  fam_l = ['INLET', 'OUTLET', 'AXIS', 'FARFIELD', 'PER1', 'PER2']
  bc2fam = {}
  bc2fam['Xmin'] = fam_l[0]
  bc2fam['Xmax'] = fam_l[1]
  bc2fam['Ymin'] = fam_l[2]
  bc2fam['Ymax'] = fam_l[3]
  bc2fam['Zmin'] = fam_l[4]
  bc2fam['Zmax'] = fam_l[5]
  
  #----------------------------
  # Generate cube
  dist_tree = maia.factory.generate_dist_block([nx,ny,nz], 'HEXA_8', comm)
  maia.algo.scale_mesh(dist_tree, [nx-1,ny-1,nz-1])
  
  #----------------------------
  # Add families
  base_n = PT.get_all_CGNSBase_t(dist_tree)[0] 
  for bc_n in PT.get_nodes_from_predicates(dist_tree, 'CGNSBase_t/Zone_t/ZoneBC_t/BC_t'):
      PT.set_value(bc_n, 'FamilySpecified')
      PT.update_child(bc_n,'FamilyName',label='FamilyName_t',value=bc2fam[bc_n[0]])
  
  for fam in fam_l:
      PT.update_child(base_n,fam,label='Family_t')
  
  #----------------------------
  # Add families
  zone_n = PT.get_node_from_predicates(dist_tree, 'CGNSBase_t/Zone_t')
  ymin_n = PT.get_node_from_predicates(zone_n, 'ZoneBC_t/Ymin')
  pl_ymin = PT.get_value(PT.get_node_from_name(ymin_n, 'PointList'))[0]
  ymax_n = PT.get_node_from_predicates(zone_n, 'ZoneBC_t/Ymax')
  pl_ymax = PT.get_value(PT.get_node_from_name(ymax_n, 'PointList'))[0]
  if comm.rank == 0:
    pl1 = np.array([[pl_ymin[0]]], dtype=pdm_gnum_dtype)
    data1 = np.array([1], dtype=pdm_gnum_dtype)
    zsr_data1 = PT.new_ZoneSubRegion(name='ZSR_Data1', loc='FaceCenter', point_list=pl1, fields = {'Data1': data1}, parent=zone_n)
    cgns_dist1 = PT.new_UserDefinedData(name=':CGNS#Distribution', parent=zsr_data1)
    PT.new_DataArray('Index', [0,1,1], parent=cgns_dist1)
    pl2 = np.array([[pl_ymin[0], pl_ymax[0]]], dtype=pdm_gnum_dtype)
    data2 = np.array([1,2], dtype=pdm_gnum_dtype)
    zsr_data2 = PT.new_ZoneSubRegion(name='ZSR_Data2', loc='FaceCenter', point_list=pl2, fields = {'Data2': data2}, parent=zone_n)
    cgns_dist2 = PT.new_UserDefinedData(name=':CGNS#Distribution', parent=zsr_data2)
    PT.new_DataArray('Index', [0,2,2], parent=cgns_dist2)
  else:
    pl1 = np.array([[]], dtype=pdm_gnum_dtype)
    data1 = np.array([], dtype=pdm_gnum_dtype)
    zsr_data1 = PT.new_ZoneSubRegion(name='ZSR_Data1', loc='FaceCenter', point_list=pl1, fields = {'Data1': data1}, parent=zone_n)
    cgns_dist1 = PT.new_UserDefinedData(name=':CGNS#Distribution', parent=zsr_data1)
    PT.new_DataArray('Index', [1,1,1], parent=cgns_dist1)
    pl2 = np.array([[]], dtype=pdm_gnum_dtype)
    data2 = np.array([], dtype=pdm_gnum_dtype)
    zsr_data2 = PT.new_ZoneSubRegion(name='ZSR_Data2', loc='FaceCenter', point_list=pl2, fields = {'Data2': data2}, parent=zone_n)
    cgns_dist2 = PT.new_UserDefinedData(name=':CGNS#Distribution', parent=zsr_data2)
    PT.new_DataArray('Index', [2,2,2], parent=cgns_dist2)
  
  #----------------------------
  # Prepare test case with ZSR
  if ZSR:
    PT.rm_nodes_from_name(ymin_n, 'FamilyName')
    PT.new_ZoneSubRegion(name='ZSR_Ymin', loc='FaceCenter', bc_name='Ymin', family=bc2fam['Ymin'], parent=zone_n)
  
  #----------------------------
  # Prepare test case with join
  if JN:
    periodic = {'translation' : np.array([nx-1, 0, 0], np.float32)}
    maia.algo.dist.connect_1to1_families(dist_tree, ('INLET', 'OUTLET'), comm, periodic=periodic)
  
  #----------------------------
  # Move nodes to generate sector of cylinder
  theta_x = sector_angle/(nz-1)
  theta_y = 0.
  theta_z = 0.
  
  zone_n = PT.get_node_from_label(dist_tree, 'Zone_t')
  vtx_distri = PT.get_value(PT.maia.getDistribution(zone_n, 'Vertex'))
  
  coords_n = PT.get_node_from_label(dist_tree, 'GridCoordinates_t')
  coord_x_n, coord_y_n, coord_z_n = [PT.get_node_from_name(coords_n, f"Coordinate{suffix}") for suffix in ['X', 'Y', 'Z']]
  coord_x,   coord_y,   coord_z   = [PT.get_value(coord) for coord in [coord_x_n, coord_y_n, coord_z_n]]
  
  multiple, remainder = np.divmod(vtx_distri, nx*ny)
  assert remainder[2] == 0
  
  start = max(multiple[0], 1) # max because the first plane does not move
  stop  = multiple[1] if remainder[1] == 0 else multiple[1] + 1
  
  for i in np.arange(start,stop):
    beg = max(i*nx*ny, vtx_distri[0]) - vtx_distri[0]
    end = min((i+1)*nx*ny, vtx_distri[1]) - vtx_distri[0]
    coord_z[beg:end] = 0.
    coord_x[beg:end], coord_y[beg:end], coord_z[beg:end] = maia.utils.ndarray.np_utils.transform_cart_vectors(coord_x[beg:end], coord_y[beg:end], coord_z[beg:end], rotation_angle = np.array([i*theta_x,theta_y,theta_z]))
  
  #----------------------------
  # Convert to ngon
  # When convert_elements_to_ngon will be parallel independent, this part
  # could be reduce to
  # maia.algo.dist.convert_elements_to_ngon(dist_tree, comm)
  maia.algo.dist.redistribute_tree(dist_tree, 'gather', comm)
  group = comm.Get_group()
  newGroup = group.Incl([0])
  sub_comm  = comm.Create(newGroup)
  # if comm.rank==22:
  #   PT.print_tree(dist_tree, 'dist_tree.txt')
  # exit()
  if comm.rank == 0:
    # print(PT.get_node_from_name(PT.get_node_from_name(dist_tree, 'Ymin'), 'PointList')[1][0])
    # print(PT.get_node_from_name(PT.get_node_from_name(dist_tree, 'ZSR_Data1'), 'PointList')[1][0])
    maia.algo.dist.convert_elements_to_ngon(dist_tree, sub_comm)
    # print(PT.get_node_from_name(PT.get_node_from_name(dist_tree, 'Ymin'), 'PointList')[1][0])
    # print(PT.get_node_from_name(PT.get_node_from_name(dist_tree, 'ZSR_Data1'), 'PointList')[1][0])
    # fix bug in convert_elements_to_ngon
    PT.set_value(PT.get_node_from_name(PT.get_node_from_name(dist_tree, 'ZSR_Data1'), 'PointList'), 
                [PT.get_node_from_name(PT.get_node_from_name(dist_tree, 'Ymin'), 'PointList')[1][0][0]])
    PT.set_value(PT.get_node_from_name(PT.get_node_from_name(dist_tree, 'ZSR_Data2'), 'PointList'), 
                [[PT.get_node_from_name(PT.get_node_from_name(dist_tree, 'Ymin'), 'PointList')[1][0][0],
                  PT.get_node_from_name(PT.get_node_from_name(dist_tree, 'Ymax'), 'PointList')[1][0][0]]])
    full_tree = maia.factory.dist_to_full_tree(dist_tree, sub_comm, target=0)
  else:
    full_tree = None
  dist_tree = maia.factory.full_to_dist_tree(full_tree, comm, owner=0)
  
  #-------------
  # Test
  fam_to_remove        = 'AXIS'
  fam_for_intersection = 'PER2'
  MDB.delete_degen_faces_from_family(dist_tree, fam_to_remove, fam_for_intersection, comm)
  
  #----------------------------
  # Prepare result with ZSR
  if ZSR:
    PT.rm_nodes_from_name(dist_tree, 'Ymin')
  PT.rm_nodes_from_name(dist_tree, 'ZSR_Data1')
  PT.rm_nodes_from_name(dist_tree, 'ZSR_Data2')
  
  #----------------------------
  # To be sure to have the same distribution with reference
  maia.algo.dist.redistribute_tree(dist_tree, 'uniform', comm)
  
  #----------------------------
  # Prepare reference
  if comm.rank == 0:
    ref_full_tree = PT.yaml.to_cgns_tree(yaml_ref)
    if JN:
      ref_zgc = PT.yaml.to_node(yaml_ref_zgc)
      ref_zone = PT.get_node_from_label(ref_full_tree, 'Zone_t')
      PT.add_child(ref_zone, ref_zgc)
      PT.rm_nodes_from_name(ref_full_tree, 'Xmin')
      PT.rm_nodes_from_name(ref_full_tree, 'Xmax')
  else:
    ref_full_tree = None
  ref_dist_tree = maia.factory.full_to_dist_tree(ref_full_tree, comm, owner=0)
  for zone_n in PT.get_nodes_from_label(ref_dist_tree, 'Zone_t'):
    if not PT.Zone.has_nface_elements(zone_n):
      maia.algo.pe_to_nface(zone_n, comm)
  
  #----------------------------
  # Assertion test
  assert maia.pytree.is_same_tree(dist_tree, ref_dist_tree)
  
