import math
import numpy as np
import tensorflow as tf

# # BCH(63,45)
Rate = 45 / 63
H = [
    [
        1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0,
        1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0,
        1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0,
        0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1,
        0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0,
        1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0,
        0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
        0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0,
        1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1,
        0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [
        1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1,
        0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [
        1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [
        0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [
        0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [
        1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1,
        1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [
        1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [
        1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
        0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
Hi = [
    [1, 2, 0, 0, 3, 4, 5, 6, 0, 7, 0, 8, 0, 0, 9, 0, 10, 11, 0, 0, 12, 13, 14, 15, 16, 0, 0, 17, 0, 0, 18, 19, 0, 0, 0,
     0, 0, 20, 0, 0, 21, 22, 0, 0, 23, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        25, 0, 26, 0, 27, 0, 0, 0, 28, 29, 30, 31, 32, 0, 33, 34, 35, 0, 36, 0, 37, 0, 0, 0, 0, 38, 0, 39, 40, 0, 41, 0,
        42, 0, 0, 0, 0, 43, 44, 0, 45, 0, 46, 0, 47, 0, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        49, 0, 0, 50, 51, 0, 52, 53, 0, 0, 54, 0, 55, 56, 57, 58, 0, 0, 0, 59, 60, 0, 61, 62, 63, 0, 64, 65, 66, 67, 68,
        0, 0, 69, 0, 0, 0, 70, 71, 72, 73, 0, 0, 74, 75, 0, 0, 76, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        77, 0, 0, 0, 0, 0, 78, 0, 79, 80, 0, 0, 0, 81, 0, 82, 0, 83, 0, 0, 0, 0, 84, 0, 0, 85, 0, 0, 86, 87, 0, 0, 0, 0,
        88, 0, 0, 89, 90, 91, 0, 0, 0, 0, 0, 0, 0, 0, 92, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        0, 93, 0, 0, 0, 0, 0, 94, 0, 95, 96, 0, 0, 0, 97, 0, 98, 0, 99, 0, 0, 0, 0, 100, 0, 0, 101, 0, 0, 102, 103, 0,
        0, 0, 0, 104, 0, 0, 105, 106, 107, 0, 0, 0, 0, 0, 0, 0, 0, 108, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        0, 0, 109, 0, 0, 0, 0, 0, 110, 0, 111, 112, 0, 0, 0, 113, 0, 114, 0, 115, 0, 0, 0, 0, 116, 0, 0, 117, 0, 0, 118,
        119, 0, 0, 0, 0, 120, 0, 0, 121, 122, 123, 0, 0, 0, 0, 0, 0, 0, 0, 124, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        0, 0, 0, 125, 0, 0, 0, 0, 0, 126, 0, 127, 128, 0, 0, 0, 129, 0, 130, 0, 131, 0, 0, 0, 0, 132, 0, 0, 133, 0, 0,
        134, 135, 0, 0, 0, 0, 136, 0, 0, 137, 138, 139, 0, 0, 0, 0, 0, 0, 0, 0, 140, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        0, 0, 0, 0, 141, 0, 0, 0, 0, 0, 142, 0, 143, 144, 0, 0, 0, 145, 0, 146, 0, 147, 0, 0, 0, 0, 148, 0, 0, 149, 0,
        0, 150, 151, 0, 0, 0, 0, 152, 0, 0, 153, 154, 155, 0, 0, 0, 0, 0, 0, 0, 0, 156, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        0, 0, 0, 0, 0, 157, 0, 0, 0, 0, 0, 158, 0, 159, 160, 0, 0, 0, 161, 0, 162, 0, 163, 0, 0, 0, 0, 164, 0, 0, 165,
        0, 0, 166, 167, 0, 0, 0, 0, 168, 0, 0, 169, 170, 171, 0, 0, 0, 0, 0, 0, 0, 0, 172, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        173, 174, 0, 0, 175, 176, 0, 177, 0, 178, 0, 179, 180, 0, 0, 181, 182, 183, 0, 184, 185, 0, 186, 0, 187, 0, 0,
        188, 189, 0, 190, 0, 0, 0, 191, 192, 0, 193, 0, 0, 0, 194, 0, 195, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 196, 0, 0, 0,
        0, 0, 0, 0, 0],
    [
        0, 197, 198, 0, 0, 199, 200, 0, 201, 0, 202, 0, 203, 204, 0, 0, 205, 206, 207, 0, 208, 209, 0, 210, 0, 211, 0,
        0, 212, 213, 0, 214, 0, 0, 0, 215, 216, 0, 217, 0, 0, 0, 218, 0, 219, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 220, 0, 0,
        0, 0, 0, 0, 0],
    [
        221, 222, 223, 224, 225, 226, 0, 0, 0, 0, 0, 0, 0, 227, 0, 0, 228, 0, 229, 230, 231, 0, 0, 232, 0, 0, 233, 234,
        0, 235, 0, 236, 237, 0, 0, 0, 238, 0, 0, 239, 240, 241, 0, 242, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 244, 0, 0,
        0, 0, 0, 0],
    [
        245, 0, 246, 247, 0, 0, 0, 248, 0, 249, 0, 250, 0, 0, 0, 0, 251, 0, 0, 252, 0, 0, 253, 254, 0, 0, 0, 0, 255, 0,
        0, 256, 257, 258, 0, 0, 0, 0, 0, 0, 0, 0, 259, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 260, 0, 0, 0, 0, 0],
    [
        0, 261, 0, 262, 263, 0, 0, 0, 264, 0, 265, 0, 266, 0, 0, 0, 0, 267, 0, 0, 268, 0, 0, 269, 270, 0, 0, 0, 0, 271,
        0, 0, 272, 273, 274, 0, 0, 0, 0, 0, 0, 0, 0, 275, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 276, 0, 0, 0, 0],
    [
        0, 0, 277, 0, 278, 279, 0, 0, 0, 280, 0, 281, 0, 282, 0, 0, 0, 0, 283, 0, 0, 284, 0, 0, 285, 286, 0, 0, 0, 0,
        287, 0, 0, 288, 289, 290, 0, 0, 0, 0, 0, 0, 0, 0, 291, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 292, 0, 0, 0],
    [
        293, 294, 0, 295, 296, 0, 0, 297, 0, 298, 299, 300, 301, 0, 0, 0, 302, 303, 0, 304, 305, 306, 0, 307, 308, 309,
        310, 311, 0, 0, 312, 0, 0, 0, 313, 314, 315, 316, 0, 0, 317, 318, 0, 0, 319, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 320, 0, 0],
    [
        321, 0, 322, 0, 0, 0, 323, 324, 325, 326, 327, 0, 328, 329, 330, 0, 331, 0, 332, 0, 0, 0, 0, 333, 0, 334, 335,
        0, 336, 0, 337, 0, 0, 0, 0, 338, 339, 0, 340, 0, 341, 0, 342, 0, 343, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 344, 0],
    [
        345, 0, 0, 346, 347, 348, 349, 0, 350, 0, 351, 0, 0, 352, 0, 353, 354, 0, 0, 355, 356, 357, 358, 359, 0, 0, 360,
        0, 0, 361, 362, 0, 0, 0, 0, 0, 363, 0, 0, 364, 365, 0, 0, 366, 367, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 368]
]
Hr = [
    [1, 2, 5, 6, 7, 8, 10, 12, 15, 17, 18, 21, 22, 23, 24, 25, 28, 31, 32, 38, 41, 42, 45, 46, 0, 0, 0, 0],
    [1, 3, 5, 9, 10, 11, 12, 13, 15, 16, 17, 19, 21, 26, 28, 29, 31, 33, 38, 39, 41, 43, 45, 47, 0, 0, 0, 0],
    [1, 4, 5, 7, 8, 11, 13, 14, 15, 16, 20, 21, 23, 24, 25, 27, 28, 29, 30, 31, 34, 38, 39, 40, 41, 44, 45, 48],
    [1, 7, 9, 10, 14, 16, 18, 23, 26, 29, 30, 35, 38, 39, 40, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 8, 10, 11, 15, 17, 19, 24, 27, 30, 31, 36, 39, 40, 41, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 9, 11, 12, 16, 18, 20, 25, 28, 31, 32, 37, 40, 41, 42, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 10, 12, 13, 17, 19, 21, 26, 29, 32, 33, 38, 41, 42, 43, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 11, 13, 14, 18, 20, 22, 27, 30, 33, 34, 39, 42, 43, 44, 53, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [6, 12, 14, 15, 19, 21, 23, 28, 31, 34, 35, 40, 43, 44, 45, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 2, 5, 6, 8, 10, 12, 13, 16, 17, 18, 20, 21, 23, 25, 28, 29, 31, 35, 36, 38, 42, 44, 55, 0, 0, 0, 0],
    [2, 3, 6, 7, 9, 11, 13, 14, 17, 18, 19, 21, 22, 24, 26, 29, 30, 32, 36, 37, 39, 43, 45, 56, 0, 0, 0, 0],
    [1, 2, 3, 4, 5, 6, 14, 17, 19, 20, 21, 24, 27, 28, 30, 32, 33, 37, 40, 41, 42, 44, 45, 57, 0, 0, 0, 0],
    [1, 3, 4, 8, 10, 12, 17, 20, 23, 24, 29, 32, 33, 34, 43, 58, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 4, 5, 9, 11, 13, 18, 21, 24, 25, 30, 33, 34, 35, 44, 59, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 5, 6, 10, 12, 14, 19, 22, 25, 26, 31, 34, 35, 36, 45, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 2, 4, 5, 8, 10, 11, 12, 13, 17, 18, 20, 21, 22, 24, 25, 26, 27, 28, 31, 35, 36, 37, 38, 41, 42, 45, 61],
    [1, 3, 7, 8, 9, 10, 11, 13, 14, 15, 17, 19, 24, 26, 27, 29, 31, 36, 37, 39, 41, 43, 45, 62, 0, 0, 0, 0],
    [1, 4, 5, 6, 7, 9, 11, 14, 16, 17, 20, 21, 22, 23, 24, 27, 30, 31, 37, 40, 41, 44, 45, 63, 0, 0, 0, 0]]
Hc = [
    [1, 1, 2, 3, 1, 1, 1, 1, 2, 1, 2, 1, 2, 3, 1, 2, 1, 1, 2, 3, 1, 1, 1, 1, 1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 4, 5, 6, 1,
     2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    [
        2, 5, 6, 7, 2, 9, 3, 3, 4, 2, 3, 2, 3, 4, 2, 3, 2, 4, 5, 6, 2, 8, 3, 3, 3, 4, 5, 2, 3, 4, 2, 6, 7, 8, 9, 10,
        11, 2, 3, 4, 2, 6, 7, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        3, 10, 11, 12, 3, 10, 4, 5, 6, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8, 3, 11, 4, 5, 6, 7, 8, 3, 4, 5, 3, 7, 8, 9, 10,
        11, 12, 3, 4, 5, 3, 7, 8, 9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        4, 11, 12, 13, 8, 11, 11, 10, 11, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 10, 7, 15, 9, 11, 10, 11, 12, 6, 7, 8, 5, 11,
        12, 13, 14, 15, 16, 4, 5, 6, 5, 8, 9, 10, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        10, 12, 13, 14, 10, 12, 17, 13, 14, 7, 8, 9, 10, 11, 9, 10, 10, 10, 11, 12, 9, 16, 10, 12, 14, 15, 16, 9, 10,
        11, 6, 12, 13, 14, 15, 16, 17, 7, 8, 9, 6, 10, 11, 12, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0],
    [
        12, 14, 15, 16, 12, 15, 18, 16, 17, 10, 11, 10, 11, 12, 17, 18, 11, 11, 12, 13, 10, 18, 13, 13, 15, 16, 17,
        10, 11, 12, 9, 13, 14, 15, 16, 17, 18, 10, 11, 12, 7, 12, 13, 14, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0],
    [
        13, 16, 17, 18, 14, 18, 0, 17, 18, 13, 14, 13, 14, 15, 0, 0, 12, 14, 15, 16, 11, 0, 18, 14, 16, 17, 18, 12,
        13, 14, 10, 0, 0, 0, 0, 0, 0, 16, 17, 18, 12, 16, 17, 18, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0],
    [
        16, 0, 0, 0, 15, 0, 0, 0, 0, 15, 16, 15, 16, 17, 0, 0, 13, 16, 17, 18, 12, 0, 0, 16, 0, 0, 0, 16, 17, 18, 15,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        17, 0, 0, 0, 16, 0, 0, 0, 0, 16, 17, 16, 17, 18, 0, 0, 16, 0, 0, 0, 14, 0, 0, 17, 0, 0, 0, 0, 0, 0, 16, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        18, 0, 0, 0, 18, 0, 0, 0, 0, 17, 18, 0, 0, 0, 0, 0, 17, 0, 0, 0, 16, 0, 0, 18, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
for i in range(11):
    for j in range(63):
        Hc[i][j] = Hc[i][j] - 1
for i in range(18):
    for j in range(28):
        Hr[i][j] = Hr[i][j] - 1
for i in range(18):
    for j in range(63):
        Hi[i][j] = Hi[i][j] - 1

EbN0 = 3
EsN0 = EbN0 - 10 * np.log(1 / Rate) / np.log(10)
sigma = math.sqrt(1 / 2 * 10 ** (-EsN0 / 10))
scale = 10000
trans = np.ones((scale, 63))
recei = np.ones((scale, 63))
for i in range(scale):
    recei[i] = trans[i] + sigma * np.random.randn(63)

# # graph # #
x = tf.placeholder(shape=[None, 63], dtype=tf.float32)
y = tf.placeholder(shape=[None, 63], dtype=tf.float32)
ops = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
w1 = tf.Variable(tf.ones(368, dtype=tf.float32))
w2 = tf.Variable(tf.ones(63, dtype=tf.float32))
output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
output1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
outBit = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# check node
for i in range(18):
    for j in range(28):
        if Hr[i][j] != -1:
            ops_temp = 1
            for t in range(28):
                if t != j and Hr[i][t] != -1:
                    ops_temp *= tf.tanh(x[:, Hr[i][t]] / 2)
            ops[0][Hi[i][Hr[i][j]]] = 2 * tf.atanh(ops_temp)
# variable node
for i in range(11):
    for j in range(63):
        if Hc[i][j] != -1:
            ops_temp = w2[j] * x[:, j]
            for t in range(11):
                if t != i and Hc[t][j] != -1:
                    ops_temp += w1[Hi[Hc[t][j]][j]] * ops[0][Hi[Hc[t][j]][j]]
            ops[1][Hi[Hc[i][j]][j]] = tf.tanh(ops_temp / 2)
# check node
for i in range(18):
    for j in range(28):
        if Hr[i][j] != -1:
            ops_temp = 1
            for t in range(28):
                if t != j and Hr[i][t] != -1:
                    ops_temp *= ops[1][Hi[i][Hr[i][t]]]
            ops[2][Hi[i][Hr[i][j]]] = 2 * tf.atanh(ops_temp)
# output layer
for i in range(63):
    output[i] = w2[i] * x[:, i]
for i in range(11):
    for j in range(63):
        if Hc[i][j] != -1:
            output[j] += w1[Hi[Hc[i][j]][j]] * ops[2][Hi[Hc[i][j]][j]]
# loss function
cross_temp = 0
for i in range(63):
    output1[i] = 1 / (1 + math.e ** (-output[i]))
    cross_temp += (y[:, i] + 1) / 2 * tf.log(output1[i] + 1e-10) + (1 - (y[:, i] + 1) / 2) * tf.log(
        1 - output1[i] + 1e-10)
cross_entropy = tf.reduce_mean(cross_temp) / (-63)

# # training # #
learning_rate = 1e-2
# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
loss_vec = []
batch_size = 2000
f1 = open('test.txt', 'w')
for i in range(scale):
    random_index = np.random.choice(scale, size=batch_size)
    T_recei = recei[random_index]
    T_trans = trans[random_index]
    # for j in range(1000):
    sess.run(train_step, feed_dict={x: T_recei, y: T_trans})
    temp_loss = sess.run(cross_entropy, feed_dict={x: T_recei, y: T_trans})
    if i % 50 == 0:
        print('temp_loss=' + str(temp_loss))
        f1.write(str(temp_loss) + '\n')
        # print('w1=' + str(sess.run(w1)) + 'w2=' + str(sess.run(w2)))
        f1.write('w1=' + str(sess.run(w1)) + 'w2=' + str(sess.run(w2)) + '\n')
f1.close()
sess.close()
