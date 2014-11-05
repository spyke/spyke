import numpy as np
import pyximport
pyximport.install(build_in_temp=False, inplace=True)
import util # .pyx file

import time

'''
"""Median tests"""
data = np.int16(np.random.randint(-2**15, 2**15-1, 52*(500e3+1)))
data.shape = 52, 500e3+1
print('data: %r' % data)

tmed = time.time()
mydata = np.abs(data) # does a copy
print('copy took %.3f sec' % (time.time()-tmed))
result = np.median(mydata, axis=-1) # doesn't modify data, does internal copy
print('np.median took %.3f sec' % (time.time()-tmed))
print('np.median: %r' % np.int16(result))
print('mydata: %r' % mydata)

tcy = time.time()
mydata = np.abs(data) # does a copy
print('copy took %.3f sec' % (time.time()-tcy))
result = util.median_inplace_2Dshort(mydata)
print('Cython median took %.3f sec' % (time.time()-tcy))
print('Cython median: %r' % result)
print('mydata: %r' % mydata)

tcy = time.time()
mydata = np.abs(data) # does a copy
print('copy took %.3f sec' % (time.time()-tcy))
result = util.mean_2Dshort(mydata)
print('Cython mean took %.3f sec' % (time.time()-tcy))
print('Cython mean: %r' % result)
print('mydata: %r' % mydata)
'''
'''
"""Test sharpness2D"""
signal = np.array([[-3022, -3031, -2423, -1655, -1108,  -864,  -891,  -994,  -908,  -600,  -343,  -304,  -219,    89,   377,   342,   136,    74,   112,    -8,
         -391,  -968, -1691, -2397, -2808, -2729, -2269, -1671, -1099,  -623,  -300,   -82,   235,   652,   804,   566,   308,   453,   817,   893,
          418,  -274,  -609,  -295,   397,   939,  1104,   958,   611,   285],
       [-1232, -1042,  -905,  -799,  -552,   -53,   456,   721,   740,   674,   476,   167,    25,   280,   798,  1159,  1167,   964,   914,  1182,
         1598,  1742,  1390,   581,  -442, -1371, -1896, -1973, -1851, -1804, -1811, -1742, -1585, -1434, -1296, -1117,  -929,  -780,  -681,  -607,
         -619,  -667,  -596,  -286,   136,   483,   716,   890,   932,   862],
       [ -401,  -653,  -821, -1025, -1261, -1391, -1312, -1121,  -985,  -849,  -655,  -501,  -554,  -653,  -561,  -249,    68,   166,    59,  -130,
         -267,  -448,  -749, -1087, -1266, -1338, -1330, -1055,  -378,   305,   577,   509,   532,   786,  1056,  1132,   997,   762,   614,   704,
         1005,  1309,  1369,  1148,   731,   341,    25,  -282,  -610,  -687],
       [ -945,  -951,  -804,  -566,  -471,  -558,  -589,  -567,  -582,  -685,  -791,  -858,  -918,  -882,  -690,  -247,   333,   831,  1064,  1145,
         1209,  1089,   537,  -374, -1252, -1884, -2228, -2237, -1889, -1500, -1328, -1194,  -771,  -137,   299,   352,   225,   164,   318,   600,
          912,  1034,   927,   697,   488,   438,   475,   410,   196,   159],
       [ -216,  -246,  -482,  -831, -1043, -1104, -1104, -1156, -1214, -1183, -1129, -1161, -1288, -1049,  -261,   872,  1871,  2502,  2701,  2317,
         1153,  -885, -3352, -5442, -6487, -6525, -5820, -4547, -2827, -1071,   322,  1236,  1815,  2226,  2521,  2609,  2583,  2650,  2902,  2972,
         2675,  2186,  1899,  1872,  1813,  1460,   837,   386,   351,   606],
       [ -167,  -400,  -549,  -656,  -792, -1008, -1274, -1488, -1510, -1216,  -807,  -624,  -811,  -976,  -779,  -336,    35,   240,   272,  -112,
         -900, -1712, -2101, -2144, -2049, -1872, -1467,  -848,   -69,   800,  1644,  2224,  2470,  2528,  2562,  2464,  2227,  2080,  2115,  1984,
         1530,  1136,  1218,  1600,  1773,  1552,  1071,   624,   338,   256],
       [ -375,  -308,  -326,  -503,  -683,  -798,  -799,  -667,  -410,  -154,   -65,  -226,  -489,  -461,   -29,   506,   779,   845,   954,  1114,
         1110,   792,   179,  -586, -1304, -1793, -1904, -1564,  -947,  -478,  -394,  -464,  -382,  -184,  -129,  -294,  -454,  -321,    85,   378,
          297,    11,   -47,   266,   645,   653,   184,  -364,  -580,  -474]], dtype=np.int16)

#sharp = util.sharpness2D(signal)

# nice test cases
sig0 = np.array([[-500,  500, -500,    100,  500,    -100]], dtype=np.int16)
sig1 = np.array([[-500,  500, -500,    100,  500,     100]], dtype=np.int16)
sig2 = np.array([[-1006,  -960,  -595,  -173,   -95,  -481,  -983, -1152,  -929,  -618,  -579,  -791,  -791,  -571,  -473,  -651,  -452,   631,  2109,  2986, 2495,   580, -2361, -5422, -7663, -8282, -7084, -4519, -1590,   869,  2571,  3696,  4481,  4996,  5205,  5035,  4459,  3715,  3086,  2685, 2289,  1748,  1214,   849,   632,   415,   254,   103,  -151,  -592]], dtype=np.int16)

sharp = util.sharpness2D(sig2)

# for testing rowtake_cy:
a = np.arange(20)
a.shape = 5, 4
i = np.array([[2, 1],
              [3, 1],
              [1, 1],
              [0, 0],
              [3, 1]])
'''
a = np.arange(10, dtype=np.uint8)
b = np.arange(1, 11, dtype=np.uint8)
c = np.arange(2, 12, dtype=np.uint8)
l = []
for i in range(100000):
    l.append(a)
    l.append(b)
    l.append(c)
    
t0 = time.time()
print(util.intersect1d_uint8(l))
print('intersectd1d took %.3f sec' % (time.time()-t0)) # best time is around 0.123 sec


'''
# test NDsepmetric:
i = np.row_stack(np.float32(np.random.normal(loc=0, scale=1, size=5000))) # 2D
j = np.row_stack(np.float32(np.random.normal(loc=2, scale=1, size=1000))) # 2D
util.NDsepmetric(i, j, Nmax=20000)
'''
