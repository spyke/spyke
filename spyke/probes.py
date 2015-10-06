"""Spatial layouts of various electrophys (generally silicon polytrode) probe designs.
Spatial origin is at center top of each probe. Right now, spyke assumes that all
probe designs have contiguous channel indices starting from 0"""

# TODO: add pt16c, and pt16x layouts (with 16 to HS-27 adapter)

from __future__ import division

__authors__ = ['Martin Spacek']

import numpy as np


class Probe(object):
    """self.SiteLoc maps probe chan id to (x, y) position of site in um"""
    def siteloc_arr(self):
        #return np.asarray(self.SiteLoc.values()) # doesn't ensure sorted channel order
        # ensure sorted channel order:
        chans = self.SiteLoc.keys()
        chans.sort()
        return np.asarray([ self.SiteLoc[chan] for chan in chans ])

    def unique_coords(self, axis='x'):
        """Return sorted unique xcoords or ycoords"""
        axis2col = {'x': 0, 'y': 1}
        sa = self.siteloc_arr()
        coords = np.unique(sa[:, axis2col[axis]]) # comes out sorted
        return coords

    def separation(self, axis='x'):
        """Return x or y separation of columns, assuming uniform spacing along axis"""
        coords = self.unique_coords(axis)
        sep = np.unique(np.diff(coords))
        if len(sep) != 1:
            raise ValueError("non-uniform spacing along axis %r" % axis)
            # take mean maybe?
        return sep[0] # pull scalar out of array

    def maxseparation(self, axis='x'):
        """Return maximum site separation along axis"""
        coords = self.unique_coords(axis) # comes out sorted
        return coords[-1] - coords[0]


class uMap54_1a(Probe):
    """uMap54_1a, 65 um spacing, 3 column hexagonal"""
    def __init__(self):
        self.layout = '1a'
        self.name = 'uMap54_1a'
        self.nchans = 54
        self.ncols = 3
        sl = {}
        sl[0] = -56, 1170
        sl[1] = -56, 1105
        sl[2] = -56, 1040
        sl[3] = -56, 975
        sl[4] = -56, 910
        sl[5] = -56, 845
        sl[6] = -56, 585
        sl[7] = -56, 455
        sl[8] = -56, 325
        sl[9] = -56, 195
        sl[10] = -56, 65
        sl[11] = -56, 130
        sl[12] = -56, 260
        sl[13] = -56, 390
        sl[14] = -56, 520
        sl[15] = -56, 650
        sl[16] = -56, 715
        sl[17] = -56, 780
        sl[18] = 0, 1072
        sl[19] = 0, 942
        sl[20] = 0, 812
        sl[21] = 0, 682
        sl[22] = 0, 552
        sl[23] = 0, 422
        sl[24] = 0, 162
        sl[25] = 0, 97
        sl[26] = 0, 292
        sl[27] = 0, 227
        sl[28] = 0, 357
        sl[29] = 0, 487
        sl[30] = 0, 617
        sl[31] = 0, 747
        sl[32] = 0, 877
        sl[33] = 0, 1007
        sl[34] = 0, 1137
        sl[35] = 0, 1202
        sl[36] = 56, 780
        sl[37] = 56, 650
        sl[38] = 56, 520
        sl[39] = 56, 390
        sl[40] = 56, 260
        sl[41] = 56, 130
        sl[42] = 56, 65
        sl[43] = 56, 195
        sl[44] = 56, 325
        sl[45] = 56, 455
        sl[46] = 56, 585
        sl[47] = 56, 715
        sl[48] = 56, 845
        sl[49] = 56, 910
        sl[50] = 56, 975
        sl[51] = 56, 1105
        sl[52] = 56, 1170
        sl[53] = 56, 1040
        assert len(sl) == self.nchans
        self.SiteLoc = sl


class uMap54_1b(Probe):
    """uMap54_1b, 50 um horizontal/43 um vertical spacing, 3 column collinear"""
    def __init__(self):
        self.layout = '1b'
        self.name = 'uMap54_1b'
        self.nchans = 54
        self.ncols = 3
        sl = {}
        sl[0] = -43, 900
        sl[1] = -43, 850
        sl[2] = -43, 800
        sl[3] = -43, 750
        sl[4] = -43, 700
        sl[5] = -43, 650
        sl[6] = -43, 600
        sl[7] = -43, 550
        sl[8] = -43, 500
        sl[9] = -43, 450
        sl[10] = -43, 400
        sl[11] = -43, 350
        sl[12] = -43, 300
        sl[13] = -43, 250
        sl[14] = -43, 200
        sl[15] = -43, 150
        sl[16] = -43, 50
        sl[17] = -43, 100
        sl[18] = 0, 900
        sl[19] = 0, 800
        sl[20] = 0, 700
        sl[21] = 0, 600
        sl[22] = 0, 500
        sl[23] = 0, 400
        sl[24] = 0, 200
        sl[25] = 0, 100
        sl[26] = 0, 300
        sl[27] = 0, 50
        sl[28] = 0, 150
        sl[29] = 0, 250
        sl[30] = 0, 350
        sl[31] = 0, 450
        sl[32] = 0, 550
        sl[33] = 0, 650
        sl[34] = 0, 750
        sl[35] = 0, 850
        sl[36] = 43, 200
        sl[37] = 43, 100
        sl[38] = 43, 50
        sl[39] = 43, 150
        sl[40] = 43, 250
        sl[41] = 43, 300
        sl[42] = 43, 350
        sl[43] = 43, 400
        sl[44] = 43, 450
        sl[45] = 43, 500
        sl[46] = 43, 550
        sl[47] = 43, 600
        sl[48] = 43, 650
        sl[49] = 43, 700
        sl[50] = 43, 750
        sl[51] = 43, 850
        sl[52] = 43, 900
        sl[53] = 43, 800
        assert len(sl) == self.nchans
        self.SiteLoc = sl


class uMap54_1c(Probe):
    """uMap54_1c, 75 um spacing, 3 column, hexagonal"""
    def __init__(self):
        self.layout = '1c'
        self.name = 'uMap54_1c'
        self.nchans = 54
        self.ncols = 3
        sl = {}
        sl[0] = -65, 1251
        sl[1] = -65, 1101
        sl[2] = -65, 951
        sl[3] = -65, 801
        sl[4] = -65, 651
        sl[5] = -65, 501
        sl[6] = -65, 351
        sl[7] = -65, 201
        sl[8] = -65, 51
        sl[9] = -65, 126
        sl[10] = -65, 276
        sl[11] = -65, 426
        sl[12] = -65, 576
        sl[13] = -65, 726
        sl[14] = -65, 876
        sl[15] = -65, 1026
        sl[16] = -65, 1176
        sl[17] = -65, 1326
        sl[18] = 0, 1364
        sl[19] = 0, 1214
        sl[20] = 0, 1064
        sl[21] = 0, 914
        sl[22] = 0, 764
        sl[23] = 0, 614
        sl[24] = 0, 314
        sl[25] = 0, 164
        sl[26] = 0, 464
        sl[27] = 0, 89
        sl[28] = 0, 239
        sl[29] = 0, 389
        sl[30] = 0, 539
        sl[31] = 0, 689
        sl[32] = 0, 839
        sl[33] = 0, 989
        sl[34] = 0, 1139
        sl[35] = 0, 1289
        sl[36] = 65, 1326
        sl[37] = 65, 1251
        sl[38] = 65, 1176
        sl[39] = 65, 1026
        sl[40] = 65, 876
        sl[41] = 65, 726
        sl[42] = 65, 576
        sl[43] = 65, 426
        sl[44] = 65, 276
        sl[45] = 65, 126
        sl[46] = 65, 51
        sl[47] = 65, 201
        sl[48] = 65, 351
        sl[49] = 65, 501
        sl[50] = 65, 651
        sl[51] = 65, 951
        sl[52] = 65, 1101
        sl[53] = 65, 801
        assert len(sl) == self.nchans
        self.SiteLoc = sl


class uMap54_2a(Probe):
    """uMap54_2a, 65 um spacing, 2 column, staggered"""
    def __init__(self):
        self.layout = '2a'
        self.name = 'uMap54_2a'
        self.nchans = 54
        self.ncols = 2
        sl = {}
        sl[0] = -28, 1235
        sl[1] = -28, 1170
        sl[2] = -28, 1105
        sl[3] = -28, 1040
        sl[4] = -28, 975
        sl[5] = -28, 910
        sl[6] = -28, 845
        sl[7] = -28, 780
        sl[8] = -28, 715
        sl[9] = -28, 650
        sl[10] = -28, 585
        sl[11] = -28, 520
        sl[12] = -28, 455
        sl[13] = -28, 390
        sl[14] = -28, 325
        sl[15] = -28, 260
        sl[16] = -28, 195
        sl[17] = -28, 130
        sl[18] = -28, 65
        sl[19] = -28, 1300
        sl[20] = -28, 1365
        sl[21] = -28, 1430
        sl[22] = -28, 1495
        sl[23] = -28, 1560
        sl[24] = -28, 1690
        sl[25] = -28, 1755
        sl[26] = -28, 1625
        sl[27] = 28, 1722
        sl[28] = 28, 1657
        sl[29] = 28, 1592
        sl[30] = 28, 1527
        sl[31] = 28, 1462
        sl[32] = 28, 1397
        sl[33] = 28, 1332
        sl[34] = 28, 32
        sl[35] = 28, 97
        sl[36] = 28, 162
        sl[37] = 28, 227
        sl[38] = 28, 292
        sl[39] = 28, 357
        sl[40] = 28, 422
        sl[41] = 28, 487
        sl[42] = 28, 552
        sl[43] = 28, 617
        sl[44] = 28, 682
        sl[45] = 28, 747
        sl[46] = 28, 812
        sl[47] = 28, 877
        sl[48] = 28, 942
        sl[49] = 28, 1007
        sl[50] = 28, 1072
        sl[51] = 28, 1202
        sl[52] = 28, 1267
        sl[53] = 28, 1137
        assert len(sl) == self.nchans
        self.SiteLoc = sl


class uMap54_2b(Probe):
    """uMap54_2b, 50 um spacing, 2 column, staggered"""
    def __init__(self):
        self.layout = '2b'
        self.name = 'uMap54_2b'
        self.nchans = 54
        self.ncols = 2
        sl = {}
        sl[0] = -25, 1275
        sl[1] = -25, 1175
        sl[2] = -25, 1075
        sl[3] = -25, 975
        sl[4] = -25, 875
        sl[5] = -25, 775
        sl[6] = -25, 725
        sl[7] = -25, 675
        sl[8] = -25, 625
        sl[9] = -25, 575
        sl[10] = -25, 525
        sl[11] = -25, 475
        sl[12] = -25, 425
        sl[13] = -25, 375
        sl[14] = -25, 325
        sl[15] = -25, 275
        sl[16] = -25, 225
        sl[17] = -25, 175
        sl[18] = -25, 125
        sl[19] = -25, 75
        sl[20] = -25, 25
        sl[21] = -25, 825
        sl[22] = -25, 925
        sl[23] = -25, 1025
        sl[24] = -25, 1225
        sl[25] = -25, 1325
        sl[26] = -25, 1125
        sl[27] = 25, 1300
        sl[28] = 25, 1200
        sl[29] = 25, 1100
        sl[30] = 25, 1000
        sl[31] = 25, 900
        sl[32] = 25, 0
        sl[33] = 25, 50
        sl[34] = 25, 100
        sl[35] = 25, 150
        sl[36] = 25, 200
        sl[37] = 25, 250
        sl[38] = 25, 300
        sl[39] = 25, 350
        sl[40] = 25, 400
        sl[41] = 25, 450
        sl[42] = 25, 500
        sl[43] = 25, 550
        sl[44] = 25, 600
        sl[45] = 25, 650
        sl[46] = 25, 700
        sl[47] = 25, 750
        sl[48] = 25, 800
        sl[49] = 25, 850
        sl[50] = 25, 950
        sl[51] = 25, 1150
        sl[52] = 25, 1250
        sl[53] = 25, 1050
        assert len(sl) == self.nchans
        self.SiteLoc = sl


class pt16a_HS27(Probe):
    """pt16a in DIP-16 to HS-27 adapter"""
    def __init__(self):
        self.layout = 'pt16a_HS27'
        self.name = 'pt16a_HS27'
        self.nchans = 20
        self.ncols = 2
        sl = {}
        sl[0] = -27, 279
        sl[1] = -27, 217
        sl[2] = -27, 155
        sl[3] = -27, 93
        sl[4] = -27, 31
        sl[5] = -27, 341
        sl[6] = -27, 403
        sl[7] = -27, 465
        # Gap of 4 (grounded) chans in the adapter, give them sites below the probe:
        sl[8] = -27, 650
        sl[9] = -27, 700
        sl[10] = 27, 650
        sl[11] = 27, 700
        # Back to actual polytrode sites:
        sl[12] = 27, 434
        sl[13] = 27, 372
        sl[14] = 27, 310
        sl[15] = 27, 0
        sl[16] = 27, 62
        sl[17] = 27, 124
        sl[18] = 27, 186
        sl[19] = 27, 248
        assert len(sl) == self.nchans
        self.SiteLoc = sl


class pt16b_HS27(Probe):
    """pt16b in DIP-16 to HS-27 adapter"""
    def __init__(self):
        self.layout = 'pt16b_HS27'
        self.name = 'pt16b_HS27'
        self.nchans = 20
        self.ncols = 2
        sl = {}
        sl[0] = -27, 155
        sl[1] = -27, 93
        sl[2] = -27, 217
        sl[3] = -27, 341
        sl[4] = -27, 31
        sl[5] = -27, 279
        sl[6] = -27, 403
        sl[7] = -27, 465
        # Gap of 4 (grounded) chans in the adapter, give them sites below the probe:
        sl[8] = -27, 650
        sl[9] = -27, 700
        sl[10] = 27, 650
        sl[11] = 27, 700
        # Back to actual polytrode sites:
        sl[12] = 27, 434
        sl[13] = 27, 372
        sl[14] = 27, 248
        sl[15] = 27, 0
        sl[16] = 27, 310
        sl[17] = 27, 186
        sl[18] = 27, 62
        sl[19] = 27, 124
        assert len(sl) == self.nchans
        self.SiteLoc = sl


class single(Probe):
    """Single channel"""
    def __init__(self):
        self.layout = 'single'
        self.name = 'single'
        self.nchans = 1
        self.ncols = 1
        sl = {}
        sl[0] = 0, 0
        self.SiteLoc = sl


class IMEC30(Probe):
    """30 chan IMEC probe snippet, 2 column, 22 um rectangular spacing"""
    def __init__(self):
        self.layout = 'IMEC30'
        self.name = 'IMEC30'
        self.nchans = 30
        self.ncols = 2
        sl = {}
        sl[0] = 0, 1050
        sl[1] = 22, 1050
        sl[2] =  0, 1072
        sl[3] = 22, 1072
        sl[4] =  0, 1094
        sl[5] = 22, 1094
        sl[6] =  0, 1116
        sl[7] = 22, 1116
        sl[8] =  0, 1138
        sl[9] = 22, 1138
        sl[10] =  0, 1160
        sl[11] = 22, 1160
        sl[12] =  0, 1182
        sl[13] = 22, 1182
        sl[14] =  0, 1204
        sl[15] = 22, 1204
        sl[16] =  0, 1226
        sl[17] = 22, 1226
        sl[18] =  0, 1248
        sl[19] = 22, 1248
        sl[20] =  0, 1270
        sl[21] = 22, 1270
        sl[22] =  0, 1292
        sl[23] = 22, 1292
        sl[24] =  0, 1314
        sl[25] = 22, 1314
        sl[26] =  0, 1336
        sl[27] = 22, 1336
        sl[28] =  0, 1358
        sl[29] = 22, 1358
        self.SiteLoc = sl


TYPES = [uMap54_1a, uMap54_1b, uMap54_1c, uMap54_2a, uMap54_2b,
         pt16a_HS27, pt16b_HS27, single, IMEC30]
