"""Spatial layouts of various electrophys (generally silicon polytrode) probe designs.
Spatial origin is at center top of each probe."""

# TODO: add pt16c, and pt16x layouts (with 16 to HS-27 adapter)

from __future__ import division
from __future__ import print_function

__authors__ = ['Martin Spacek']

import numpy as np

from .core import tolist

DEFNSXPROBETYPE = 'A1x32' # default .nsx probe type
# SPIKEDTYPE currently uses uint8 for chans, ensure during Probe instantiation that
# number of chans doesn't exceed this:
MAXNCHANS = 2**8


class Probe(object):
    """self.SiteLoc maps probe chan id to spatial position (x, y) in um.
    Note that y coordinates designate depth from the top site, i.e. increasing y coords
    correspond to positions further down the probe"""
    def __getitem__(self, chan):
        return self.SiteLoc[chan]

    def siteloc_arr(self):
        """Return site locations in an array, sorted by channel ID"""
        return np.asarray([ self[chan] for chan in self.chans ])

    def unique_coords(self, axis=1):
        """Return sorted unique coords along axis"""
        sa = self.siteloc_arr()
        coords = np.unique(sa[:, axis]) # comes out sorted
        return coords

    def separation(self, axis=1):
        """Return site separation along axis (0:x, 1:y, 2:z), assuming uniform spacing
        along axis"""
        coords = self.unique_coords(axis)
        sep = np.unique(np.diff(coords))
        if len(sep) == 0:
            return 0
        elif len(sep) > 1:
            raise ValueError("non-uniform spacing along axis %r" % axis)
            # take mean maybe?
        return sep[0] # pull scalar out of array

    def maxseparation(self, axis=1):
        """Return maximum site separation along axis"""
        coords = self.unique_coords(axis) # comes out sorted
        return coords[-1] - coords[0]

    def check(self):
        """Check probe attributes"""
        assert len(self.SiteLoc) == self.nchans <= MAXNCHANS

    @property
    def nrows(self):
        """ncols is set manually for each Probe class, but nrows is calculated automatically"""
        return len(self.unique_coords(axis=1))

    @property
    def chan0(self):
        """Are channel IDs 0-based or 1-based for this probe?"""
        chan0 = min(self.SiteLoc)
        assert chan0 in [0, 1]
        return chan0

    @property
    def chans(self):
        """Return all channel IDs, sorted"""
        return np.asarray(sorted(self.SiteLoc))

    def chansort(self, axis=['x', 'y'], reverse=[False, False]):
        """Return channels in one of many spatial arrangements. y origin is top of probe.

        Examples:

        axis='y'               : top to bottom
        axis='y', reverse=True : bottom to top
        axis='x'               : left to right
        axis='x', reverse=True : right to left

        axis=['x', 'y']                        : sort left to right, then top to bottom
        axis=['x', 'y'], reverse=[False, True] : sort left to right, then bottom to top
        axis=['x', 'y'], reverse=[True, False] : sort right to left, then top to bottom
        axis=['x', 'y'], reverse=[True, True]  : sort right to left, then bottom to top

        axis=['y', 'x']                        : sort top to bottom, then left to right
        axis=['y', 'x'], reverse=[False, True] : sort top to bottom, then right to left
        axis=['y', 'x'], reverse=[True, False] : sort bottom to top, then left to right
        axis=['y', 'x'], reverse=[True, True]  : sort bottom to top, then right to left
        """
        ax2col = {'x':0, 'y':1}
        axis = tolist(axis)
        reverse = tolist(reverse)
        assert len(axis) == len(reverse)
        for rev in reverse:
            assert rev in [True, False]

        # fancy logic here, this takes some careful thought to understand:
        if reverse == [False, False]:
            pass
        elif reverse == [False, True]:
            reverse = [True, True] # reverse 1st axis twice
        elif reverse == [True, False]:
            pass
        elif reverse == [True, True]:
            reverse = [False, True] # reverse 1st and 2nd axes simultaneously at the end

        chans = self.chans # sorted
        xy = self.siteloc_arr() # chan coords in chan order
        #print('initial:')
        #for (x, y), chan in zip(xy, chans):
        #    print('%3d, %3d, %3d' % (x, y, chan))
        for ax, rev in zip(axis, reverse):
            assert ax in ['x', 'y']
            col = ax2col[ax]
            # default kind quicksort isn't stable, but mergesort is:
            chanis = xy[:, col].argsort(kind='mergesort')
            if rev:
                chanis = chanis[::-1]
            chans = chans[chanis]
            xy = xy[chanis]
            #print('after axis', ax)
            #for (x, y), chan in zip(xy, chans):
            #    print('%3d, %3d, %3d' % (x, y, chan))
        return chans

    @property
    def chans_lrtb(self):
        return self.chansort(axis=['x', 'y'])

    @property
    def chans_lrbt(self):
        return self.chansort(axis=['x', 'y'], reverse=[False, True])

    @property
    def chans_tblr(self):
        return self.chansort(axis=['y', 'x'])

    @property
    def chans_btlr(self):
        return self.chansort(axis=['y', 'x'], reverse=[True, False])


class uMap54_1a(Probe):
    """uMap54_1a, 65 um spacing, 3 column hexagonal"""
    def __init__(self):
        self.layout = '1a'
        self.name = 'uMap54_1a'
        self.nchans = 54
        self.ncols = 3
        sl = {}
        sl[0] =  -56, 1170
        sl[1] =  -56, 1105
        sl[2] =  -56, 1040
        sl[3] =  -56, 975
        sl[4] =  -56, 910
        sl[5] =  -56, 845
        sl[6] =  -56, 585
        sl[7] =  -56, 455
        sl[8] =  -56, 325
        sl[9] =  -56, 195
        sl[10] = -56, 65
        sl[11] = -56, 130
        sl[12] = -56, 260
        sl[13] = -56, 390
        sl[14] = -56, 520
        sl[15] = -56, 650
        sl[16] = -56, 715
        sl[17] = -56, 780
        sl[18] =   0, 1072
        sl[19] =   0, 942
        sl[20] =   0, 812
        sl[21] =   0, 682
        sl[22] =   0, 552
        sl[23] =   0, 422
        sl[24] =   0, 162
        sl[25] =   0, 97
        sl[26] =   0, 292
        sl[27] =   0, 227
        sl[28] =   0, 357
        sl[29] =   0, 487
        sl[30] =   0, 617
        sl[31] =   0, 747
        sl[32] =   0, 877
        sl[33] =   0, 1007
        sl[34] =   0, 1137
        sl[35] =   0, 1202
        sl[36] =  56, 780
        sl[37] =  56, 650
        sl[38] =  56, 520
        sl[39] =  56, 390
        sl[40] =  56, 260
        sl[41] =  56, 130
        sl[42] =  56, 65
        sl[43] =  56, 195
        sl[44] =  56, 325
        sl[45] =  56, 455
        sl[46] =  56, 585
        sl[47] =  56, 715
        sl[48] =  56, 845
        sl[49] =  56, 910
        sl[50] =  56, 975
        sl[51] =  56, 1105
        sl[52] =  56, 1170
        sl[53] =  56, 1040
        self.SiteLoc = sl
        self.check()


class uMap54_1b(Probe):
    """uMap54_1b, 50 um horizontal/43 um vertical spacing, 3 column collinear"""
    def __init__(self):
        self.layout = '1b'
        self.name = 'uMap54_1b'
        self.nchans = 54
        self.ncols = 3
        sl = {}
        sl[0] =  -43, 900
        sl[1] =  -43, 850
        sl[2] =  -43, 800
        sl[3] =  -43, 750
        sl[4] =  -43, 700
        sl[5] =  -43, 650
        sl[6] =  -43, 600
        sl[7] =  -43, 550
        sl[8] =  -43, 500
        sl[9] =  -43, 450
        sl[10] = -43, 400
        sl[11] = -43, 350
        sl[12] = -43, 300
        sl[13] = -43, 250
        sl[14] = -43, 200
        sl[15] = -43, 150
        sl[16] = -43, 50
        sl[17] = -43, 100
        sl[18] =   0, 900
        sl[19] =   0, 800
        sl[20] =   0, 700
        sl[21] =   0, 600
        sl[22] =   0, 500
        sl[23] =   0, 400
        sl[24] =   0, 200
        sl[25] =   0, 100
        sl[26] =   0, 300
        sl[27] =   0, 50
        sl[28] =   0, 150
        sl[29] =   0, 250
        sl[30] =   0, 350
        sl[31] =   0, 450
        sl[32] =   0, 550
        sl[33] =   0, 650
        sl[34] =   0, 750
        sl[35] =   0, 850
        sl[36] =  43, 200
        sl[37] =  43, 100
        sl[38] =  43, 50
        sl[39] =  43, 150
        sl[40] =  43, 250
        sl[41] =  43, 300
        sl[42] =  43, 350
        sl[43] =  43, 400
        sl[44] =  43, 450
        sl[45] =  43, 500
        sl[46] =  43, 550
        sl[47] =  43, 600
        sl[48] =  43, 650
        sl[49] =  43, 700
        sl[50] =  43, 750
        sl[51] =  43, 850
        sl[52] =  43, 900
        sl[53] =  43, 800
        self.SiteLoc = sl
        self.check()


class uMap54_1c(Probe):
    """uMap54_1c, 75 um spacing, 3 column, hexagonal"""
    def __init__(self):
        self.layout = '1c'
        self.name = 'uMap54_1c'
        self.nchans = 54
        self.ncols = 3
        sl = {}
        sl[0] =  -65, 1251
        sl[1] =  -65, 1101
        sl[2] =  -65, 951
        sl[3] =  -65, 801
        sl[4] =  -65, 651
        sl[5] =  -65, 501
        sl[6] =  -65, 351
        sl[7] =  -65, 201
        sl[8] =  -65, 51
        sl[9] =  -65, 126
        sl[10] = -65, 276
        sl[11] = -65, 426
        sl[12] = -65, 576
        sl[13] = -65, 726
        sl[14] = -65, 876
        sl[15] = -65, 1026
        sl[16] = -65, 1176
        sl[17] = -65, 1326
        sl[18] =   0, 1364
        sl[19] =   0, 1214
        sl[20] =   0, 1064
        sl[21] =   0, 914
        sl[22] =   0, 764
        sl[23] =   0, 614
        sl[24] =   0, 314
        sl[25] =   0, 164
        sl[26] =   0, 464
        sl[27] =   0, 89
        sl[28] =   0, 239
        sl[29] =   0, 389
        sl[30] =   0, 539
        sl[31] =   0, 689
        sl[32] =   0, 839
        sl[33] =   0, 989
        sl[34] =   0, 1139
        sl[35] =   0, 1289
        sl[36] =  65, 1326
        sl[37] =  65, 1251
        sl[38] =  65, 1176
        sl[39] =  65, 1026
        sl[40] =  65, 876
        sl[41] =  65, 726
        sl[42] =  65, 576
        sl[43] =  65, 426
        sl[44] =  65, 276
        sl[45] =  65, 126
        sl[46] =  65, 51
        sl[47] =  65, 201
        sl[48] =  65, 351
        sl[49] =  65, 501
        sl[50] =  65, 651
        sl[51] =  65, 951
        sl[52] =  65, 1101
        sl[53] =  65, 801
        self.SiteLoc = sl
        self.check()


class uMap54_2a(Probe):
    """uMap54_2a, 65 um spacing, 2 column, staggered"""
    def __init__(self):
        self.layout = '2a'
        self.name = 'uMap54_2a'
        self.nchans = 54
        self.ncols = 2
        sl = {}
        sl[0] =  -28, 1235
        sl[1] =  -28, 1170
        sl[2] =  -28, 1105
        sl[3] =  -28, 1040
        sl[4] =  -28, 975
        sl[5] =  -28, 910
        sl[6] =  -28, 845
        sl[7] =  -28, 780
        sl[8] =  -28, 715
        sl[9] =  -28, 650
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
        sl[27] =  28, 1722
        sl[28] =  28, 1657
        sl[29] =  28, 1592
        sl[30] =  28, 1527
        sl[31] =  28, 1462
        sl[32] =  28, 1397
        sl[33] =  28, 1332
        sl[34] =  28, 32
        sl[35] =  28, 97
        sl[36] =  28, 162
        sl[37] =  28, 227
        sl[38] =  28, 292
        sl[39] =  28, 357
        sl[40] =  28, 422
        sl[41] =  28, 487
        sl[42] =  28, 552
        sl[43] =  28, 617
        sl[44] =  28, 682
        sl[45] =  28, 747
        sl[46] =  28, 812
        sl[47] =  28, 877
        sl[48] =  28, 942
        sl[49] =  28, 1007
        sl[50] =  28, 1072
        sl[51] =  28, 1202
        sl[52] =  28, 1267
        sl[53] =  28, 1137
        self.SiteLoc = sl
        self.check()


class uMap54_2b(Probe):
    """uMap54_2b, 50 um spacing, 2 column, staggered"""
    def __init__(self):
        self.layout = '2b'
        self.name = 'uMap54_2b'
        self.nchans = 54
        self.ncols = 2
        sl = {}
        sl[0] =  -25, 1275
        sl[1] =  -25, 1175
        sl[2] =  -25, 1075
        sl[3] =  -25, 975
        sl[4] =  -25, 875
        sl[5] =  -25, 775
        sl[6] =  -25, 725
        sl[7] =  -25, 675
        sl[8] =  -25, 625
        sl[9] =  -25, 575
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
        sl[27] =  25, 1300
        sl[28] =  25, 1200
        sl[29] =  25, 1100
        sl[30] =  25, 1000
        sl[31] =  25, 900
        sl[32] =  25, 0
        sl[33] =  25, 50
        sl[34] =  25, 100
        sl[35] =  25, 150
        sl[36] =  25, 200
        sl[37] =  25, 250
        sl[38] =  25, 300
        sl[39] =  25, 350
        sl[40] =  25, 400
        sl[41] =  25, 450
        sl[42] =  25, 500
        sl[43] =  25, 550
        sl[44] =  25, 600
        sl[45] =  25, 650
        sl[46] =  25, 700
        sl[47] =  25, 750
        sl[48] =  25, 800
        sl[49] =  25, 850
        sl[50] =  25, 950
        sl[51] =  25, 1150
        sl[52] =  25, 1250
        sl[53] =  25, 1050
        self.SiteLoc = sl
        self.check()


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
        self.SiteLoc = sl
        self.check()


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
        self.SiteLoc = sl
        self.check()


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
        self.check()


class IMEC30(Probe):
    """30 chan IMEC probe snippet, 2 column, 22 um rectangular spacing"""
    def __init__(self):
        self.layout = 'IMEC30'
        self.name = 'IMEC30'
        self.nchans = 30
        self.ncols = 2
        sl = {}
        sl[0] =   0, 1050
        sl[1] =  22, 1050
        sl[2] =   0, 1072
        sl[3] =  22, 1072
        sl[4] =   0, 1094
        sl[5] =  22, 1094
        sl[6] =   0, 1116
        sl[7] =  22, 1116
        sl[8] =   0, 1138
        sl[9] =  22, 1138
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
        self.check()


class A1x32(Probe):
    """A1x32, 25 um spacing, single column, 1-based channel IDs"""
    def __init__(self):
        self.layout = 'A1x32'
        self.name = 'A1x32'
        self.nchans = 32
        self.ncols = 1
        sl = {}
        sl[1] =  0, 775
        sl[2] =  0, 725
        sl[3] =  0, 675
        sl[4] =  0, 625
        sl[5] =  0, 575
        sl[6] =  0, 525
        sl[7] =  0, 475
        sl[8] =  0, 425
        sl[9] =  0, 375
        sl[10] = 0, 325
        sl[11] = 0, 275
        sl[12] = 0, 225
        sl[13] = 0, 175
        sl[14] = 0, 125
        sl[15] = 0, 75
        sl[16] = 0, 25
        sl[17] = 0, 0
        sl[18] = 0, 50
        sl[19] = 0, 100
        sl[20] = 0, 150
        sl[21] = 0, 200
        sl[22] = 0, 250
        sl[23] = 0, 300
        sl[24] = 0, 350
        sl[25] = 0, 400
        sl[26] = 0, 450
        sl[27] = 0, 500
        sl[28] = 0, 550
        sl[29] = 0, 600
        sl[30] = 0, 650
        sl[31] = 0, 700
        sl[32] = 0, 750
        self.SiteLoc = sl
        self.check()


class A1x32_edge(Probe):
    """A1x32 edge, 20 um spacing, single column, 1-based channel IDs"""
    def __init__(self):
        self.layout = 'A1x32_edge'
        self.name = 'A1x32_edge'
        self.nchans = 32
        self.ncols = 1
        sl = {}
        sl[1] =  0, 620
        sl[2] =  0, 600
        sl[3] =  0, 580
        sl[4] =  0, 560
        sl[5] =  0, 540
        sl[6] =  0, 520
        sl[7] =  0, 500
        sl[8] =  0, 480
        sl[9] =  0, 460
        sl[10] = 0, 440
        sl[11] = 0, 420
        sl[12] = 0, 400
        sl[13] = 0, 380
        sl[14] = 0, 360
        sl[15] = 0, 340
        sl[16] = 0, 320
        sl[17] = 0, 300
        sl[18] = 0, 280
        sl[19] = 0, 260
        sl[20] = 0, 240
        sl[21] = 0, 220
        sl[22] = 0, 200
        sl[23] = 0, 180
        sl[24] = 0, 160
        sl[25] = 0, 140
        sl[26] = 0, 120
        sl[27] = 0, 100
        sl[28] = 0, 80
        sl[29] = 0, 60
        sl[30] = 0, 40
        sl[31] = 0, 20
        sl[32] = 0, 0
        self.SiteLoc = sl
        self.check()


class A1x64(Probe):
    """A1x64, 30-38 um spacing (23 um vertical spacing), 2 column, 1-based channel IDs"""
    def __init__(self):
        self.layout = 'A1x64_Poly2_6mm_23s_160'
        self.name = 'A1x64'
        self.nchans = 64
        self.ncols = 2
        sl = {}
        sl[1] =  -15, 1196
        sl[2] =  -15, 1150
        sl[3] =  -15, 1104
        sl[4] =  -15, 1058
        sl[5] =  -15, 1012
        sl[6] =  -15, 966
        sl[7] =  -15, 920
        sl[8] =  -15, 874
        sl[9] =  -15, 828
        sl[10] = -15, 782
        sl[11] = -15, 736
        sl[12] = -15, 690
        sl[13] = -15, 644
        sl[14] = -15, 598
        sl[15] = -15, 552
        sl[16] = -15, 506
        sl[17] = -15, 460
        sl[18] = -15, 414
        sl[19] = -15, 368
        sl[20] = -15, 322
        sl[21] = -15, 276
        sl[22] = -15, 230
        sl[23] = -15, 184
        sl[24] = -15, 138
        sl[25] = -15, 92
        sl[26] = -15, 46
        sl[27] = -15, 0
        sl[28] = -15, 1242
        sl[29] = -15, 1288
        sl[30] = -15, 1334
        sl[31] = -15, 1380
        sl[32] = -10, 1426
        sl[33] =   9, 1449
        sl[34] =  15, 1403
        sl[35] =  15, 1357
        sl[36] =  15, 1311
        sl[37] =  15, 23
        sl[38] =  15, 69
        sl[39] =  15, 115
        sl[40] =  15, 161
        sl[41] =  15, 207
        sl[42] =  15, 253
        sl[43] =  15, 299
        sl[44] =  15, 345
        sl[45] =  15, 391
        sl[46] =  15, 437
        sl[47] =  15, 483
        sl[48] =  15, 529
        sl[49] =  15, 575
        sl[50] =  15, 621
        sl[51] =  15, 667
        sl[52] =  15, 713
        sl[53] =  15, 759
        sl[54] =  15, 805
        sl[55] =  15, 851
        sl[56] =  15, 897
        sl[57] =  15, 943
        sl[58] =  15, 989
        sl[59] =  15, 1035
        sl[60] =  15, 1081
        sl[61] =  15, 1127
        sl[62] =  15, 1173
        sl[63] =  15, 1219
        sl[64] =  15, 1265
        self.SiteLoc = sl
        self.check()


class H3(Probe):
    """Cambridge Neurotech H3 probe, 20 um spacing, single column single shaft,
    1-based channel IDs. Uses rev3 electrode interface board layout, which maps directly
    to Blackrock HSF_A64 analog headstage, and therefore doesn't require an adapter in spyke"""
    def __init__(self):
        self.layout = 'H3'
        self.name = 'H3'
        self.nchans = 64
        self.ncols = 1
        sl = {}
        sl[1] =  0, 320,
        sl[2] =  0, 620,
        sl[3] =  0, 340,
        sl[4] =  0, 600,
        sl[5] =  0, 360,
        sl[6] =  0, 580,
        sl[7] =  0, 380,
        sl[8] =  0, 560,
        sl[9] =  0, 400,
        sl[10] = 0, 540,
        sl[11] = 0, 420,
        sl[12] = 0, 520,
        sl[13] = 0, 440,
        sl[14] = 0, 500,
        sl[15] = 0, 460,
        sl[16] = 0, 100,
        sl[17] = 0, 200,
        sl[18] = 0, 120,
        sl[19] = 0, 180,
        sl[20] = 0, 140,
        sl[21] = 0, 0,
        sl[22] = 0, 300,
        sl[23] = 0, 20,
        sl[24] = 0, 40,
        sl[25] = 0, 220,
        sl[26] = 0, 240,
        sl[27] = 0, 160,
        sl[28] = 0, 280,
        sl[29] = 0, 80,
        sl[30] = 0, 60,
        sl[31] = 0, 480,
        sl[32] = 0, 260,
        sl[33] = 0, 1220,
        sl[34] = 0, 800,
        sl[35] = 0, 1020,
        sl[36] = 0, 1040,
        sl[37] = 0, 1240,
        sl[38] = 0, 1120,
        sl[39] = 0, 1200,
        sl[40] = 0, 1180,
        sl[41] = 0, 1000,
        sl[42] = 0, 980,
        sl[43] = 0, 1260,
        sl[44] = 0, 960,
        sl[45] = 0, 1100,
        sl[46] = 0, 1140,
        sl[47] = 0, 1080,
        sl[48] = 0, 1160,
        sl[49] = 0, 1060,
        sl[50] = 0, 780,
        sl[51] = 0, 820,
        sl[52] = 0, 760,
        sl[53] = 0, 840,
        sl[54] = 0, 740,
        sl[55] = 0, 860,
        sl[56] = 0, 720,
        sl[57] = 0, 880,
        sl[58] = 0, 700,
        sl[59] = 0, 900,
        sl[60] = 0, 680,
        sl[61] = 0, 920,
        sl[62] = 0, 660,
        sl[63] = 0, 940,
        sl[64] = 0, 640
        self.SiteLoc = sl
        self.check()



PROBETYPES = [uMap54_1a, uMap54_1b, uMap54_1c, uMap54_2a, uMap54_2b, pt16a_HS27, pt16b_HS27,
              single, IMEC30, A1x32, A1x32_edge, A1x64, H3]


def getprobe(name):
    """Get instantiated probe type by name or layout. Could potentially handle
    probe renaming here as well, but currently isn't necessary"""
    for probetype in PROBETYPES:
        probe = probetype()
        if probe.name == name or probe.layout == name:
            return probe
    raise ValueError("unknown probe name or layout %r" % name)

def findprobe(siteloc):
    """Return instantiation of first probe type whose layout matches siteloc"""
    for probetype in PROBETYPES:
        probe = probetype()
        if (probe.siteloc_arr().shape == siteloc.shape and
            (probe.siteloc_arr() == siteloc).all()):
            return probe
    raise ValueError("siteloc array:\n%r\ndoesn't match any known probe type" % siteloc)


class Adapter(object):
    """Anything that changes the channel mapping from one stage of recording to another,
    such as a plug adapter (say MOLEX to Omnetics)"""
    def __call__(self, probechan):
        ADchan = self.probe2AD[probechan]
        return ADchan

    def check(self):
        """Check adapter attributes"""
        assert len(self.probe2AD) == self.nchans <= MAXNCHANS

    @property
    def probechans(self):
        """Return all probe channel IDs, sorted"""
        return np.asarray(sorted(self.probe2AD))

    @property
    def ADchans(self):
        """Return all AD (amplifier) channels, sorted by probe channel ID"""
        return np.asarray([ self(probechan) for probechan in self.probechans ])

    @property
    def ADchansortis(self):
        """Return array that indexes into AD (amplifier) channels to return them sorted by
        probe channels"""
        return np.argsort(self.ADchans)


class HSF_A64(Adapter):
    """Blackrock HSF-A64 analog headstage to Blackrock NSP. This is required because the
    HSF-A64 is wired to match the older rev3 A64 probe board, whose probe chan to MOLEX pin
    mapping is different from rev4 and up. All new A64 probes from NeuroNexus use the newer
    rev4+ mapping. This remapping was done by inspecting the datasheet for the HSF-A64
    and finding the corresponding pin on the A64 probe"""
    def __init__(self):
        self.name = 'HSF_A64'
        self.nchans = 64
        p2a = {} # probe channel to AD (amplifier) channel mapping
        p2a[1] =  27
        p2a[2] =  25
        p2a[3] =  29
        p2a[4] =  23
        p2a[5] =  28
        p2a[6] =  21
        p2a[7] =  26
        p2a[8] =  19
        p2a[9] =  24
        p2a[10] = 17
        p2a[11] = 22
        p2a[12] = 15
        p2a[13] = 20
        p2a[14] = 13
        p2a[15] = 18
        p2a[16] = 11
        p2a[17] = 16
        p2a[18] = 9
        p2a[19] = 14
        p2a[20] = 7
        p2a[21] = 12
        p2a[22] = 3
        p2a[23] = 10
        p2a[24] = 1
        p2a[25] = 8
        p2a[26] = 6
        p2a[27] = 2
        p2a[28] = 4
        p2a[29] = 5
        p2a[30] = 30
        p2a[31] = 31
        p2a[32] = 32
        p2a[33] = 33
        p2a[34] = 34
        p2a[35] = 35
        p2a[36] = 60
        p2a[37] = 61
        p2a[38] = 63
        p2a[39] = 59
        p2a[40] = 57
        p2a[41] = 64
        p2a[42] = 55
        p2a[43] = 62
        p2a[44] = 53
        p2a[45] = 58
        p2a[46] = 51
        p2a[47] = 56
        p2a[48] = 49
        p2a[49] = 54
        p2a[50] = 47
        p2a[51] = 52
        p2a[52] = 45
        p2a[53] = 50
        p2a[54] = 43
        p2a[55] = 48
        p2a[56] = 41
        p2a[57] = 46
        p2a[58] = 39
        p2a[59] = 44
        p2a[60] = 37
        p2a[61] = 42
        p2a[62] = 36
        p2a[63] = 40
        p2a[64] = 38
        self.probe2AD = p2a
        self.check()


class Adpt_A64_OM32x2_sm_CerePlex_Mini(Adapter):
    """NeuroNexus Adpt-A64-OM32x2-sm (MOLEX to OM32x2-sm) adapter, to Blackrock Cereplex Mini
    64 channel (banks A and B) digital headstage, to Blackrock NSP. This was
    mapped by hand by Gregory Born by injecting signal into one channel on the MOLEX
    connectors at a time, and checking which channel showed signal on the Blackrock NSP.
    This was then confirmed by Martin Spacek by examining the pinout in the Cereplex Mini
    spec sheet."""
    def __init__(self):
        self.name = 'Adpt_A64_OM32x2_sm_CerePlex_Mini'
        self.nchans = 64
        p2a = {} # probe channel to AD (amplifier) channel mapping
        p2a[1] =  33
        p2a[2] =  37
        p2a[3] =  38
        p2a[4] =  41
        p2a[5] =  42
        p2a[6] =  43
        p2a[7] =  35
        p2a[8] =  46
        p2a[9] =  39
        p2a[10] = 48
        p2a[11] = 44
        p2a[12] = 2
        p2a[13] = 45
        p2a[14] = 4
        p2a[15] = 47
        p2a[16] = 6
        p2a[17] = 1
        p2a[18] = 8
        p2a[19] = 3
        p2a[20] = 10
        p2a[21] = 5
        p2a[22] = 13
        p2a[23] = 7
        p2a[24] = 15
        p2a[25] = 9
        p2a[26] = 11
        p2a[27] = 16
        p2a[28] = 14
        p2a[29] = 12
        p2a[30] = 36
        p2a[31] = 34
        p2a[32] = 40
        p2a[33] = 58
        p2a[34] = 64
        p2a[35] = 62
        p2a[36] = 22
        p2a[37] = 20
        p2a[38] = 18
        p2a[39] = 21
        p2a[40] = 23
        p2a[41] = 17
        p2a[42] = 25
        p2a[43] = 19
        p2a[44] = 27
        p2a[45] = 24
        p2a[46] = 29
        p2a[47] = 26
        p2a[48] = 31
        p2a[49] = 28
        p2a[50] = 49
        p2a[51] = 30
        p2a[52] = 51
        p2a[53] = 32
        p2a[54] = 54
        p2a[55] = 50
        p2a[56] = 57
        p2a[57] = 52
        p2a[58] = 61
        p2a[59] = 53
        p2a[60] = 56
        p2a[61] = 55
        p2a[62] = 60
        p2a[63] = 59
        p2a[64] = 63
        self.probe2AD = p2a
        self.check()


class CN_Adpt_A64_OM32x2_sm_CerePlex_Mini(Adapter):
    """Cambridge Neurotech dual MOLEX, to NeuroNexus Adpt-A64-OM32x2-sm (MOLEX to OM32x2-sm)
    adapter, to Blackrock Cereplex Mini 64 channel (banks A and B) digital headstage, to
    Blackrock NSP. This takes into account the fact that the dual MOLEX connectors on
    Cambridge Neurotech probes have a different channel mapping (A64 rev3) than those on
    recent NeuroNexus probes (A64 rev4 and up)."""
    def __init__(self):
        self.name = 'CN_Adpt_A64_OM32x2_sm_CerePlex_Mini'
        self.nchans = 64
        p2a = {} # probe channel to AD (amplifier) channel mapping
        p2a[1] =  15
        p2a[2] =  16
        p2a[3] =  13
        p2a[4] =  14
        p2a[5] =  12
        p2a[6] =  11
        p2a[7] =  10
        p2a[8] =  9
        p2a[9] =  8
        p2a[10] = 7
        p2a[11] = 6
        p2a[12] = 5
        p2a[13] = 4
        p2a[14] = 3
        p2a[15] = 2
        p2a[16] = 1
        p2a[17] = 48
        p2a[18] = 47
        p2a[19] = 46
        p2a[20] = 45
        p2a[21] = 43
        p2a[22] = 44
        p2a[23] = 41
        p2a[24] = 39
        p2a[25] = 37
        p2a[26] = 35
        p2a[27] = 33
        p2a[28] = 42
        p2a[29] = 38
        p2a[30] = 36
        p2a[31] = 34
        p2a[32] = 40
        p2a[33] = 58
        p2a[34] = 64
        p2a[35] = 62
        p2a[36] = 60
        p2a[37] = 56
        p2a[38] = 63
        p2a[39] = 61
        p2a[40] = 59
        p2a[41] = 57
        p2a[42] = 55
        p2a[43] = 54
        p2a[44] = 53
        p2a[45] = 51
        p2a[46] = 52
        p2a[47] = 49
        p2a[48] = 50
        p2a[49] = 31
        p2a[50] = 32
        p2a[51] = 29
        p2a[52] = 30
        p2a[53] = 27
        p2a[54] = 28
        p2a[55] = 25
        p2a[56] = 26
        p2a[57] = 23
        p2a[58] = 24
        p2a[59] = 21
        p2a[60] = 22
        p2a[61] = 20
        p2a[62] = 19
        p2a[63] = 18
        p2a[64] = 17
        self.probe2AD = p2a
        self.check()


class Adpt_A32_OM32_RHD2132(Adapter):
    """NeuroNexus A32 probe (single MOLEX) to Adpt-A32-OM32 (single MOLEX to OM32) adapter to
    Intan RHD2132 32 channel headstage (OM32 connector). For some reason, the pin numbers on
    the Adpt-A32-OM32 adapter map don't match the pin numbers on the usual NeuroNexus 32 chan
    probe A32 MOLEX block. Also, the omnetics pinouts of the adapter don't match the omnetics
    pin-ins of the RHD2132 headstage. This Adapter maps 1-based A32 probe channels all the way
    to 1-based RHD2132 headstage channels. This is for when both the adapter and the RHD2132
    headstage are connected facing the same way (both face up on one side, and both face down
    on the other)."""
    def __init__(self):
        self.name = 'Adpt_A32_OM32_RHD2132'
        self.nchans = 32
        p2a = {} # probe channel to AD (amplifier) channel mapping
        p2a[1] =  31
        p2a[2] =  27
        p2a[3] =  22
        p2a[4] =  18
        p2a[5] =  28
        p2a[6] =  23
        p2a[7] =  21
        p2a[8] =  26
        p2a[9] =  29
        p2a[10] = 24
        p2a[11] = 20
        p2a[12] = 25
        p2a[13] = 30
        p2a[14] = 19
        p2a[15] = 32
        p2a[16] = 17
        p2a[17] = 1
        p2a[18] = 16
        p2a[19] = 3
        p2a[20] = 14
        p2a[21] = 9
        p2a[22] = 10
        p2a[23] = 8
        p2a[24] = 2
        p2a[25] = 7
        p2a[26] = 15
        p2a[27] = 11
        p2a[28] = 12
        p2a[29] = 6
        p2a[30] = 13
        p2a[31] = 5
        p2a[32] = 4
        self.probe2AD = p2a
        self.check()


ADAPTERTYPES = [HSF_A64, Adpt_A64_OM32x2_sm_CerePlex_Mini, CN_Adpt_A64_OM32x2_sm_CerePlex_Mini,
                Adpt_A32_OM32_RHD2132]

def getadapter(name):
    """Get instantiated adapter type by name"""
    for adaptertype in ADAPTERTYPES:
        adapter = adaptertype()
        if adapter.name == name:
            return adapter
    raise ValueError("unknown adapter name %r" % name)
