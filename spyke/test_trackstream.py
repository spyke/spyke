from surf import File
from core import TrackStream
import numpy as np

fnames = ['/data/ptc15/87 - track 7c spontaneous craziness.srf',
          '/data/ptc15/92 - track 7c mseq32 0.4deg.srf']

srffs = []
for fname in fnames:
    srff = File(fname)
    srff.parse()
    srffs.append(srff)

ts = TrackStream(srffs=srffs)
ts.chans = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53])
wave = ts[0:1000000]
