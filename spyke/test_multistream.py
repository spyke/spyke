from surf import File
from stream import MultiStream
import numpy as np

path = '~/data/slab/ptc15/'

fnames = ['87 - track 7c spontaneous craziness.srf',
          '92 - track 7c mseq32 0.4deg.srf',
          '93 - track 7c final CSD.srf'
          ]


fs = []
for fname in fnames:
    f = File(fname, path)
    f.parse()
    fs.append(f)

ts = MultiStream(fs, 'test.track')
ts.chans = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
                      18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                      37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53])
t = ts.tranges

waves = []
waves += [ts[t[0,0]    :t[0,0]+100]]
waves += [ts[t[0,1]-100:t[0,1]+100]]

waves += [ts[t[0,1]    :t[0,1]+100]]

waves += [ts[t[1,0]-100:t[1,0]+100]]
waves += [ts[t[1,0]+100:t[1,0]+200]]
waves += [ts[t[1,1]-100:t[1,1]+100]]

# spans portions of 2 streams, with time gap in between
waves += [ts[t[1,1]-100:t[2,0]+100]]

waves += [ts[t[2,0]-100:t[2,0]+100]]
waves += [ts[t[2,0]+100:t[2,0]+200]]
waves += [ts[t[2,1]-100:t[2,1]    ]]

for wavei, wave in enumerate(waves):
    print('wave %d:\n%r' % (wavei, wave.data))
