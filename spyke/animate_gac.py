"""Animate scout movement along first 2 dimensions during GAC. Modified from
http://matplotlib.org/examples/animation/simple_anim.html"""

import numpy as np
import pylab as pl
import matplotlib.animation as animation

# load scout position history saved by test_gac.py:
poshist1 = np.load('/home/mspacek/Desktop/poshist_manhattan_real_sigma_0.2.npy')
poshist2 = np.load('/home/mspacek/Desktop/poshist_euclid_real_sigma_0.2.npy')
#poshist1 = np.load('/home/mspacek/Desktop/poshist_manhattan_real_sigma_0.25.npy')
#poshist2 = np.load('/home/mspacek/Desktop/poshist_euclid_real_sigma_0.25.npy')
#poshist1 = np.load('/home/mspacek/Desktop/poshist_manhattan_real_sigma_0.3.npy')
#poshist2 = np.load('/home/mspacek/Desktop/poshist_euclid_real_sigma_0.3.npy')
#poshist1 = np.load('/home/mspacek/Desktop/poshist_manhattan_g.npy')
#poshist2 = np.load('/home/mspacek/Desktop/poshist_euclid_g.npy')
#poshist1 = np.load('/home/mspacek/Desktop/poshist_manhattan_g2.npy')
#poshist2 = np.load('/home/mspacek/Desktop/poshist_euclid_g2.npy')
#poshist1 = np.load('/home/mspacek/Desktop/poshist_manhattan_g3.npy')
#poshist2 = np.load('/home/mspacek/Desktop/poshist_euclid_g3.npy')


#nframes1 = len(poshist1)
#nframes2 = len(poshist2)
nframes = 200

f = pl.figure(figsize=(10, 10))
a1 = f.add_subplot(121)
a1.set_title('Manhattan')
a2 = f.add_subplot(122)
a2.set_title('Euclidean')

a1.set_aspect('equal')
a2.set_aspect('equal')

line1, = a1.plot(poshist1[0][:, 0], poshist1[0][:, 1], 'k.', ms=2) # plot x, y for first frame
line2, = a2.plot(poshist2[0][:, 0], poshist2[0][:, 1], 'k.', ms=2) # plot x, y for first frame

def animate1(i):
    line1.set_xdata(poshist1[i][:, 0]) # update x values
    line1.set_ydata(poshist1[i][:, 1]) # update y values
    return line1,

def animate2(i):
    line2.set_xdata(poshist2[i][:, 0]) # update x values
    line2.set_ydata(poshist2[i][:, 1]) # update y values
    return line2,
'''
def init():
    # only required for blitting to give a clean slate
    line.set_ydata(np.ma.array(line.get_xdata, mask=True))
    return line,
'''
ani2 = animation.FuncAnimation(f, animate2, np.arange(nframes),# init_func=init,
                               interval=500, blit=False) # interval is in ms
ani1 = animation.FuncAnimation(f, animate1, np.arange(nframes),# init_func=init,
                               interval=500, blit=False) # interval is in ms

pl.show()
