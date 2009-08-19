"""Demonstrate translation and rotational transformation of point
coordinates to match that of an ellipsoid"""

import numpy as np
from enthought.mayavi import mlab
from enthought.mayavi.sources.api import ParametricSurface
from enthought.mayavi.modules.api import Surface


c = np.cos
s = np.sin

def Rx(t):
    return np.array([[1, 0,     0   ],
                     [0, c(t), -s(t)],
                     [0, s(t),  c(t)]])

def Ry(t):
    return np.array([[ c(t), 0, s(t)],
                     [ 0,    1, 0   ],
                     [-s(t), 0, c(t)]])

def Rz(t):
    return np.array([[c(t), -s(t), 0],
                     [s(t),  c(t), 0],
                     [0,     0,    1]])

#def R(tx, ty, tz):
#    return Rz(tz)*Ry(ty)*Rx(tx)

# translation params
x0 = 50
y0 = 1023
z0 = -23

# scaling params
A = 17.2
B = 99
C = 43.2

# orientation params
txdeg = 15.23
tydeg = 187.7
tzdeg = -420

tx = txdeg * np.pi / 180
ty = tydeg * np.pi / 180
tz = tzdeg * np.pi / 180

#RxRy = np.dot(Rx(tx), Ry(ty))
#RxRyRz = np.dot(RxRy, Rz(tz))

# mayavi (tvtk actually) rotates axes in Z, X, Y order, for some unknown reason
RzRx = np.dot(Rz(tz), Rx(tx))
RzRxRy = np.dot(RzRx, Ry(ty))

#RzRy = np.dot(Rz(tz), Ry(ty))
#RzRyRx = np.dot(RzRy, Rx(tx))

#RyRx = np.dot(Ry(ty), Rx(tx))
#RyRxRz = np.dot(RyRx, Rz(tz))

#R = Rz(tz)*Ry(ty)*Rx(tx) # they need to be matrices for this to work, yet plotted result is still wrong

# create a figure
f = mlab.figure(bgcolor=(0, 0, 0))

# draw an ellipsoid
engine = f.parent
f.scene.disable_render = True # for speed
source = ParametricSurface()
source.function = 'ellipsoid'
engine.add_source(source)
surface = Surface()
source.add_module(surface)
actor = surface.actor # mayavi actor, actor.actor is tvtk actor
actor.property.opacity = 0.5
actor.property.color = 1, 0, 0
# don't colour ellipses by their scalar indices into builtin colour map,
# since I can't figure out how to set the scalar value of an ellipsoid anyway
actor.mapper.scalar_visibility = False
actor.property.backface_culling = True # gets rid of weird rendering artifact when opacity is < 1
#actor.property.frontface_culling = True
actor.actor.orientation = txdeg, tydeg, tzdeg
#actor.actor.origin = 0, 0, 0
actor.actor.position = x0, y0, z0
actor.actor.scale = A, B, C

# set up cube of points centerd on and engulfing the ellipsoid
npoints = 100000
p = 2 * np.random.random((npoints, 3)) # random set of points from 0 to 2
p -= 1 # from -1 to 1
p *= max([A, B, C]) # scaled, from -max to +max
# translated, centered on (x0, y0, z0)
p[:, 0] += x0
p[:, 1] += y0
p[:, 2] += z0

# To find which points fall within the ellipse, need to do the inverse of all the operations that
# translate, and rotate the ellipse, in the correct order. Need to do those operations on the points,
# just to figure out which points to pick out, then pick them out of the original set of
# unmodified points

# undo the translation
p2 = p.copy()
p2[:, 0] -= x0
p2[:, 1] -= y0
p2[:, 2] -= z0

# undo the rotation by taking product of inverse of rotation matrix (which == its transpose) and the untranslated points
p3 = np.dot(RzRxRy.T, p2.T).T

# which points are inside the ellipsoid?
ini, = np.where((p3[:, 0])**2/A**2 + (p3[:, 1])**2/B**2 + (p3[:, 2])**2/C**2 <= 1)
assert len(ini) > 0, "no points fall within the ellipsoid"
pin = p[ini] # pick out those points

# which points are outside the ellipsoid?
alli = np.arange(npoints)
outi = list(set(alli).difference(ini))
pout = p[outi]

# plot the points inside
glyph_in = mlab.points3d(pin[:, 0], pin[:, 1], pin[:, 2], mode='point') # plotted points should lie within the ellipse
glyph_in.actor.property.color = 1, 1, 1 # inside is white

# plot the points outside
glyph_out = mlab.points3d(pout[:, 0], pout[:, 1], pout[:, 2], mode='point') # plotted points should lie within the ellipse
glyph_out.actor.property.color = 0.3, 0.3, 0.3 # outside is grey

# optionally turn off the points outside to better view the points inside
#glyph_out.visible = False
