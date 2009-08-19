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


x0 = 50
y0 = 1023
z0 = -23

A = 17.2
B = 99
C = 43.2

# mayavi (tvtk actually) rotates axes in Z, X, Y order, for some unknown reason
txdeg = 15.23
tydeg = 187.7
tzdeg = -420

tx = txdeg * np.pi / 180
ty = tydeg * np.pi / 180
tz = tzdeg * np.pi / 180

#RxRy = np.dot(Rx(tx), Ry(ty))
#RxRyRz = np.dot(RxRy, Rz(tz))

RzRx = np.dot(Rz(tz), Rx(tx))
RzRxRy = np.dot(RzRx, Ry(ty))

#RzRy = np.dot(Rz(tz), Ry(ty))
#RzRyRx = np.dot(RzRy, Rx(tx))

#RyRx = np.dot(Ry(ty), Rx(tx))
#RyRxRz = np.dot(RyRx, Rz(tz))

#R = Rz(tz)*Ry(ty)*Rx(tx) # they need to be matrices for this to work, yet plotted result is still wrong

# draw an ellipse
f = mlab.figure()
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
actor.actor.orientation = txdeg, tydeg, tzdeg #np.random.rand(3) * 360 # in degrees
#actor.actor.origin = np.random.rand(3)
actor.actor.position = x0, y0, z0
actor.actor.scale = A, B, C

# set up cube of points centerd on and engulfing the ellipsoid
p = 2 * np.random.random((100000, 3)) # random set of points from 0 to 2
p -= 1 # from -1 to 1
p *= 100 # scaled, from -100 to 100
# translated - centered on (x0, y0, z0)
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

# undo the rotation by taking product of inverse of rotation matrix (which == its transpose) and the untranlated points
p3 = np.dot(RzRxRy.T, p2.T).T
i, = np.where((p3[:, 0])**2/A**2 + (p3[:, 1])**2/B**2 + (p3[:, 2])**2/C**2 <= 1) # which points are inside the ellipsoid?
assert len(i) > 0, "no points fall within ellipsoid"
pinside = p[i] # pick out those points

# plot the points
mlab.points3d(pinside[:, 0], pinside[:, 1], pinside[:, 2], mode='point') # plotted points should lie within the ellipse
