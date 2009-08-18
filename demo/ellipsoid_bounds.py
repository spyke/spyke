"""Demos rotational transformation of coordinates to match that of
an ellipsoid"""

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

A = 20
B = 20
C = 50

# does mayavi rotates axes in order from biggest to smallest angles?
txdeg = 15
tydeg = 20
tzdeg = 25

tx = txdeg * np.pi / 180
ty = tydeg * np.pi / 180
tz = tzdeg * np.pi / 180

#RxRy = np.dot(Rx(tx), Ry(ty))
#RxRyRz = np.dot(RxRy, Rz(tz))

#RzRx = np.dot(Rz(tz), Rx(tx))
#RzRxRz = np.dot(RzRx, Rz(tz))

RzRy = np.dot(Rz(tz), Ry(ty))
RzRyRx = np.dot(RzRy, Rx(tx))

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
actor.actor.position = 0, 0, 0
actor.actor.scale = A, B, C


p = np.random.random((100000, 3)) # random set of points from 0 to 1
p -= 0.5 # centered
p *= 100 # scaled

#p2 = np.empty(p.shape, dtype=p.dtype)
#Rxt = Rx(tx)
#for pointi, point in enumerate(p):
#    p2[pointi] = np.dot(Rxt.T, point) # do the coordinate transform, should call this prime

#x = p[:, 0]
#y = p[:, 1]
#z = p[:, 2]

#i = []
#for pointi, point in enumerate(p):
#    if point[0]**2/A**2 + point[1]**2/B**2 + point[2]**2/C**2 <= 1:
#        i.append(pointi)
#i, = np.where(x**2/A**2 + y**2/B**2 + z**2/C**2 <= 1) # which points are in the ellipse?
# first need to pick out the points that fall within the unrotated elipsoid, then rotate the points
i, = np.where(p[:, 0]**2/A**2 + p[:, 1]**2/B**2 + p[:, 2]**2/C**2 <= 1) # which points are in the (unrotated) ellipse?
p = p[i] # pick out those points
#p = np.dot(RzRyRx, p.T).T # now rotate those points to match the rotated ellipse
p = np.dot(RzRyRx, p.T).T # now rotate those points to match the rotated ellipse

# plot the points
mlab.points3d(p[:, 0], p[:, 1], p[:, 2], mode='point') # you should see that all points lie within the ellipse
