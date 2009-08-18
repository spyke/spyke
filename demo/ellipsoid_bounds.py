"""Demos rotational transformation of coordinates to match that of
an ellipsoid"""

import numpy as np

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


from enthought.mayavi import mlab
from enthought.mayavi.sources.api import ParametricSurface
from enthought.mayavi.modules.api import Surface


A = 20
B = 20
C = 50

tx = 15
ty = 0
tz = 0

RxRy = np.dot(Rx(tx*np.pi/180), Ry(ty*np.pi/180))
R = np.dot(RxRy, Rz(tz*np.pi/180))

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
actor.actor.orientation = tx, ty, tz #np.random.rand(3) * 360 # in degrees
#actor.actor.origin = np.random.rand(3)
actor.actor.position = 0, 0, 0
actor.actor.scale = A, B, C


p = np.random.random((10000, 3)) # random set of points from 0 to 1
p -= 0.5 # centered
p *= 100 # scaled

p = np.dot(Rx(tx*np.pi/180), p.T).T # do the coordinate transform, should call this prime

x = p[:, 0]
y = p[:, 1]
z = p[:, 2]

i, = np.where(x**2/A**2 + y**2/B**2 + z**2/C**2 <= 1) # which points are in the ellipse?

# plot the points
mlab.points3d(x[i], y[i], z[i], mode='point') # you should see that all points lie within the ellipse
