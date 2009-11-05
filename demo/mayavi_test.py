"""Example of how to generate 3D ellipsoids in mayavi.

If you're using the mlab API:

from enthought.mayavi import mlab
f = mlab.figure() # returns the current scene
engine = mlab.get_engine() # returns the running mayavi engine

"""

from enthought.mayavi.api import Engine
from enthought.mayavi.sources.api import ParametricSurface
from enthought.mayavi.modules.api import Surface
from enthought.mayavi import mlab

import numpy as np

engine = Engine()
engine.start()
scene = engine.new_scene()
scene.scene.disable_render = True # for speed

surfaces = []
for i in range(10):

    source = ParametricSurface()
    source.function = 'ellipsoid'
    engine.add_source(source)

    surface = Surface()
    source.add_module(surface)

    actor = surface.actor # mayavi actor, actor.actor is tvtk actor
    #actor.property.ambient = 1 # defaults to 0 for some reason, ah don't need it, turn off scalar visibility instead
    actor.property.opacity = 0.5
    actor.property.color = tuple(np.random.rand(3))
    actor.mapper.scalar_visibility = False # don't colour ellipses by their scalar indices into colour map
    actor.property.backface_culling = True # gets rid of weird rendering artifact when opacity is < 1
    #actor.property.frontface_culling = True
    actor.actor.orientation = np.random.rand(3) * 360 # in degrees
    actor.actor.origin = np.random.rand(3)
    actor.actor.position = np.random.rand(3)
    actor.actor.scale = np.random.rand(3)

    surfaces.append(surface)

scene.scene.disable_render = False # now turn it on

# set the scalars, this has to be done some indeterminate amount of time
# after each surface is created, otherwise the scalars get overwritten
# later by their default of 1.0
for i, surface in enumerate(surfaces):
    vtk_srcs = mlab.pipeline.get_vtk_src(surface)
    print('len(vtk_srcs) = %d' % len(vtk_srcs))
    vtk_src = vtk_srcs[0]
    try: npoints = len(vtk_src.point_data.scalars)
    except TypeError:
        print('hit the TypeError on surface i=%d' % i)
        npoints = 2500
    vtk_src.point_data.scalars = np.tile(i, npoints)

# on pick, find the ellipsoid with origin closest to the picked coord,
# then check if that coord falls within that nearest ellipsoid, and if
# so, print out the ellispoid id, or pop it up in a tooltip

mlab.show()

# use mlab.outline() to draw an outline around an object? hopefully you can get it to
# outline an ellipsoid on mouse hover, the way the orientation axes do


