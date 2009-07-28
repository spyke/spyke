"""Example of how to generate 3D ellipsoids in mayavi.

If you're using the mlab API:

from enthought.mayavi import mlab
f = mlab.figure() # returns the current scene
engine = mlab.get_engine() # returns the running mayavi engine

"""

from enthought.mayavi.api import Engine
from enthought.mayavi.sources.api import ParametricSurface
from enthought.mayavi.modules.api import Surface

import numpy as np

engine = Engine()
engine.start()
scene = engine.new_scene()
scene.scene.disable_render = True # for speed

source = ParametricSurface()
source.function = 'ellipsoid'
engine.add_source(source)


#surfaces, actors = [], []
for i in range(10):
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

    #surfaces.append(surface)
    #actors.append(actor)

scene.scene.disable_render = False # now turn it on

# use mlab.outline() to draw an outline around an object? hopefully you can get it to
# outline an ellipsoid on mouse hover, the way the orientation axes do
