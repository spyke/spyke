"""Example of how to generate 3D ellipsoids in mayavi.

If you're using the mlab API:

from enthought.mayavi import mlab
f = mlab.figure() # Returns the current scene.
engine = mlab.get_engine() # Returns the running mayavi engine.

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


surfaces, actors = [], []
for i in range(10):
    surface = Surface()
    source.add_module(surface)
    surfaces.append(surface)
    actor = surface.actor
    actors.append(actor)

    actor.property.ambient = 1 # defaults to 0 for some reason
    actor.property.opacity = 0.5
    actor.property.color = tuple(np.random.rand(3))
    actor.actor.orientation = np.random.rand(3)
    actor.actor.origin = np.random.rand(3)
    actor.actor.position = np.random.rand(3)
    actor.actor.scale = np.random.rand(3)

scene.scene.disable_render = False # now turn it on
