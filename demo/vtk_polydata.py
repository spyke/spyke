"""Example of how to generate 3D ellipsoids in mayavi, using glyphs.
Adapted from Gael Varoquaux"""

from enthought.tvtk.api import tvtk
from enthought.mayavi import mlab
#from enthought.mayavi.sources.vtk_data_source import VTKDataSource
import numpy as np

point = np.array([0, 0, 0])

tensor = np.array([20, 0, 0,
                   0, 20, 0,
                   0, 0, 20])

engine = mlab.get_engine()
engine.start()
scene = engine.new_scene()
scene.scene.disable_render = True # for speed

glyphs = []
for i in range(10):
    data = tvtk.PolyData(points=[point])
    data.point_data.tensors = [tensor]
    data.point_data.tensors.name = 'some_name'
    data.point_data.scalars = [i]
    #data.point_data.scalars.name = 'some_other_name'
    #mlab.clf()
    #src = VTKDataSource(data=data)

    #e = mlab.get_engine()
    #e.add_source(src)

    glyph = mlab.pipeline.tensor_glyph(data)
    glyph.glyph.glyph_source.glyph_source.theta_resolution = 50
    glyph.glyph.glyph_source.glyph_source.phi_resolution = 50

    actor = glyph.actor # mayavi actor, actor.actor is tvtk actor
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

    glyphs.append(glyph)

scene.scene.disable_render = False # now turn it on

mlab.show()
