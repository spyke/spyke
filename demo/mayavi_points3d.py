"""Demo how to plot coloured points in 3D using mayavi.
This was for troubleshooting why assigning all points the same
scalar index into a colourmap didn't colour them as excepted.
This was due to mayavi automatically restricting the range
of the colourmap based on the range of the scalars passed to it.
This was fixed by passing the vmin and vmax args to points3d(),
which define the scalar range to use"""

import numpy as np
from spyke.gui.plot import CMAP
from enthought.mayavi import mlab

npoints = 10000

x = np.random.random(npoints)
y = np.random.random(npoints)
z = np.random.random(npoints)
s = np.tile(3, npoints) # CMAP[3] is green

f = mlab.figure(bgcolor=(0, 0, 0))
f.scene.disable_render = True # for speed
glyph = mlab.points3d(x, y, z, s, mode='point', vmin=0, vmax=len(CMAP)-1)
glyph.module_manager.scalar_lut_manager.load_lut_from_list(list(CMAP)) # assign colourmap
f.scene.disable_render = False
