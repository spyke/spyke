"""Demo how to plot coloured points in 3D using mayavi.
This was for troubleshooting why assigning all points the same
scalar index into a colourmap didn't colour them as excepted.
This was due to mayavi automatically restricting the range
of the colourmap based on the range of the scalars passed to it.
This was fixed by passing the vmin and vmax args to points3d(),
which define the scalar range to use"""
'''
import numpy as np
#from spyke.plot import CMAP
from enthought.mayavi.mlab import points3d, axes, show

npoints = 8000000

x = np.float32(np.random.random(npoints))
y = np.float32(np.random.random(npoints))
z = np.float32(np.random.random(npoints))
s = np.tile(3, npoints) # CMAP[3] is green

f = mlab.figure(bgcolor=(0, 0, 0))
f.scene.disable_render = True # for speed
glyph = mlab.points3d(x, y, z, s, mode='point', scale_factor=0.25)
#glyph = mlab.points3d(x, y, z, s, mode='point', vmin=0, vmax=len(CMAP)-1)
#glyph.module_manager.scalar_lut_manager.load_lut_from_list(list(CMAP)) # assign colourmap
f.scene.disable_render = False

# to change the scalar data after the glyph has been created:
#glyph.mlab_source.scalars = replacement_array
# OR:
#glyph.mlab_source.scalars[startindex:endindex] = value(s)
#glyph.mlab_source.update()
# although .update() often doesn't seem to work as expected for some reason

'''

import numpy as np
from enthought.mayavi import mlab
x = np.linspace(0, 1, 20)
y = np.linspace(0, 5, 20)
z = np.linspace(0, 1, 20)
mlab.points3d(x, y, z, z)


"""
Hi Christoph,

Sorry to bug you again. I was having some kind of minor problem with Mayavi
(can't even remember what it was anymore). I tried upgrading from your VTK 5.4.2
and ETS 3.4.0 amd64 Py2.6 binaries to your VTK 5.6.1 and ETS 3.5.0 amd64 Py2.6
binaries. Running the following in "ipython -pylab" seems to work:

from enthought.mayavi import mlab
mlab.test_points3d()

Here's the def for test_points3d in enthought.mayavi.tools.helper_functions:

def test_points3d():
      t = numpy.linspace(0, 4*numpy.pi, 20)
      cos = numpy.cos
      sin = numpy.sin
      x = sin(2*t)
      y = cos(t)
      z = cos(2*t)
      s = 2+sin(t)
      return points3d(x, y, z, s, colormap="copper", scale_factor=.25)

However, I've found that if I try calling mlab.points3d directly, without
specifying a scale_factor, Python crashes, and MSVC reports an unhandled
exception in the Python process, main thread, with vtkCommon.dll in the stack
frame. Here's the code I paste into ipython -pylab (using the "paste" command)
that causes the crash:

import numpy as np
from enthought.mayavi import mlab
x = np.arange(20)
mlab.points3d(x, x, x)

This happens from within both wx and qt4. I tried reverting back to your older
VTK 5.6.0 and ETS 3.4.2dev binaries, as well as to VTK 5.4.2 and ETS 3.4.0
(which came in zip files), to no avail. Note the above code works fine in linux.
Does that mean the culprit must be VTK and not mayavi?

I also get a different, more mysterious crash while importing
enthought.mayavi.mlab in my wx app in win64. Again, reverting VTK and ETS
doesn't help. I'm at a total loss on this one...

Any ideas?

Thanks for your time,

Martin
"""
