"""From Gael Varoquaux"""

from enthought.tvtk.api import tvtk
from enthought.mayavi import mlab

import numpy as np

x, y, z = np.random.random((3, 10))
points  = np.vstack((x, y, z)).T

u, v, w = np.random.random((3, 10))
o       = np.zeros_like(u)

tensors = np.vstack((u, o, o,
                     o, v, o,
                     o, o, w)).T

scalars = np.arange(10)

src = tvtk.PolyData(points=points)

src.point_data.tensors = tensors
src.point_data.scalars = scalars
# without the following 2 lines, I get this exception in a windows dialog, yet the visualization still comes up
'''
Exception
In C:\bib\Python26\lib\site-packages\mayavi-3.3.0-py2.6-win32.egg\enthought\tvtk\tvtk_base.py:523
RuntimeError: ERROR: In ..\..\..\archive\VTK\Graphics\vtkAssignAttribute.cxx, line 276
vtkAssignAttribute (0BBDF978): Data must be point or cell for vtkDataSet

(in _wrap_call)
'''
#src.point_data.tensors.name = 'tensor'
#srf.point_data.scalars.name = 'scalar'

mlab.clf()
mlab.pipeline.tensor_glyph(src)
