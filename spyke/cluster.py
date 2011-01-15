"""Define the cluster frame, with methods for dealing with ellipsoids"""

from __future__ import division

__authors__ = ['Martin Spacek']

#import os
#os.environ['ETS_TOOLKIT'] = 'qt4'
import sys
import time
import numpy as np

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt

from enthought.traits.api import HasTraits, Instance
from enthought.traits.ui.api import View, Item
from enthought.tvtk.pyface.scene_editor import SceneEditor # this takes forever
from enthought.mayavi.tools.mlab_scene_model import MlabSceneModel # so does this
from enthought.mayavi.core.ui.mayavi_scene import MayaviScene
from enthought.mayavi import mlab
from enthought.mayavi.tools.engine_manager import get_engine

from core import SpykeToolWindow
from plot import CMAP, CMAPPLUSTRANSWHITE, TRANSWHITEI


class Cluster(object):
    """Just a simple container for multidim ellipsoid parameters. A
    Cluster will always correspond to a Neuron"""
    def __init__(self, neuron, ellipsoid=None):
        self.neuron = neuron
        self.ellipsoid = ellipsoid
        # cluster attribs store true values of each dim
        self.pos = {'x0':0, 'y0':0, 'sx':0, 'sy':0, 'Vpp':0, 'V0':0, 'V1':0, 'w0':0, 'w1':0, 'w2':0, 'w3':0, 'w4':0, 'dphase':0, 't':0, 's0':0, 's1':0, 'mVpp':0, 'mV0':0, 'mV1':0, 'mdphase':0}
        # for ori, each dict entry for each dim is (otherdim1, otherdim2): ori_value
        # reversing the dims in the key requires negating the ori_value
        self.ori = {'x0':{}, 'y0':{}, 'sx':{}, 'sy':{}, 'Vpp':{}, 'V0':{}, 'V1':{}, 'w0':{}, 'w1':{}, 'w2':{}, 'w3':{}, 'w4':{}, 'dphase':{}, 't':{}, 's0':{}, 's1':{}, 'mVpp':{}, 'mV0':{}, 'mV1':{}, 'mdphase':{}}
        # set scale to 0 to exclude a param from consideration as a
        # dim when checking which points fall within which ellipsoid
        self.scale = {'x0':0.25, 'y0':0.25, 'sx':0, 'sy':0, 'Vpp':0.25, 'V0':0, 'V1':0, 'w0':0, 'w1':0, 'w2':0, 'w3':0, 'w4':0, 'dphase':0, 't':0, 's0':0, 's1':0, 'mVpp':0, 'mV0':0, 'mV1':0, 'mdphase':0}

    def get_id(self):
        return self.neuron.id

    def set_id(self, id):
        self.neuron.id = id

    id = property(get_id, set_id)

    def update_ellipsoid(self, params=None, dims=None):
        ellipsoid = self.ellipsoid
        if ellipsoid == None:
            return
        if params == None:
            params = ['pos', 'ori', 'scale']
        if 'pos' in params:
            ellipsoid.actor.actor.position = [ self.pos[dim] for dim in dims ]
        if 'ori' in params:
            oris = []
            for dimi, dim in enumerate(dims):
                # pick the two next highest dims, in that order,
                # wrapping back to lowest one when out of bounds
                # len(dims) is always 3, so mod by 3 for the wrap around
                otherdims = dims[(dimi+1)%3], dims[(dimi+2)%3]
                rotherdims = otherdims[1], otherdims[0] # reversed
                if otherdims in self.ori[dim]:
                    oris.append(self.ori[dim][otherdims])
                elif rotherdims in self.ori[dim]:
                    oris.append(-self.ori[dim][rotherdims])
                else: # otherdims/rotherdims is not a key, ori for this combination hasn't been set
                    oris.append(0)
            ellipsoid.actor.actor.orientation = oris
        if 'scale' in params:
            ellipsoid.actor.actor.scale = [ self.scale[dim] for dim in dims ]

    def __getstate__(self):
        d = self.__dict__.copy() # copy it cuz we'll be making changes
        del d['ellipsoid'] # remove Mayavi ellipsoid surface, recreate on unpickle
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.ellipsoid = None # restore attrib, proper value is restored later


class SpykeMayaviScene(MayaviScene):
    def __init__(self, *args, **kwargs):
        MayaviScene.__init__(self, *args, **kwargs)
        qw = self._vtk_control # QWidget
        #qw.setMouseTracking(True) # unnecessary
        # probably not the best way to do this, but works:
        qw.mouseMoveEvent = self.mouseMoveEvent
        qw.keyPressEvent = self.keyPressEvent

    def mouseMoveEvent(self, event):
        """Pop up a nid tooltip on mouse movement"""
        #QtGui.QToolTip.hideText() # hide first if you want tooltip to move even when text is unchanged
        qw = self._vtk_control
        if event.buttons() != Qt.NoButton: # don't show tooltip if mouse buttons are pressed
            QtGui.QToolTip.hideText()
            qw.__class__.mouseMoveEvent(qw, event) # pass the event on
            return
        pos = event.pos()
        x = pos.x()
        y = qw.size().height() - pos.y()
        data = self.picker.pick_point(x, y)
        if data.data != None:
            scalar = data.data.scalars[0] # just grab the first value
            if scalar < 0: # -ve vals are clusters, +ve vals are plotted points
                nid = int(-(scalar + 1))
                tip = 'nid: %d' % nid
                QtGui.QToolTip.showText(event.globalPos(), tip)
            else:
                QtGui.QToolTip.hideText()
        else:
            QtGui.QToolTip.hideText()
        qw.__class__.mouseMoveEvent(qw, event) # pass the event on

    def keyPressEvent(self, event):
        qw = self._vtk_control # QWidget
        spw = qw.topLevelWidget().spykewindow # can't do this in __init__ due to mayavi weirdness
        sw = spw.windows['Sort']
        cw = spw.windows['Cluster']
        key = event.key()
        if key in [Qt.Key_S, Qt.Key_Space]:
            # toggle selection of cluster under the cursor
            globalPos = QtGui.QCursor.pos()
            pos = qw.mapFromGlobal(globalPos)
            x = pos.x()
            y = qw.size().height() - pos.y()
            data = self.picker.pick_point(x, y)
            if data.data != None:
                scalar = data.data.scalars[0] # just grab the first value
                if scalar < 0: # -ve vals are clusters, +ve vals are plotted points
                    nid = int(-(scalar + 1))
                    spw.ToggleCluster(nid)
        elif key in [Qt.Key_Escape, Qt.Key_Delete, Qt.Key_M, Qt.Key_NumberSign,
                     Qt.Key_O, Qt.Key_Period, Qt.Key_R]:
            sw.keyPressEvent(event) # pass it on to Sort window
        elif key == Qt.Key_F11:
            cw.keyPressEvent(event) # pass it on
        else:
            qw.__class__.keyPressEvent(qw, event) # pass it on


class Visualization(HasTraits):
    """Don't really understand this.
    Copied from http://code.enthought.com/projects/mayavi/docs/development/
    html/mayavi/_downloads/qt_embedding.py"""
    scene = Instance(MlabSceneModel, ())
    editor = SceneEditor(scene_class=SpykeMayaviScene)
    item = Item('scene', editor=editor, show_label=False)
    view = View(item, resizable=True) # resize with the parent widget
    '''
    @on_trait_change('scene.activated')
    def update_plot(self):
        """Called when the view is opened. We don't populate the scene when the view
        is not yet open, as some VTK features require a GLContext."""
        # We can do normal mlab calls on the embedded scene.
        #self.scene.mlab.test_points3d()
    '''

class ClusterWindow(SpykeToolWindow):
    def __init__(self, parent, pos=None, size=None):
        SpykeToolWindow.__init__(self, parent, flags=QtCore.Qt.Tool)
        self.spykewindow = parent
        self.setWindowTitle("Cluster Window")
        self.move(*pos)
        self.resize(*size)

        # also copied from qt_embedding.py:
        self.vis = Visualization()
        self.ui = self.vis.edit_traits(parent=self, kind='subpanel').control # generates widget to embed
        self.setCentralWidget(self.ui)

        # this is a hack to remove the vtkObserver that catches 'a' and 'c' VTK CharEvents
        # to see all registered observers, print the interactor
        #self.vis.scene.interactor.remove_observer(1)
        # here's how to add your own observer to catch vtk keypress events
        #self.vis.scene.interactor.add_observer('KeyPressEvent', self.on_vtkkeypress)

        self.f = get_engine().current_scene
        self.f.scene.background = 0, 0, 0 # set it to black
        self.f.scene.picker.tolerance = 0.0025

    def closeEvent(self, event):
        self.spykewindow.HideWindow('Cluster')
    '''
    def keyPressEvent(self, event):
        print('keyPressEvent in cluster window')
        SpykeToolWindow.keyPressEvent(self, event) # pass it on
    '''
    '''
    def on_vtkkeypress(self, obj, evt):
        """Custom VTK key press event. Here just for instructive purposes.
        See http://article.gmane.org/gmane.comp.python.enthought.devel/10491"""
        key = obj.GetKeyCode()
        print('key == %s' % key)
        spykeframe = self.Parent
    '''
    def plot(self, X, scale=None,
             mode='point', scale_factor=0.5, alpha=None,
             mask_points=None, resolution=8, line_width=2.0, envisage=False):
        """Plot 3D projection of (possibly clustered) spike params in X. scale
        each dimension in X by scale. Mode can be '2darrow', '2dcircle', '2dcross',
        '2ddash', '2ddiamond', '2dhooked_arrow', '2dsquare', '2dthick_arrow',
        '2dthick_cross', '2dtriangle', '2dvertex', 'arrow', 'cone', 'cube',
        'cylinder', 'point', 'sphere'. 3D glyphs like 'sphere' come out
        looking almost black if OpenGL isn't working right, and are slower -
        use 'point' instead. if mask_points is not None, plots only 1 out
        of every mask_points points, to reduce number of plotted points for
        big data sets. envisage=True gives mayavi's full envisage GUI

        NOTE: use glyph.mlab_source.x, .y, .z, and .scalars traits to modify
        data in-place. If you're not replacing the whole trait, say just
        a slice of it, you need to call glyph.mlab_source.update() afterwards.
        Actually, .update() only seems to be effective for scalar updates,
        doesn't seem to work for x, y and z.
        You can also use the .set() method to update multiple traits at once
        """
        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]
        cmap = CMAPPLUSTRANSWHITE
        s = np.tile(TRANSWHITEI, len(X))

        if envisage == True:
            mlab.options.backend = 'envisage' # full GUI instead of just simple window
        # plot it
        t0 = time.time()
        f = self.f
        f.scene.disable_render = True # for speed
        # clear just the plotted glyph representing the points, not the whole scene including the ellipsoids
        try:
            f.scene.remove_actor(self.glyph.actor.actor)
            view, roll = self.view, self.roll
        except AttributeError: pass # no self.glyph exists yet
        #mlab.clf(f) # clear the whole scene
        #f.scene.camera.view_transform_matrix.scale(3, 1, 1) # this doesn't seem to work
        kwargs = {'figure': f, 'mode': mode,
                  #'opacity': alpha,
                  'transparent': True, # make the alpha of each point depend on the alpha of each scalar?
                  'mask_points': mask_points,
                  'resolution': resolution,
                  'line_width': line_width,
                  'scale_mode': 'none', # keep all points the same size
                  'scale_factor': scale_factor,
                  'vmin': 0, # make sure mayavi respects full range of cmap indices
                  'vmax': len(cmap)-1}
        glyph = mlab.points3d(x, y, z, s, **kwargs)
        try:
            self.view, self.roll = view, roll
        except NameError: # view and roll weren't set above cuz no self.glyph existed yet
            pass
        glyph.module_manager.scalar_lut_manager.load_lut_from_list(cmap) # assign colourmap
        glyph.module_manager.scalar_lut_manager.data_range = np.array([0, len(cmap)-1]) # need to force it again for some reason
        if scale: glyph.actor.actor.scale = scale
        f.scene.disable_render = False
        print("Plotting took %.3f sec" % (time.time()-t0))
        return glyph

    def add_ellipsoid(self, cluster, dims, alpha=0.5, update=True):
        """Add an ellipsoid to figure self.f, given its corresponding cluster
        TODO: turn on 4th light source - looks great!
        """
        #from enthought.mayavi.sources.api import ParametricSurface
        #from enthought.mayavi.modules.api import Surface
        from enthought.tvtk.api import tvtk

        f = self.f # the current scene
        #x, y, z = dims # dimension names
        #engine = f.parent
        f.scene.disable_render = True # for speed

        #source = ParametricSurface()
        #source.function = 'ellipsoid'
        #engine.add_source(source)
        #ellipsoid = Surface() # the surface is the ellipsoid
        #source.add_module(ellipsoid)
        point = np.array([0, 0, 0])
        # tensor seems to require 20 along the diagonal for the glyph to be the expected size
        tensor = np.array([20, 0, 0,
                           0, 20, 0,
                           0, 0, 20])
        data = tvtk.PolyData(points=[point])
        data.point_data.tensors = [tensor]
        data.point_data.tensors.name = 'some_name'
        data.point_data.scalars = [-cluster.id-1] # make them all -ve to distinguish them from plotted points
        # save camera view and roll
        view, roll = self.view, self.roll
        glyph = mlab.pipeline.tensor_glyph(data) # resets camera view and roll unfortunately
        # restore camera view and roll
        self.view, self.roll = view, roll
        glyph.glyph.glyph_source.glyph_source.theta_resolution = 50
        glyph.glyph.glyph_source.glyph_source.phi_resolution = 50

        #actor = ellipsoid.actor # mayavi actor, actor.actor is tvtk actor
        actor = glyph.actor # mayavi actor, actor.actor is tvtk actor
        actor.property.opacity = alpha
        # use cluster id (from associated neuron) as index into CMAP to colour the ellipse
        actor.property.color = tuple(CMAP[cluster.id % len(CMAP)][0:3]) # leave out alpha
        # don't colour ellipsoids by their scalar indices into builtin colour map
        actor.mapper.scalar_visibility = False
        # get rid of weird rendering artifact when opacity is < 1:
        actor.property.backface_culling = True
        #actor.property.frontface_culling = True
        #actor.actor.origin = 0, 0, 0
        cluster.ellipsoid = glyph
        cluster.update_ellipsoid(dims=dims) # update all params
        if update:
            f.scene.disable_render = False

    def get_view(self):
        return mlab.view()

    def set_view(self, view):
        mlab.view(*view)

    view = property(get_view, set_view)

    def get_roll(self):
        return mlab.roll()

    def set_roll(self, roll):
        mlab.roll(roll)

    roll = property(get_roll, set_roll)
