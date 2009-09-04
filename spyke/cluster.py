"""Define the cluster frame, with methods for dealing with ellipsoids"""

import sys
import time
import numpy as np

import wx

from enthought.traits.api import HasTraits, Instance
from enthought.traits.ui.api import View, Item
from enthought.tvtk.pyface.scene_editor import SceneEditor
from enthought.mayavi.tools.mlab_scene_model import MlabSceneModel
from enthought.mayavi.core.ui.mayavi_scene import MayaviScene
from enthought.mayavi import mlab
from enthought.mayavi.tools.engine_manager import get_engine

from spyke.plot import CMAP, CMAPWITHJUNK


class Cluster(object):
    """Just a simple container for multidim ellipsoid parameters. A
    Cluster will always correspond to a Neuron, but not all Neurons
    will necessarily have a cluster. It's possible to create a Neuron
    purely by manually sorting individual spikes, without using a
    multidim ellipsoid at all"""
    def __init__(self, neuron, ellipsoid=None):
        self.neuron = neuron
        self.ellipsoid = ellipsoid
        self.pos =   {'x0':0,  'y0':0,  'Vpp':100, 'dphase':200}
        # for ori, each dict entry for each dim is (otherdim1, otherdim2): ori_value
        # reversing the dims in the key requires negating the ori_value
        self.ori =   {'x0':{}, 'y0':{}, 'Vpp':{},  'dphase':{}}
        # set scale to 0 to exclude dphase from consideration as a
        # dim when checking which points fall within which ellipsoid
        self.scale = {'x0':20, 'y0':20, 'Vpp':50,  'dphase':0}

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
        d.pop('spikeis', None) # exclude spikeis which hold indices of currently coloured points
        del d['ellipsoid'] # remove Mayavi ellipsoid surface, recreate on unpickle
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.ellipsoid = None # restore attrib, proper value is restored later


class SpykeMayaviScene(MayaviScene):
    def OnKeyDown(self, event):
        """Set camera focus point and move current cluster to
        that point in one step, with a single keypress"""
        key = event.GetKeyCode()
        modifiers = event.HasModifiers()
        if key in [ord('a'), ord('d'), ord('x'), ord('c')]:
            # can't call spykeframe from here, do it in on_vtkkeypress
            if key == ord('x'): # x marks the spot
                # set camera focal point. Copied over from tvtk.pyface.ui.wx.scene.OnKeyUp
                if not modifiers:
                    if sys.platform == 'darwin':
                        x, y = self._interactor.event_position
                    else:
                        x = event.GetX()
                        y = self._vtk_control.GetSize()[1] - event.GetY()
                    data = self.picker.pick_world(x, y)
                    coord = data.coordinate
                    if coord is not None:
                        self.camera.focal_point = coord
                        self.render()
                        self._record_methods('camera.focal_point = %r\n'\
                                             'render()'%list(coord))
                # then, set current cluster position to == camera focal point
                # can't call spykeframe from here, do it in on_vtkkeypress
            self._vtk_control.OnKeyDown(event)
        else: # propagate event to parent class
            MayaviScene.OnKeyDown(self, event)


class Visualization(HasTraits):
    """I don't understand this. See http://code.enthought.com/projects/mayavi/
    docs/development/htm/mayavi/building_applications.html"""
    scene = Instance(MlabSceneModel, ())
    editor = SceneEditor(scene_class=SpykeMayaviScene)
    item = Item('scene', editor=editor, show_label=False)
    view = View(item)


class ClusterFrame(wx.MiniFrame):

    STYLE = wx.CAPTION|wx.CLOSE_BOX|wx.MAXIMIZE_BOX|wx.SYSTEM_MENU|wx.RESIZE_BORDER|wx.FRAME_TOOL_WINDOW

    def __init__(self, *args, **kwds):
        kwds["style"] = self.STYLE
        wx.MiniFrame.__init__(self, *args, **kwds)

        self.vis = Visualization()
        self.control = self.vis.edit_traits(parent=self, kind='subpanel').control
        self.SetTitle("cluster window")
        #self.Show(True)
        # this is a hack to remove the vtkObserver that catches 'a' and 'c' VTK CharEvents
        # to see all registered observers, print the interactor
        self.vis.scene.interactor.remove_observer(1)
        # add my own observer to catch keypress events that need access to spykeframe
        self.vis.scene.interactor.add_observer('KeyPressEvent', self.on_vtkkeypress)

        self.Bind(wx.EVT_CLOSE, self.OnClose)

        self.f = get_engine().current_scene
        self.f.scene.background = 0, 0, 0 # set it to black

    def OnClose(self, evt):
        frametype = type(self).__name__.lower().replace('frame', '') # remove 'Frame' from class name
        self.Parent.HideFrame(frametype)

    def on_vtkkeypress(self, obj, evt):
        """Custom VTK key press event.
        See http://article.gmane.org/gmane.comp.python.enthought.devel/10491"""
        key = obj.GetKeyCode()
        spykeframe = self.Parent
        # finish handling the 'x' keypress. First part was handled by SpykeMayaviScene.OnKeyDown()
        if key == 'a':
            spykeframe.OnAddCluster()
        elif key == 'd':
            spykeframe.OnDelCluster()
        elif key == 'x':
            spykeframe.OnFocus()
        elif key == 'c':
            spykeframe.OnApplyCluster()

    def plot(self, X, scale=None, nids=None, minspikes=1,
             mode='point', scale_factor=0.5, alpha=0.5,
             mask_points=None, resolution=8, line_width=2.0, envisage=False):
        """Plot 3D projection of (possibly clustered) spike params in X. scale
        each dimension in X by scale. nids is a sequence of neuron ids
        corresponding to a sorted sequence of spike ids. "Neurons" with than
        minspikes will all be coloured the same dark grey.
        Mode can be '2darrow', '2dcircle', '2dcross',
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
        cmap = CMAP
        if nids: # figure out scalar value to assign to each spike to colour it correctly
            t0 = time.clock()
            nids = np.asarray(nids)
            sortednidis = nids.argsort() # indices to get nids in sorted order
            unsortednidis = sortednidis.argsort() # indices that unsort nids back to original spike id order
            nids = nids[sortednidis] # nids is now sorted
            maxnid = max(nids)
            consecutivenids = np.arange(maxnid+1)
            if set(nids) != set(consecutivenids):
                print("***WARNING: nids has gaps in it")
            # the extra +1 gives us the correct rightmost bin edge
            # for histogram's end inclusive semantics
            bins = np.arange(maxnid+1+1)
            hist, bins = np.histogram(nids, bins=bins)
            # assume lowest numbered nids are the most frequent ones
            # is hist in decreasing order, ie is difference between subsequent entries <= 0?
            try:
                assert (np.diff(hist) <= 0).all()
            except AssertionError:
                import pdb; pdb.set_trace()
            # find histi where hist drops to minspikes
            # searchsorted requires ascending order, not descending
            histifromend = hist[::-1].searchsorted(minspikes)
            histi = len(hist) - histifromend
            # take bins[histi] to find junknid, at which point all subsequently
            # numbered nids occur less than minspikes
            junknid = bins[histi]
            # should really get junknid == histi if everything's right
            try:
                assert junknid == histi
            except AssertionError:
                import pdb; pdb.set_trace()
            # junknidi is first index into sorted nids which occurs <= minspikes times,
            # and is considered junk, as are all subsequent ones
            junknidi = nids.searchsorted(junknid)
            # or maybe junknidi = sum(hist[:histi]) would work as well? faster?
            njunk = len(nids) - junknidi # number of junk points
            # s are indices into colourmap
            s = nids % len(CMAP)
            if njunk > 0:
                # use CMAPWITHJUNK with its extra junk colour only if it's needed,
                # otherwise mayavi rescales and throws out a middle colour
                # (like light blue), and you end up with dk grey points even
                # though you don't have any junk points
                cmap = CMAPWITHJUNK # has extra dk grey colour at end for junk
                s[junknidi:] = len(cmap) - 1 # assign last colour (dk grey) to junk clusters
            # unsort, so mayavi pick indices match spike indices
            nids = nids[unsortednidis] # unsort nids back to its original spike id order
            s = s[unsortednidis] # do the same for the colourmap indices
            print("Figuring out colours took %.3f sec" % (time.clock()-t0))
            # TODO: order colours consecutively according to cluster mean y location, to
            # make neighbouring clusters in X-Y space less likely to be assigned the same colour
        else:
            s = np.tile(9, len(X)) # CMAP[9] is WHITE

        if envisage == True:
            mlab.options.backend = 'envisage' # full GUI instead of just simple window
        # plot it
        t0 = time.clock()
        f = self.f
        f.scene.disable_render = True # for speed
        # clear just the plotted glyph representing the points, not the whole scene including the ellipsoids
        try: f.scene.remove_actor(self.glyph.actor.actor)
        except AttributeError: pass # no glyph exists yet
        #mlab.clf(f) # clear the whole scene
        #f.scene.camera.view_transform_matrix.scale(3, 1, 1) # this doesn't seem to work
        kwargs = {'figure': f, 'mode': mode,
                  'opacity': alpha,
                  #'transparent': True, # make the alpha of each point depend on the alpha of each scalar?
                  'mask_points': mask_points,
                  'resolution': resolution,
                  'line_width': line_width,
                  'scale_mode': 'none', # keep all points the same size
                  'scale_factor': scale_factor,
                  'vmin': 0, # make sure mayavi respects full range of cmap indices
                  'vmax': len(CMAP)-1}
        glyph = mlab.points3d(x, y, z, s, **kwargs)
        glyph.module_manager.scalar_lut_manager.load_lut_from_list(cmap) # assign colourmap
        if scale: glyph.actor.actor.scale = scale
        f.scene.disable_render = False
        print("Plotting took %.3f sec" % (time.clock()-t0))
        return glyph

    def add_ellipsoid(self, cluster, dims, alpha=0.5):
        """Add an ellipsoid to figure self.f, given its corresponding cluster
        TODO: turn on 4th light source - looks great!
        """
        from enthought.mayavi.sources.api import ParametricSurface
        from enthought.mayavi.modules.api import Surface

        f = self.f # the current scene
        x, y, z = dims # dimension names
        engine = f.parent
        f.scene.disable_render = True # for speed
        source = ParametricSurface()
        source.function = 'ellipsoid'
        engine.add_source(source)
        ellipsoid = Surface() # the surface is the ellipsoid
        source.add_module(ellipsoid)
        actor = ellipsoid.actor # mayavi actor, actor.actor is tvtk actor
        actor.property.opacity = alpha
        # use cluster id (from associated neuron) as index into CMAP to colour the ellipse
        actor.property.color = tuple(CMAP[cluster.id % len(CMAP)][0:3]) # leave out alpha
        # don't colour ellipsoids by their scalar indices into builtin colour map,
        # since I can't figure out how to set the scalar value of an ellipsoid anyway
        actor.mapper.scalar_visibility = False
        # get rid of weird rendering artifact when opacity is < 1:
        actor.property.backface_culling = True
        #actor.property.frontface_culling = True
        #actor.actor.origin = 0, 0, 0
        cluster.ellipsoid = ellipsoid
        cluster.update_ellipsoid(dims=dims) # update all params
        f.scene.disable_render = False
    '''
    # unnecessary, just use ellipsoid.remove()
    def remove_ellipsoid(self, cluster):
        """Remove ellipsoid from scene, given its corresponding cluster"""
        self.f.scene.remove_actor(cluster.ellipsoid.actor.actor)
    '''
