"""Define the cluster frame and methods for dealing with ellipsoids"""

import wx

from enthought.traits.api import HasTraits, Instance
from enthought.traits.ui.api import View, Item
from enthought.tvtk.pyface.scene_editor import SceneEditor
from enthought.mayavi.tools.mlab_scene_model import MlabSceneModel
from enthought.mayavi.core.ui.mayavi_scene import MayaviScene


class Viz(HasTraits):
    scene = Instance(MlabSceneModel, ())
    '''
    def __init__(self):
        HasTraits.__init__(self)
    '''
    # the layout of the dialog created
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=300, width=300, show_label=False))


class ClusterFrame(wx.MiniFrame):

    STYLE = wx.CAPTION|wx.CLOSE_BOX|wx.MAXIMIZE_BOX|wx.SYSTEM_MENU|wx.RESIZE_BORDER|wx.FRAME_TOOL_WINDOW

    def __init__(self, *args, **kwds):
        kwds["style"] = self.STYLE
        wx.MiniFrame.__init__(self, *args, **kwds)

        self.viz = Viz()
        self.control = self.viz.edit_traits(parent=self, kind='subpanel').control
        self.SetTitle("cluster window")
        #self.Show(True)

        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def OnClose(self, evt):
        frametype = type(self).__name__.lower().replace('frame', '') # remove 'Frame' from class name
        self.Parent.HideFrame(frametype)

    def plot(self, nids=None, dims=[0, 1, 2], weighting=None, scale=(3, 1, 1),
             minspikes=1, mode='point', alpha=0.5, scale_factor=0.5,
             mask_points=None, resolution=8, line_width=2.0, envisage=False):
        """Plot 3D projection of (possibly clustered) spike data. nids is
        a sequence of neuron ids corresponding to sorted sequence of spike
        ids. Make sure to pass the weighting that was used when clustering
        the data. "Clusters" with less than minspikes will all be coloured
        the same dark grey. Mode can be '2darrow', '2dcircle', '2dcross',
        '2ddash', '2ddiamond', '2dhooked_arrow', '2dsquare', '2dthick_arrow',
        '2dthick_cross', '2dtriangle', '2dvertex', 'arrow', 'cone', 'cube',
        'cylinder', 'point', 'sphere'. 3D glyphs like 'sphere' come out
        looking almost black if OpenGL isn't working right, and are slower -
        use 'point' instead. if mask_points is not None, plots only 1 out
        of every mask_points points, to reduce number of plotted points for
        big data sets. envisage=True gives mayavi's full envisage GUI

        NOTE: glyph.module_manager.source.data.get_point(point_id) gets you
        the coordinates of point point_id. Or, the plotted data is available
        at glyph.module_manager.source.data.points, so you can just index
        into that to get point coords. scalars are at
        glyph.module_manager.source.data.point_data.scalars

        """

        from enthought.mayavi import mlab # can't delay this any longer

        if envisage == True:
            mlab.options.backend = 'envisage' # full GUI instead of just simple window
        assert len(dims) == 3
        t0 = time.clock()
        if weighting:
            X = self.get_cluster_data(weighting=weighting) # in spike id order
        else:
            X = self.get_param_matrix() # in spike id order
        print("Getting param matrix took %.3f sec" % (time.clock()-t0))
        cmap = CMAP
        if nids:
            t0 = time.clock()
            nids = np.asarray(nids)
            sortednidis = nids.argsort() # indices to get nids in sorted order
            unsortednidis = sortednidis.argsort() # indices that unsort nids back to original order
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

        # plot it
        t0 = time.clock()
        f.scene.disable_render = True # for speed
        #f.scene.camera.view_transform_matrix.scale(3, 1, 1) # this doesn't seem to work
        x = X[:, dims[0]]
        y = X[:, dims[1]]
        z = X[:, dims[2]]
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

    def add_ellipsoid(self, id=0, f=None):
        """Add an ellipsoid to figure f. id is used to index into CMAP
        to colour the ellipsoid

        TODO: turn on 4th light source - looks great!
        """
        #from enthought.mayavi import mlab
        from enthought.mayavi.sources.api import ParametricSurface
        from enthought.mayavi.modules.api import Surface

        f = f or self.f # returns the current scene #mlab.figure()
        #engine = mlab.get_engine() # returns the running mayavi engine
        engine = f.parent
        f.scene.disable_render = True # for speed
        source = ParametricSurface()
        source.function = 'ellipsoid'
        engine.add_source(source)
        surface = Surface()
        source.add_module(surface)
        actor = surface.actor # mayavi actor, actor.actor is tvtk actor
        actor.property.opacity = 0.5
        actor.property.color = tuple(CMAP[id % len(CMAP)][0:3]) # leave out alpha
        # don't colour ellipsoids by their scalar indices into builtin colour map,
        # since I can't figure out how to set the scalar value of an ellipsoid anyway
        actor.mapper.scalar_visibility = False
        actor.property.backface_culling = True # gets rid of weird rendering artifact when opacity is < 1
        #actor.property.frontface_culling = True
        #actor.actor.orientation = 0, 0, 0
        #actor.actor.origin = 0, 0, 0
        actor.actor.position = 0, 0, 50
        actor.actor.scale = 20, 20, 50
        f.scene.disable_render = False
        return surface
    '''
    def apply_ellipsoids(self):
        for ellipsoid in self.ellipsoids.values():
    '''

