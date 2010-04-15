"""Define the cluster frame, with methods for dealing with ellipsoids"""

from __future__ import division

__authors__ = ['Martin Spacek']

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

from spyke.plot import CMAP, CMAPPLUSJUNK, CMAPPLUSTRANSWHITE, TRANSWHITEI


class Cluster(object):
    """Just a simple container for multidim ellipsoid parameters. A
    Cluster will always correspond to a Neuron. But, if manual sorting is enabled
    (which it isn't), not all Neurons will necessarily have a cluster."""
    def __init__(self, neuron, ellipsoid=None):
        self.neuron = neuron
        self.ellipsoid = ellipsoid
        # cluster attribs store true values of each dim, unscaled by the
        # visualization sort.SCALE factor
        self.pos = {'x0':0, 'y0':0, 'sx':0, 'sy':0, 'Vpp':100, 'V0':50, 'V1':50, 'w0':0, 'w1':0, 'w2':0, 'w3':0, 'w4':0, 'dphase':200, 't':0, 's0':0, 's1':0, 'mVpp':0, 'mV0':0, 'mV1':0, 'mdphase':0}
        # for ori, each dict entry for each dim is (otherdim1, otherdim2): ori_value
        # reversing the dims in the key requires negating the ori_value
        self.ori = {'x0':{}, 'y0':{}, 'sx':{}, 'sy':{}, 'Vpp':{}, 'V0':{}, 'V1':{}, 'w0':{}, 'w1':{}, 'w2':{}, 'w3':{}, 'w4':{}, 'dphase':{}, 't':{}, 's0':{}, 's1':{}, 'mVpp':{}, 'mV0':{}, 'mV1':{}, 'mdphase':{}}
        # set scale to 0 to exclude a param from consideration as a
        # dim when checking which points fall within which ellipsoid
        self.scale = {'x0':10, 'y0':20, 'sx':0, 'sy':0, 'Vpp':0, 'V0':0, 'V1':0, 'w0':0, 'w1':0, 'w2':0, 'w3':0, 'w4':0, 'dphase':0, 't':0, 's0':0, 's1':0, 'mVpp':0, 'mV0':0, 'mV1':0, 'mdphase':0}

    def get_id(self):
        return self.neuron.id

    def set_id(self, id):
        self.neuron.id = id

    id = property(get_id, set_id)

    def update_ellipsoid(self, params=None, dims=None):
        ellipsoid = self.ellipsoid
        SCALE = self.neuron.sort.SCALE
        if ellipsoid == None:
            return
        if params == None:
            params = ['pos', 'ori', 'scale']
        if 'pos' in params:
            ellipsoid.actor.actor.position = [ self.pos[dim]*SCALE[dim] for dim in dims ]
        if 'ori' in params:
            # TODO: need to take SCALE into account here somehow
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
            ellipsoid.actor.actor.scale = [ self.scale[dim]*SCALE[dim] for dim in dims ]

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
        tooltip = wx.ToolTip('\n') # create a tooltip, stick a newline in there so subsequent ones are recognized
        tooltip.Enable(False) # leave disabled for now
        tooltip.SetDelay(0) # set popup delay in ms
        self._vtk_control.SetToolTip(tooltip) # connect it to self

        self._vtk_control.Bind(wx.EVT_MOTION, self.OnMotion)
        self._spykeframe = self._vtk_control.TopLevelParent.Parent # need _ to bypass traits check

    def OnMotion(self, event):
        """Pop up a nid tooltip on mouse movement"""
        if event.LeftIsDown() or event.MiddleIsDown() or event.RightIsDown():
            event.Skip() # leave button down cases for navigation

        pos = event.GetPosition()
        x = pos.x
        y = self._vtk_control.GetSize()[1] - pos.y
        #x = event.GetX()
        #y = self._vtk_control.GetSize()[1] - event.GetY()
        data = self.picker.pick_point(x, y)
        tooltip = self._vtk_control.GetToolTip()
        if data.data != None:
            scalar = data.data.scalars[0] # just grab the first value
            if scalar < 0:
                nid = -(scalar + 1)
                tip = 'nid: %d' % nid
                tooltip.SetTip(tip)
                tooltip.Enable(True)
            return
        tooltip.Enable(False)

    def OnKeyDown(self, event):
        key = event.GetKeyCode()
        #modifiers = event.HasModifiers()
        if key in [ord('a'), ord('d'), ord('x'), ord('c')]:
            # can't call spykeframe from here, do it in on_vtkkeypress
            if key == ord('x'):
                # copied over from tvtk.pyface.ui.wx.scene.OnKeyUp
                x = event.GetX()
                y = self._vtk_control.GetSize()[1] - event.GetY()
                # set camera focal point and move current cluster to that point
                data = self.picker.pick_world(x, y)
                coord = data.coordinate
                if coord is not None:
                    self.camera.focal_point = coord
                    self.render()
                    self._record_methods('camera.focal_point = %r\n'\
                                         'render()'%list(coord))
            self._vtk_control.OnKeyDown(event)
            return
        spykeframe = self._spykeframe
        try: cluster = spykeframe.GetCluster() # TODO: adjust multiple clusters simult?
        except RuntimeError:
            # pass event to parent class
            MayaviScene.OnKeyDown(self, event)
        x, y, z = spykeframe.GetDimNames()
        dim, sign = None, None
        SCALE = spykeframe.sort.SCALE
        modifiers = event.GetModifiers()
        if key == wx.WXK_PAGEDOWN: # inc xdim
            dim, sign = x, 1
        elif key == wx.WXK_DELETE: # dec xdim
            dim, sign = x, -1
        elif key == wx.WXK_HOME: # inc ydim
            dim, sign = y, 1
        elif key == wx.WXK_END: # dec ydim
            dim, sign = y, -1
        elif key == wx.WXK_PAGEUP: # inc zdim
            dim, sign = z, 1
        elif key == wx.WXK_INSERT: # dec zdim
            dim, sign = z, -1

        if dim != None and sign != None:
            if modifiers == wx.MOD_NONE: # adjust pos slowly
                cluster.pos[dim] += sign * 1 / SCALE[dim]
                cluster.update_ellipsoid('pos', dims=(x, y, z))
            elif modifiers == wx.MOD_NONE|wx.MOD_ALT: # adjust pos quickly
                cluster.pos[dim] += sign * 5 / SCALE[dim]
                cluster.update_ellipsoid('pos', dims=(x, y, z))
            elif modifiers == wx.MOD_SHIFT: # adjust scale slowly
                cluster.scale[dim] += sign * 1 / SCALE[dim]
                cluster.update_ellipsoid('scale', dims=(x, y, z))
            elif modifiers == wx.MOD_SHIFT|wx.MOD_ALT: # adjust scale quickly
                cluster.scale[dim] += sign * 5 / SCALE[dim]
                cluster.update_ellipsoid('scale', dims=(x, y, z))
            elif event.ControlDown(): # adjust ori
                # TODO: ori adjustment doesn't take SCALE into account!
                axes    = {x:(y, z), y:(z, x), z:(x, y)}
                revaxes = {x:(z, y), y:(x, z), z:(y, x)}
                if modifiers == wx.MOD_CONTROL: # adjust ori slowly
                    step = 1
                elif modifiers == wx.MOD_CONTROL|wx.MOD_ALT: # adjust ori quickly
                    step = 5
                if revaxes[dim] in cluster.ori[dim]: # reversed axes already used as a key
                    cluster.ori[dim][revaxes[dim]] -= sign * step # reverse the ori
                else:
                    try:
                        cluster.ori[dim][axes[dim]] += sign * step # update non-reversed axes entry
                    except KeyError:
                        cluster.ori[dim][axes[dim]] = sign * step # add non-reversed axes entry
                cluster.update_ellipsoid('ori', dims=(x, y, z))
            # NOTE: wx.MOD_CONTROL in combination with a numeric key press
            # doesn't seem to trigger a KeyDown event - maybe it triggers
            # some other kind of event?
        else:
            # pass event to parent class
            MayaviScene.OnKeyDown(self, event)

    def OnKeyUp(self, event):
        """Update param widgets when param modifying key is released,
        don't bother updating them when a param modification
        key is being held down"""
        key = event.GetKeyCode()
        if key in [wx.WXK_PAGEDOWN, wx.WXK_DELETE,
                   wx.WXK_HOME, wx.WXK_END,
                   wx.WXK_PAGEUP, wx.WXK_INSERT]:
            spykeframe = self._spykeframe
            cluster = spykeframe.GetCluster()
            spykeframe.UpdateParamWidgets(cluster)
        # pass event to parent class
        MayaviScene.OnKeyUp(self, event)


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
        self.f.scene.picker.tolerance = 0.0025

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
            spykeframe.FocusCurrentCluster()
        elif key == 'c':
            spykeframe.OnApplyCluster()

    def plot(self, X, scale=None, nids=None, minspikes=1,
             mode='point', scale_factor=0.5, alpha=None,
             mask_points=None, resolution=8, line_width=2.0, envisage=False):
        """Plot 3D projection of (possibly clustered) spike params in X. scale
        each dimension in X by scale. nids is a sequence of neuron ids
        corresponding to a sorted sequence of spike ids. "Neurons" with <
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
        cmap = CMAPPLUSTRANSWHITE
        '''
        if nids: # figure out scalar value to assign to each spike to colour it correctly
            t0 = time.time()
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
            s = nids % len(cmap)
            if njunk > 0:
                # use CMAPPLUSJUNK with its extra junk colour only if it's needed,
                # otherwise mayavi rescales and throws out a middle colour
                # (like light blue), and you end up with dk grey points even
                # though you don't have any junk points
                cmap = CMAPPLUSJUNK # has extra dk grey colour at end for junk
                s[junknidi:] = len(cmap) - 1 # assign last colour (dk grey) to junk clusters
            # unsort, so mayavi pick indices match spike indices
            nids = nids[unsortednidis] # unsort nids back to its original spike id order
            s = s[unsortednidis] # do the same for the colourmap indices
            print("Figuring out colours took %.3f sec" % (time.time()-t0))
            # TODO: order colours consecutively according to cluster mean y location, to
            # make neighbouring clusters in X-Y space less likely to be assigned the same colour
        else:
        '''
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

    def add_ellipsoid(self, cluster, dims, alpha=0.5):
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
