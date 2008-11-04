"""wx.Panels with embedded mpl figures based on FigureCanvasWxAgg.
Everything is plotted in units of uV and us

TODO: perhaps refactor, keep info about each channel together,
make a Channel object with .id, .pos, .colour, .line, .enabled properties,
and set_enable() method, and then stick them in a dict of chans indexed by id
    - looks like I've more or less done this with the Plot object
"""

from __future__ import division

__authors__ = 'Martin Spacek, Reza Lotun'

import itertools
from copy import copy
import random
import numpy as np
import wx

from matplotlib import rcParams
rcParams['lines.linestyle'] = '-'
rcParams['lines.marker'] = ''

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from spyke.core import MU, intround

EVENTLINEWIDTH = 1 # in points
EVENTLINESTYLE = '-'
TEMPLATELINEWIDTH = 1.5
TEMPLATELINESTYLE = '-'
TREFLINEWIDTH = 0.5
VREFLINEWIDTH = 0.5
CHANVBORDER = 75 # uV, vertical border space between top and bottom chans and axes edge

DEFUVPERUM = 2
DEFUSPERUM = 17

DEFAULTCHANCOLOUR = '#00FF00' # garish green
TREFCOLOUR = '#303030' # dark grey
VREFCOLOUR = '#303030' # dark grey
CARETCOLOUR = '#202020' # light black
BACKGROUNDCOLOUR = 'black'
WXBACKGROUNDCOLOUR = wx.BLACK

RED = '#FF0000'
ORANGE = '#FF7F00'
YELLOW = '#FFFF00'
GREEN = '#00FF00'
CYAN = '#00FFFF'
LIGHTBLUE = '#007FFF'
BLUE = '#0000FF'
VIOLET = '#7F00FF'
MAGENTA = '#FF00FF'
GREY = '#888888'
WHITE = '#FFFFFF'
BROWN = '#AF5050'

COLOURS = [RED, ORANGE, YELLOW, GREEN, CYAN, LIGHTBLUE, VIOLET, MAGENTA, GREY, WHITE, BROWN]

NCLOSESTCHANSTOSEARCH = 10
PICKRADIUS = 15 # required for 'line.contains(event)' call
#PICKTHRESH = 2.0 # in pixels? has to be a float or it won't work?

SPIKESORTTW = 1000 # spike sort panel temporal window width (us)
CHARTSORTTW = 1000 # chart sort panel temporal window width (us)
EVENTTW = max(SPIKESORTTW, CHARTSORTTW) # default event time width, determines event.wave width
DEFNPLOTS = 10 # default number of plots to init in SortPanel

CARETZORDER = 0 # layering
REFLINEZORDER = 1
PLOTZORDER = 2


class ColourDict(dict):
    """Just an easy way to cycle through COLOURS given some index,
    like say a chan id or a template id. Better than using a generator,
    cuz you don't need to keep calling .next(). This is like a dict
    of inifite length"""
    def __getitem__(self, key):
        assert key.__class__ == int
        i = key % len(COLOURS)
        return COLOURS[i]

    def __setitem__(self, key, val):
        raise RuntimeError, 'ColourDict is unsettable'


COLOURDICT = ColourDict()


class Plot(object):
    """Plot slot, holds lines for all chans for plotting
    a single stretch of data, contiguous in time"""
    def __init__(self, chans, panel):
        self.lines = {} # chan to line mapping
        self.panel = panel # panel that self belongs to
        self.chans = chans # all channels available in this Plot, lines can be enabled/disabled, but .chans shouldn't change
        for chan in self.chans:
            line = Line2D([],
                          [], # TODO: will lack of data before first .draw() cause problems for blitting?
                          linewidth=EVENTLINEWIDTH,
                          linestyle=EVENTLINESTYLE,
                          color=self.panel.vcolours[chan],
                          zorder=PLOTZORDER,
                          antialiased=True,
                          animated=False, # True keeps this line from being copied to buffer on panel.copy_from_bbox() call,
                                          # but also unfortunately keeps it from being repainted upon occlusion
                          visible=False) # keep invisible until needed
            line.chan = chan
            line.set_pickradius(PICKRADIUS)
            #line.set_picker(PICKTHRESH)
            self.lines[chan] = line
            self.panel.ax.add_line(line) # add to panel's axes' pool of lines
        self.obj = None # Event or Template associated with this plot
        self.id = None # string id that indexes into SortPanel.used_plots, starts with 'e' or 't' for event or template

    def show(self, enable=True):
        """Show/hide all chans in self"""
        for line in self.lines.values():
            line.set_visible(enable)

    def hide(self):
        """Hide all chans in self"""
        self.show(False)

    def show_chans(self, chans, enable=True):
        """Show/hide specific chans in self"""
        for chan in chans:
            self.lines[chan].set_visible(enable)

    def hide_chans(self, chans):
        """Hide specific chans in self"""
        self.show_chans(chans, enable=False)

    def get_shown_chans(self):
        """Get list of currently shown chans"""
        chans = []
        for line in self.lines.values():
            if line.get_visible():
                chans.append(line.chan)
        return chans

    def update(self, wave, tref):
        """Update lines data
        TODO: most of the time, updating the xdata won't be necessary,
        but I think updating takes no time at all relative to drawing time"""
        self.tref = tref
        for chan, line in self.lines.items():
            xpos, ypos = self.panel.pos[chan]
            if wave.ts == None:
                xdata = []
                ydata = []
            else:
                xdata = wave.ts - tref + xpos
                # TODO: should be using wave.chans/wave.chan2i here?????????????????
                ydata = wave[chan]*self.panel.gain + ypos
            line.set_data(xdata, ydata) # update the line's x and y data

    def set_alpha(self, alpha):
        """Set alpha transparency for all lines in self"""
        for line in self.lines.values():
            line.set_alpha(alpha)

    def set_animated(self, enable=True):
        """Set animated flag for all lines in self"""
        for line in self.lines.values():
            line.set_animated(enable)

    def set_colours(self, colours):
        """Set colour(s) for all lines in self"""
        if len(colours) == 1:
            colours = colours * len(self.chans)
        if len(colours) != len(self.chans):
            raise ValueError, 'invalid colours length: %d' % len(colours)
        for chani, colour in enumerate(colours):
            self.lines[chani].set_color(colour)

    def set_stylewidth(self, style, width):
        """Set the line style and width for all lines in self"""
        for line in self.lines.values():
            line.set_linestyle(style)
            line.set_linewidth(width)

    def draw(self):
        """Draw all the lines to axes buffer (or whatever),
        avoiding unnecessary draws of all other artists in axes"""
        for line in self.lines.values():
            self.panel.ax.draw_artist(line)


class PlotPanel(FigureCanvasWxAgg):
    """A wx.Panel with an embedded mpl figure.
    Base class for specific types of plot panels"""

    # not necessarily constants
    uVperum = DEFUVPERUM
    usperum = DEFUSPERUM # decreasing this increases horizontal overlap between spike chans
                         # 17 gives roughly no horizontal overlap for self.tw == 1000 us
    def __init__(self, parent, id=-1, stream=None, tw=None, cw=None):
        FigureCanvasWxAgg.__init__(self, parent, id, Figure())
        self.spykeframe = self.GetTopLevelParent().Parent

        self.available_plots = [] # pool of available Plots
        self.used_plots = {} # Plots holding currently displayed event/template, indexed by eid/tid with e or t prepended
        self.quickRemovePlot = None # current quickly removable Plot with associated .background

        if stream != None:
            self.stream = stream # only bind for those frames that actually use a stream (ie DataFrames)
        self.tw = tw # temporal width of each channel, in plot units (us ostensibly)
        self.cw = cw # time width of caret, in plot units

        self.figure.set_facecolor(BACKGROUNDCOLOUR)
        self.figure.set_edgecolor(BACKGROUNDCOLOUR) # should really just turn off the edge line altogether, but how?
        #self.figure.set_frameon(False) # not too sure what this does, causes painting problems
        self.SetBackgroundColour(WXBACKGROUNDCOLOUR)

        tooltip = wx.ToolTip('\n') # create a tooltip, stick a newline in there so subsequent ones are recognized
        tooltip.Enable(False) # leave disabled for now
        tooltip.SetDelay(0) # set popup delay in ms
        self.SetToolTip(tooltip) # connect it to self

        self.mpl_connect('button_press_event', self.OnButtonPress) # bind mouse click within figure
        #self.mpl_connect('key_press_event', self.OnKeyPress)
        # TODO: mpl is doing something weird that prevents arrow key press events
        #self.mpl_connect('pick_event', self.OnPick) # happens when an artist with a .picker attrib has a mouse event happen within epsilon distance of it
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        #self.Bind(wx.EVT_NAVIGATION_KEY, self.OnNavigation)
        self.mpl_connect('motion_notify_event', self.OnMotion) # mouse motion within figure
        #self.mpl_connect('scroll_event', self.OnMouseWheel) # doesn't seem to be implemented yet in mpl's wx backend
        self.Bind(wx.EVT_MOUSEWHEEL, self.OnMouseWheel) # use wx event directly, although this requires window focus

    def callAfterFrameInit(self, probe=None):
        """Panel tasks that need to be done after parent frame has been created (and shown?)"""
        if probe == None:
            probe = self.stream.probe
        self.probe = probe
        self.SiteLoc = probe.SiteLoc # probe site locations with origin at center top
        self.chans = probe.SiteLoc.keys()
        self.chans.sort() # a sorted list of chans, keeps us from having to do this over and over
        self.nchans = probe.nchans
        # for plotting with mpl, convert probe SiteLoc to have center bottom origin instead of center top
        siteloc = copy(self.SiteLoc) # lowercase means bottom origin
        ys = [ y for x, y in siteloc.values() ]
        maxy = max(ys)
        for key, (x, y) in siteloc.items():
            y = maxy - y
            siteloc[key] = (x, y) # update
        self.siteloc = siteloc # center bottom origin

        self.init_axes()
        self.pos = {} # positions of line centers, in plot units (us, uV)
        self.vcolours = {} # colour mapping that cycles vertically in space
        self.do_layout() # defined by subclasses, sets self.pos
        self.xy_um = self.get_xy_um()
        x = self.xy_um[0]
        self.colxs = np.asarray(list(set(x))) # unique x values that demarcate columns
        self.colxs.sort() # guarantee they're in order from left to right
        self.ax.set_axis_off() # turn off the x and y axis
        self.ax.set_visible(True)
        self.ax.set_autoscale_on(False) # TODO: not sure if this is necessary
        self.draw()
        self.black_background = self.copy_from_bbox(self.ax.bbox) # init

        # add reference lines and caret in layered order
        self._show_caret(True) # call the _ methods directly, to prevent unnecessary draws
        self._show_tref(True)
        self._show_vref(True)
        for ref in ['caret', 'tref', 'vref']:
            self.spykeframe.menubar.Check(self.spykeframe.REFTYPE2ID[ref], True) # enforce menu item toggle state

        self.draw() # do a full draw of the ref lines
        self.reflines_background = self.copy_from_bbox(self.ax.bbox) # init
        self.init_plots()

    def init_axes(self):
        """Init the axes and ref lines"""
        self.ax = self.figure.add_axes([0, 0, 1, 1], # lbwh relative to figure?
                                       axisbg=BACKGROUNDCOLOUR,
                                       frameon=False,
                                       alpha=1.)

    def init_plots(self):
        """Create Plots for this panel"""
        self.quickRemovePlot = Plot(chans=self.chans, panel=self) # just one for this base class
        self.quickRemovePlot.show()
        self.used_plots[0] = self.quickRemovePlot

    def add_ref(self, ref):
        """Helper method for external use"""
        if ref == 'tref':
            self._add_tref()
        elif ref == 'vref':
            self._add_vref()
        elif ref == 'caret':
            self._add_caret()

    def show_ref(self, ref, enable=True):
        """Helper method for external use"""
        if ref == 'tref':
            self._show_tref(enable)
        elif ref == 'vref':
            self._show_vref(enable)
        elif ref == 'caret':
            self._show_caret(enable)
        else:
            raise ValueError, 'invalid ref: %r' % ref
        self.draw_refs()

    def draw_refs(self):
        """Redraws all enabled reflines, resaves reflines_background"""
        showns = {}
        for plotid, plot in self.used_plots.items():
            shown = plot.get_shown_chans()
            showns[plotid] = shown
            plot.hide_chans(shown)
        self.draw() # draw all the enabled refs
        self.reflines_background = self.copy_from_bbox(self.ax.bbox) # update
        for plotid, plot in self.used_plots.items():
            shown = showns[plotid]
            plot.show_chans(shown) # re-show just the chans that were shown previously
            plot.draw()
        self.blit(self.ax.bbox)

    def _add_tref(self):
        """Add vertical time reference line(s)"""
        # get column x positions
        cols = list(set([ xpos for chan, (xpos, ypos) in self.pos.iteritems() ]))
        ylims = self.ax.get_ylim()
        self.vlines = []
        for col in cols:
            vline = Line2D([col, col],
                           ylims,
                           linewidth=TREFLINEWIDTH,
                           color=TREFCOLOUR,
                           zorder=REFLINEZORDER,
                           antialiased=True,
                           visible=False)
            self.vlines.append(vline)
            self.ax.add_line(vline)

    def _add_vref(self):
        """Add horizontal voltage reference lines"""
        self.hlines = []
        for chan, (xpos, ypos) in self.pos.items():
            hline = Line2D([xpos-self.tw/2, xpos+self.tw/2],
                           [ypos, ypos],
                           linewidth=VREFLINEWIDTH,
                           color=VREFCOLOUR,
                           zorder=REFLINEZORDER,
                           antialiased=True,
                           visible=False)
            hline.chan = chan
            hline.set_pickradius(0) # don't normally want these to be pickable
            self.hlines.append(hline)
            self.ax.add_line(hline)

    def _add_caret(self):
        """Add a shaded rectangle to represent the time window shown in the spike frame"""
        ylim = self.ax.get_ylim()
        xy = (-self.cw/2, ylim[0]) # bottom left coord of rectangle
        width = self.cw
        height = ylim[1] - ylim[0]
        self.caret = Rectangle(xy, width, height,
                               facecolor=CARETCOLOUR,
                               zorder=CARETZORDER,
                               linewidth=0,
                               antialiased=False,
                               visible=False)
        self.ax.add_patch(self.caret)

    def _update_tref(self):
        """Update position and size of vertical time reference line(s)"""
        cols = list(set([ xpos for chan, (xpos, ypos) in self.pos.iteritems() ]))
        ylims = self.ax.get_ylim()
        for col, vline in zip(cols, self.vlines):
            vline.set_data([col, col], ylims)

    def _update_vref(self):
        """Update position and size of horizontal voltage reference lines"""
        for (xpos, ypos), hline in zip(self.pos.itervalues(), self.hlines):
            hline.set_data([xpos-self.tw/2, xpos+self.tw/2], [ypos, ypos])

    def _update_caret_width(self):
        """Update caret width"""
        # bottom left coord of rectangle
        self.caret.set_x(-self.cw/2)
        self.caret.set_width(self.cw)

    def _show_tref(self, enable=True):
        try:
            self.vlines
        except AttributeError:
            self._add_tref()
        for vline in self.vlines:
            vline.set_visible(enable)

    def _show_vref(self, enable=True):
        try:
            self.hlines
        except AttributeError:
            self._add_vref()
        for hline in self.hlines:
            hline.set_visible(enable)

    def _show_caret(self, enable=True):
        try:
            self.caret
        except AttributeError:
            self._add_caret()
        self.caret.set_visible(enable)

    def get_spatialchans(self, order='vertical'):
        """Return channels in spatial order.
        order='vertical': sort from bottom to top, left to right
        order='horziontal': sort from left to right, bottom to top
        TODO: fix code duplication"""
        if order == 'vertical':
            # first, sort x coords, then y: (secondary, then primary)
            xychans = [ (x, y, chan) for chan, (x, y) in self.siteloc.items() ] # list of (x, y, chan) 3-tuples
            xychans.sort() # stable sort in-place according to x values (first in tuple)
            yxchans = [ (y, x, chan) for (x, y, chan) in xychans ]
            yxchans.sort() # stable sort in-place according to y values (first in tuple)
            chans = [ chan for (y, x, chan) in yxchans ] # unload the chan indices, now sorted bottom to top, left to right
        elif order == 'horizontal':
            # first, sort y coords, then x: (secondary, then primary)
            yxchans = [ (y, x, chan) for chan, (x, y) in self.siteloc.items() ] # list of (y, x, chan) 3-tuples
            yxchans.sort() # stable sort in-place according to y values (first in tuple)
            xychans = [ (x, y, chan) for (y, x, chan) in yxchans ] # list of (x, y, chan) 3-tuples
            xychans.sort() # stable sort in-place according to x values (first in tuple)
            chans = [ chan for (x, y, chan) in xychans ] # unload the chan indices, now sorted left to right, bottom to top
        else:
            raise ValueError
        return chans

    def get_xy_um(self):
        """Pull xy tuples in um out of self.pos, store in (2 x nchans) array,
        in self.chans order. In chart and lfp frames, this is different from siteloc,
        since these frames have only a single column"""
        xy_um = np.asarray([ (self.us2um(self.pos[chan][0]), self.uv2um(self.pos[chan][1]))
                                  for chan in self.chans ]).T # x is row0, y is row1
        return xy_um
    '''
    def get_spatial_chan_array(self):
        """build up 2d array of chanis, with cols and rows sorted according to probe layout"""
        x, y = self.xy_um # in self.chans order
        # should correspond to unique columns in spike frame, or just single column in chart or lfp frame
        a = []
        uniquexs = set(x)
        for uniquex in uniquexs:
            x == uniquexs
    '''
    def get_closestchans(self, event, n=1):
        """Return n channels in column closest to mouse event coords,
        sorted by vertical distance from mouse event"""

        # sum of squared distances
        #d2 = (x-xdata)**2 + (y-ydata)**2
        #i = d2.argsort()[:n] # n indices sorted from smallest squared distance to largest

        # what column is this event closest to? pick that column,
        # and then the n vertically closest chans within it
        xdata = self.us2um(event.xdata) # convert mouse event to um
        ydata = self.uv2um(event.ydata)
        x, y = self.xy_um
        # find nearest column
        dx = np.abs(xdata - self.colxs) # array of x distances
        coli = dx.argmin() # index of column nearest to mouse click
        colx = self.colxs[coli] # x coord of nearest column
        i, = (x == colx).nonzero() # indices into self.chans of chans that are in the nearest col
        colchans = np.asarray(self.chans)[i] # channels in nearest col
        dy = np.abs(y[i] - ydata) # vertical distances between mouse click and all chans in this col
        i = dy.argsort()[:n] # n indices sorted from smallest to largest y distance
        chans = colchans[i] # index into channels in the nearest column
        if len(chans) == 1:
            chans = chans[0] # pull it out, return a single value
        return chans

    def get_closestline(self, event):
        """Return line that's closest to mouse event coords"""
        d2s = [] # sum squared distances
        hitlines = []
        closestchans = self.get_closestchans(event, n=NCLOSESTCHANSTOSEARCH)
        for chan in closestchans:
            line = self.quickRemovePlot.lines[chan]
            if line.get_visible(): # only consider lines that are visible
                hit, tisdict = line.contains(event)
                if hit:
                    tis = tisdict['ind'] # pull them out of the dict
                    xs = line.get_xdata()[tis]
                    ys = line.get_ydata()[tis]
                    d2 = (xs-event.xdata)**2 + (ys-event.ydata)**2
                    d2 = d2.min() # point on line closest to mouse
                    hitlines.append(line)
                    d2s.append(d2)
        d2s = np.asarray(d2s)
        if d2s.size != 0:
            linei = d2s.argmin() # index of line with smallest d2
            return hitlines[linei]
        else:
            return None

    def plot(self, wave, tref=None):
        """Plot waveforms wrt a reference time point"""
        if tref == None:
            tref = wave.ts[0] # use the first timestamp in the waveform as the reference time point
        self.restore_region(self.reflines_background)
        # update plots with new x and y vals
        self.quickRemovePlot.update(wave, tref)
        self.quickRemovePlot.draw()
        self.blit(self.ax.bbox)
        #self.gui_repaint()
        #self.draw(True)
        #self.Refresh() # possibly faster, but adds a lot of flicker

    def _zoomx(self, x):
        """Zoom x axis by factor x"""
        self.tw /= x
        self.usperum /= x
        self.do_layout() # resets axes lims and recalcs self.pos
        self._update_tref()
        self._update_vref()
        self.post_motion_notify_event() # forces tooltip update, even if mouse hasn't moved

    def post_motion_notify_event(self):
        """Posts a motion_notify_event to mpl's event queue"""
        x, y = wx.GetMousePosition() - self.GetScreenPosition() # get mouse pos relative to this window
        # now just mimic what mpl FigureCanvasWx._onMotion does
        y = self.figure.bbox.height - y
        FigureCanvasBase.motion_notify_event(self, x, y, guiEvent=None) # no wx event to pass as guiEvent

    def um2uv(self, um):
        """Vertical conversion from um in channel siteloc
        space to uV in signal space"""
        return self.uVperum * um

    def uv2um(self, uV):
        """Convert from uV to um"""
        return uV / self.uVperum

    def um2us(self, um):
        """Horizontal conversion from um in channel siteloc
        space to us in signal space"""
        return self.usperum * um

    def us2um(self, us):
        """Convert from us to um"""
        return us / self.usperum

    def OnKeyDown(self, evt):
        """Let main spyke frame handle keypress events"""
        self.spykeframe.OnKeyDown(evt)
        #key = evt.GetKeyCode()
        #print 'in dataframe.onkeypress !!: ', key
        #evt.guiEvent.Skip()
    '''
    def OnNavigation(self, evt):
        """Navigation key press"""
        #print 'nagivation event:', evt
        if evt.GetDirection(): # forward
            direction = 1
        else: # backward
            direction = -1
        self.spykeframe.seek(self.quickRemovePlot.tref + direction*self.stream.tres)
        evt.Skip() # allow left, right, pgup and pgdn to propagate to OnKeyDown handler
    '''
    def OnButtonPress(self, evt):
        """Seek to timepoint as represented on chan closest to left mouse click,
        enable/disable specific chans on Ctrl+left click or right click, enable/disable
        all chans on Shift+left click"""
        button = evt.guiEvent.GetButton()
        ctrl = evt.guiEvent.ControlDown()
        shift = evt.guiEvent.ShiftDown()
        #dclick = evt.guiEvent.ButtonDClick(but=wx.MOUSE_BTN_LEFT)
        if button == wx.MOUSE_BTN_LEFT and not ctrl and not shift:
            # seek to timepoint
            chan = self.get_closestchans(evt, n=1)
            xpos = self.pos[chan][0]
            t = evt.xdata - xpos + self.quickRemovePlot.tref # undo position correction and convert from relative to absolute time
            self.spykeframe.seek(t) # call main spyke frame's seek method
        elif button == wx.MOUSE_BTN_LEFT and ctrl and not shift: # or button == wx.MOUSE_BTN_RIGHT and not ctrl and not shift:
            # enable/disable closest line
            chan = self.get_closestchans(evt, n=1)
            line = self.quickRemovePlot.lines[chan]
            if line.chan not in self.spykeframe.chans_enabled:
                enable = True
            else:
                enable = False
            self.spykeframe.set_chans_enabled(line.chan, enable)
        elif button == wx.MOUSE_BTN_LEFT and not ctrl and shift:
            # enable/disable all chans
            if len(self.spykeframe.chans_enabled) == 0:
                enable = True
            else:
                enable = False
            self.spykeframe.set_chans_enabled(None, enable) # None means all chans

    def enable_chans(self, chans, enable=True):
        """Enable/disable a specific set of channels in this frame"""
        for chan in chans:
            self.quickRemovePlot.lines[chan].set_visible(enable)
        self.draw(True)
    '''
    def OnPick(self, evt):
        """Pop up a tooltip when mouse is within PICKTHRESH of a line"""
        tooltip = self.GetToolTip()
        if evt.mouseevent.inaxes:
            line = evt.artist # assume it's one of our SpykeLines, since those are the only ones with their .picker attrib enabled
            chan = line.chan
            xpos, ypos = self.pos[chan]
            t = evt.mouseevent.xdata - xpos + self.quickRemovePlot.tref # undo position correction and convert from relative to absolute time
            v = (evt.mouseevent.ydata - ypos) / self.gain
            if t >= self.stream.t0 and t <= self.stream.tend: # in bounds
                t = intround(t / self.stream.tres) * self.stream.tres # round to nearest (possibly interpolated) sample
                tip = 'ch%d\n' % chan + \
                      't=%d %s\n' % (t, MU+'s') + \
                      'V=%.1f %s\n' % (v, MU+'V') + \
                      'width=%.3f ms' % (self.tw/1000)
                tooltip.SetTip(tip)
                tooltip.Enable(True)
            else: # out of bounds
                tooltip.Enable(False)
        else:
            tooltip.Enable(False)
    '''
    def OnMotion(self, evt):
        """Pop up a tooltip when figure mouse movement is over axes"""
        tooltip = self.GetToolTip()
        if evt.inaxes:
            # or, maybe better to just post a pick event, and let the pointed to chan
            # (instead of clicked chan) stand up for itself
            #chan = self.get_closestchans(evt, n=1)
            line = self.get_closestline(evt)
            if line and line.get_visible():
                xpos, ypos = self.pos[line.chan]
                t = evt.xdata - xpos + self.quickRemovePlot.tref
                v = (evt.ydata - ypos) / self.gain
                if t >= self.stream.t0 and t <= self.stream.tend: # in bounds
                    t = intround(t / self.stream.tres) * self.stream.tres # round to nearest (possibly interpolated) sample
                    tip = 'ch%d\n' % line.chan + \
                          't=%d %s\n' % (t, MU+'s') + \
                          'V=%.1f %s\n' % (v, MU+'V') + \
                          'width=%.3f ms' % (self.tw/1000)
                    tooltip.SetTip(tip)
                    tooltip.Enable(True)
                else:
                    tooltip.Enable(False)
            else:
                tooltip.Enable(False)
        else:
            tooltip.Enable(False)

    def OnMouseWheel(self, evt):
        """Zoom horizontally on CTRL+mouse wheel scroll"""
        if evt.ControlDown():
            #lines = evt.GetWheelRotation() / evt.GetWheelDelta() # +ve or -ve num lines to scroll
            #x = 1.1**lines # transform -ve line to 0<x<1, and +ve line to 1<x<inf
            #self._zoomx(x)
            sign = np.sign(evt.GetWheelRotation())
            self._zoomx(1.5**sign)


class SpikePanel(PlotPanel):
    """Spike panel. Presents a narrow temporal window of all channels
    layed out according to self.siteloc"""
    def __init__(self, *args, **kwargs):
        PlotPanel.__init__(self, *args, **kwargs)
        self.gain = 1.5

    def do_layout(self):
        self.hchans = self.get_spatialchans('horizontal') # ordered left to right, bottom to top
        self.vchans = self.get_spatialchans('vertical') # ordered bottom to top, left to right
        #print 'horizontal ordered chans in Spikepanel:\n%r' % self.hchans
        self.ax.set_xlim(self.um2us(self.siteloc[self.hchans[0]][0]) - self.tw/2,
                         self.um2us(self.siteloc[self.hchans[-1]][0]) + self.tw/2) # x origin at center
        self.ax.set_ylim(self.um2uv(self.siteloc[self.vchans[0]][1]) - CHANVBORDER,
                         self.um2uv(self.siteloc[self.vchans[-1]][1]) + CHANVBORDER)
        colourgen = itertools.cycle(iter(COLOURS))
        for chan in self.vchans:
            # chan order doesn't matter for setting .pos, but it does for setting .colours
            self.pos[chan] = (self.um2us(self.siteloc[chan][0]),
                              self.um2uv(self.siteloc[chan][1]))
            self.vcolours[chan] = colourgen.next() # assign colours so that they cycle vertically in space

    def _zoomx(self, x):
        """Zoom x axis by factor x"""
        PlotPanel._zoomx(self, x)
        # update main spyke frame so its plot calls send the right amount of data
        self.spykeframe.spiketw = self.tw
        self.spykeframe.frames['chart'].panel.cw = self.tw
        self.spykeframe.frames['chart'].panel._update_caret_width()
        self.spykeframe.plot(frametypes='spike') # replot

    def _add_caret(self):
        """Disable for SpikePanel"""
        pass

    def _update_caret_width(self):
        """Disable for SpikePanel"""
        pass

    def _show_caret(self, enable=True):
        """Disable for SpikePanel"""
        pass


class ChartPanel(PlotPanel):
    """Chart panel. Presents all channels layed out vertically according
    to the vertical order of their site coords in .siteloc"""
    def __init__(self, *args, **kwargs):
        PlotPanel.__init__(self, *args, **kwargs)
        self.gain = 1

    def do_layout(self):
        """Sets axes limits and calculates self.pos"""
        self.vchans = self.get_spatialchans('vertical') # ordered bottom to top, left to right
        self.ax.set_xlim(0 - self.tw/2, 0 + self.tw/2) # x origin at center
        miny = self.um2uv(self.siteloc[self.vchans[0]][1])
        maxy = self.um2uv(self.siteloc[self.vchans[-1]][1])
        vspace = (maxy - miny) / (self.nchans-1) # average vertical spacing between chans, in uV
        self.ax.set_ylim(miny - CHANVBORDER, maxy + CHANVBORDER)
        colourgen = itertools.cycle(iter(COLOURS))
        for chani, chan in enumerate(self.vchans):
            #self.pos[chan] = (0, self.um2uv(self.siteloc[chan][1])) # x=0 centers horizontally
            self.pos[chan] = (0, chani*vspace) # x=0 centers horizontally, equal vertical spacing
            self.vcolours[chan] = colourgen.next() # assign colours so that they cycle vertically in space

    def _zoomx(self, x):
        """Zoom x axis by factor x"""
        PlotPanel._zoomx(self, x)
        # update main spyke frame so its plot calls send the right amount of data
        self.spykeframe.charttw = self.tw
        self.spykeframe.frames['lfp'].panel.cw = self.tw
        self.spykeframe.frames['lfp'].panel._update_caret_width()
        self.spykeframe.plot(frametypes='chart') # replot

    def _add_vref(self):
        """Disable for ChartPanel"""
        pass

    def _update_vref(self):
        """Disable for ChartPanel"""
        pass

    def _show_vref(self, enable=True):
        """Disable for ChartPanel"""
        pass

    def _update_caret_width(self):
        """Set optimal paint method"""
        PlotPanel._update_caret_width(self)
        #self.draw(True) # can be quite slow
        self.Refresh() # can be faster, but adds flicker


class LFPPanel(ChartPanel):
    """LFP Panel"""
    def __init__(self, *args, **kwargs):
        ChartPanel.__init__(self, *args, **kwargs)
        self.gain = 1

    def do_layout(self):
        ChartPanel.do_layout(self)
        # need to specifically get a list of keys, not an iterator,
        # since self.pos dict changes size during iteration
        # don't remember why this was sometimes necessary to do:
        for chan in self.pos.keys():
            if chan not in self.stream.chans:
                del self.pos[chan] # remove siteloc channels that don't exist in the lowpassmultichan record
                try:
                    self.chans.remove(chan) # in place
                except ValueError: # already removed from list on previous do_layout() call
                    pass

    def _zoomx(self, x):
        """Zoom x axis by factor x"""
        PlotPanel._zoomx(self, x)
        # update main spyke frame so its plot calls send the right amount of data
        self.spykeframe.lfptw = self.tw
        self.spykeframe.plot(frametypes='lfp') # replot

    def _add_vref(self):
        """Override ChartPanel"""
        PlotPanel._add_vref(self)

    def _update_vref(self):
        """Override ChartPanel"""
        PlotPanel._update_vref(self)

    def _show_vref(self, enable=True):
        """Override ChartPanel"""
        PlotPanel._show_vref(self, enable)

    def _update_caret_width(self):
        """Set optimal paint method"""
        PlotPanel._update_caret_width(self)
        self.draw(True)
        #self.Refresh() # possibly faster, but adds a lot of flicker


class SortPanel(PlotPanel):
    """A plot panel specialized for overplotting spike events and templates"""
    def __init__(self, *args, **kwargs):
        PlotPanel.__init__(self, *args, **kwargs)
        #self.spykeframe = self.GetTopLevelParent().Parent

    def init_plots(self, nplots=DEFNPLOTS):
        """Add Plots to the pool of available ones"""
        totalnplots = len(self.available_plots) + len(self.used_plots) # total number of existing plots
        for ploti in range(totalnplots, totalnplots+nplots):
            plot = Plot(chans=self.chans, panel=self)
            self.available_plots.append(plot)

    def _add_vref(self):
        """Increase pick radius for vrefs from default zero, since we're
        relying on them for tooltips"""
        PlotPanel._add_vref(self)
        for hline in self.hlines:
            hline.set_pickradius(PICKRADIUS)

    def show_ref(self, ref, enable=True):
        PlotPanel.show_ref(self, ref, enable)
        self.quickRemovePlot = None
        self.background = None

    def addObjects(self, objects):
        """Add events/templates to self"""
        if objects == []:
            return # do nothing
        if len(objects) == 1:
            # before blitting this single object to screen, grab current buffer,
            # save as new background for quick restore if the next action is removal of this very same object
            self.background = self.copy_from_bbox(self.ax.bbox)
            self.quickRemovePlot = self.addObject(objects[0]) # add the single object, save reference to its plot
            #print 'saved quick remove plot %r' % self.quickRemovePlot
        else:
            self.background = None
            for obj in objects: # add all objects
                self.addObject(obj)
        self.blit(self.ax.bbox)

    def addObject(self, obj):
        """Put object in an available Plot, return the Plot"""
        if len(self.available_plots) == 0: # if we've run out of plots for additional events
            self.init_plots() # init another batch of plots
        plot = self.available_plots.pop() # pop a Plot to assign this object to
        try:
            obj.events # it's a template
            plot.id = 't' + str(obj.id)
            colours = [COLOURDICT[obj.id]]
            alpha = 1
            style = TEMPLATELINESTYLE
            width = TEMPLATELINEWIDTH
        except AttributeError: # it's an event
            plot.id = 'e' + str(obj.id)
            style = EVENTLINESTYLE
            width = EVENTLINEWIDTH
            try:
                obj.template # it's a member event of a template. colour it the same as its template
                alpha = 0.5
                colours = [COLOURDICT[obj.template.id]]
            except AttributeError: # it's an unsorted spike, colour each chan separately
                alpha = 1
                colours = [ self.vcolours[chan] for chan in plot.chans ] # remap to cycle vertically in space
        plot.set_colours(colours)
        plot.set_alpha(alpha)
        plot.set_stylewidth(style, width)
        plot.obj = obj # bind object to plot
        obj.plot = plot  # bind plot to object
        if obj.wave.data == None: # if it hasn't already been loaded
            obj.update_wave()
        self.used_plots[plot.id] = plot # push it to the used plot stack
        wave = obj.wave[obj.t-self.tw/2 : obj.t+self.tw/2] # slice wave according to the width of this panel
        plot.update(wave, obj.t)
        plot.show_chans(obj.chans) # unhide object's enabled chans
        plot.draw()
        return plot

    def removeObjects(self, objects):
        """Remove objects from plots"""
        if objects == []: # do nothing
            return
        for obj in objects:
            # remove specified objects from .used_plots, use contents of
            # .used_plots to decide how to do the actual plot removal
            plot = self.removeObject(obj)
        # remove all objects
        if self.used_plots == {}:
            self.restore_region(self.reflines_background) # restore blank background with just the ref lines
        # remove the last added plot if a saved bg is available
        elif len(objects) == 1 and plot == self.quickRemovePlot and self.background != None:
            #print 'quick removing plot %r' % self.quickRemovePlot
            self.restore_region(self.background) # restore saved bg
        # remove more than one, but not all objects
        else:
            self.restore_region(self.reflines_background) # restore blank background with just the ref lines
            for plot in self.used_plots.values():
                plot.draw() # redraw the remaining plots in .used_plots
        self.background = None # what was background is no longer useful for quick restoration on any other object removal
        self.quickRemovePlot = None # quickRemovePlot set in addObjects is no longer quickly removable
        self.blit(self.ax.bbox) # blit everything to screen

    def removeAllObjects(self):
        """Shortcut for removing all object from plots"""
        objects = [ plot.obj for plot in self.used_plots.values() ]
        self.removeObjects(objects)

    def removeObject(self, obj):
        """Restore object's Plot from used to available plot pool, return the Plot"""
        plot = self.used_plots.pop(obj.plot.id)
        # TODO: reset plot colour and line style here, or just set them each time in addObject?
        plot.id = None # clear its index into .used_plots
        plot.obj = None # unbind object from plot
        obj.plot = None # unbind plot from object
        plot.hide() # hide all chan lines
        self.available_plots.append(plot)
        return plot

    def updateObjects(self, objects):
        """Re-plot objects, potentially because their WaveForms have changed.
        Typical use case: event is added to a template, template's mean waveform has changed"""
        if objects == []: # do nothing
            return
        if len(objects) == 1 and objects[0].plot != None and objects[0].plot == self.quickRemovePlot and self.background != None:
            print 'quick removing and replotting plot %r' % self.quickRemovePlot
            self.restore_region(self.background) # restore saved bg
            self.updateObject(objects[0])
        else: # update and redraw all objects
            self.restore_region(self.reflines_background) # restore blank background with just the ref lines
            for obj in objects:
                self.updateObject(obj)
            self.background = None # what was background is no longer useful for quick restoration on any other object removal
            self.quickRemovePlot = None # quickRemovePlot set in addObjects is no longer quickly removable
        self.blit(self.ax.bbox) # blit everything to screen

    def updateObject(self, obj):
        """Update and draw an event's/template's plot"""
        wave = obj.wave[obj.t-self.tw/2 : obj.t+self.tw/2] # slice wave according to the width of this panel
        obj.plot.update(wave, obj.t)
        obj.plot.draw()

    def get_closestline(self, evt):
        """Return line that's closest to mouse event coords
        Slightly modified from PlotPanel's version"""
        d2s = [] # sum squared distances
        hitlines = []
        closestchans = self.get_closestchans(evt, n=NCLOSESTCHANSTOSEARCH)
        for chan in closestchans:
            line = self.hlines[chan] # consider all hlines, even if invisible
            hit, tisdict = line.contains(evt)
            if hit:
                tis = tisdict['ind'] # pull them out of the dict
                xs = line.get_xdata()[tis]
                ys = line.get_ydata()[tis]
                d2 = (xs-evt.xdata)**2 + (ys-evt.ydata)**2
                d2 = d2.min() # point on line closest to mouse
                hitlines.append(line)
                d2s.append(d2)
        d2s = np.asarray(d2s)
        if d2s.size != 0:
            linei = d2s.argmin() # index of line with smallest d2
            return hitlines[linei]
        else:
            return None

    def OnMotion(self, evt):
        """Pop up a tooltip when figure mouse movement is over axes.
        Slightly modified from PlotPanel's version"""
        tooltip = self.GetToolTip()
        if evt.inaxes:
            # or, maybe better to just post a pick event, and let the pointed to chan
            # (instead of clicked chan) stand up for itself
            #chan = self.get_closestchans(evt, n=1)
            line = self.get_closestline(evt) # get closest hline only
            if line:
                xpos, ypos = self.pos[line.chan]
                try:
                    tres = self.spykeframe.hpstream.tres
                except AttributeError: # no srf file loaded in spyke
                    tres = 1
                t = evt.xdata - xpos # make it relative to the vertical tref line only, don't try to get absolute times
                v = (evt.ydata - ypos) / self.gain
                t = intround(t / tres) * tres # round to nearest (possibly interpolated) sample
                tip = 'ch%d\n' % line.chan + \
                      't=%d %s\n' % (t, MU+'s') + \
                      'V=%.1f %s\n' % (v, MU+'V') + \
                      'width=%.3f ms' % (self.tw/1000)
                tooltip.SetTip(tip)
                tooltip.Enable(True)
            else:
                tooltip.Enable(False)
        else:
            tooltip.Enable(False)


class SpikeSortPanel(SortPanel, SpikePanel):
    def __init__(self, *args, **kwargs):
        kwargs['tw'] = SPIKESORTTW
        SortPanel.__init__(self, *args, **kwargs)
        self.gain = 1.5


class ChartSortPanel(SortPanel, ChartPanel):
    def __init__(self, *args, **kwargs):
        kwargs['tw'] = CHARTSORTTW
        kwargs['cw'] = SPIKESORTTW
        SortPanel.__init__(self, *args, **kwargs)
        self.gain = 1.5

    def _add_vref(self):
        """Override ChartPanel, use vrefs for tooltips"""
        SortPanel._add_vref(self)
