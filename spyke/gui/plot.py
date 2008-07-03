"""wx.Panels with embedded mpl figures based on FigureCanvasWxAgg.
Everything is plotted in units of uV and us

TODO: perhaps refactor, keep info about each channel together,
make a Channel object with .id, .pos, .colour, .line, .enabled properties,
and set_enable() method, and then stick them in a dict of chans indexed by id
    - looks like I've more or less done this with the Plot object"""

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

from spyke import surf
from spyke.core import MU, intround

SPIKELINEWIDTH = 1 # mpl units - pixels? points? plot units (us)?
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

DEFSPIKESORTTW = 1000 # spike sort panel temporal window width (us)
DEFCHARTSORTTW = 2000 # chart sort panel temporal window width (us)
DEFEVENTTW = max(DEFSPIKESORTTW, DEFCHARTSORTTW) # default event time width, determines event.wave width
DEFNPLOTS = 10 # default number of plots to init in SortPanel

CARETZORDER = 0 # layering
REFLINEZORDER = 1
PLOTZORDER = 2


class Plot(object):
    """Plot slot, holds lines for all chans for plotting
    a single stretch of data, contiguous in time"""
    def __init__(self, chans, panel):
        self.lines = {} # chan to line mapping
        self.panel = panel # panel that self belongs to
        self.chans = chans # all channels available in this Plot, lines can be enabled/disabled
        self.colours = self.panel.colours
        #self.background = None
        for chan in self.chans:
            line = Line2D([0], # x and y data are just placeholders for now
                          [0], # TODO: will such a small amount of data before first .draw() cause problems for blitting?
                          linewidth=SPIKELINEWIDTH,
                          color=self.colours[chan],
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

    def show(self, enable=True):
        """Show/hide all chans in self"""
        for line in self.lines.values():
            line.set_visible(enable)

    def hide(self):
        """Hide all chans in self"""
        self.show(False)

    def update(self, wave, tref):
        """Update lines data
        TODO: most of the time, updating the xdata won't be necessary,
        but I think updating takes no time at all relative to drawing time"""
        self.tref = tref
        for chan, line in self.lines.items():
            xpos, ypos = self.panel.pos[chan]
            xdata = wave.ts - tref + xpos
            ydata = wave[chan]*self.panel.gain + ypos
            line.set_data(xdata, ydata) # update the line's x and y data

    def set_animated(self, enable=True):
        """Set animated flag for all lines in self"""
        for line in self.lines.values():
            line.set_animated(enable)

    def set_colours(self, colours):
        """Set colour(s) for all lines in self"""
        colours = toiter(colours)
        if len(colours) == 1:
            colours = colours * len(self.chans)
        if len(colours) != len(self.chans):
            raise ValueError, 'invalid colours length: %d' % len(colours)
        self.colours = colours # now safe to save it
        for chani, colour in enumerate(self.colours):
            self.lines[chani].set_color(colour)

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
        self.spykeframe = self.GrandParent

        self.available_plots = [] # pool of available Plots
        self.event_plots = {} # Plots holding currently displayed event data, indexed by event id
        self.template_plots = {} # Plots holding currently displayed template mean, indexed by template id
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

    def get_used_plots(self):
        return list(np.concatenate((self.event_plots, self.template_plots)))

    def set_used_plots(self):
        raise RunTimeError, "PlotPanel's .used_plots not setable"

    used_plots = property(get_used_plots, set_used_plots)

    def callAfterFrameInit(self, probe=None):
        """Panel tasks that need to be done after parent frame has been created (and shown?)"""
        if probe == None:
            probe = self.stream.probe
        self.probe = probe
        self.SiteLoc = probe.SiteLoc # probe site locations with origin at center top
        self.chans = probe.SiteLoc.keys()
        self.chans.sort() # a sorted list of chans, keeps us from having to do this over and over
        self.nchans = probe.nchans
        self.colours = dict(zip(range(self.nchans), [DEFAULTCHANCOLOUR]*self.nchans))
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
        self.event_plots[0] = self.quickRemovePlot

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
        for plot in self.event_plots.values():
            plot.hide()
        self.draw() # draw the new ref
        self.reflines_background = self.copy_from_bbox(self.ax.bbox) # update
        for plot in self.event_plots.values():
            plot.show()
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
            self.colours[chan] = colourgen.next() # assign colours so that they cycle nicely in space

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
            self.colours[chan] = colourgen.next() # assign colours so that they cycle nicely in space

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
        for chan in self.pos.keys():
            if chan not in self.stream.layout.chanlist:
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
    """A plot panel specialized for overplotting spike events"""
    def __init__(self, *args, **kwargs):
        PlotPanel.__init__(self, *args, **kwargs)
        self.spykeframe = self.GrandParent.GrandParent # plot pane, splitter, sort frame, spyke frame

    def init_plots(self, nplots=DEFNPLOTS):
        """Add Plots to the pool of available ones"""
        totalnplots = len(self.available_plots) + len(self.event_plots) # total number of existing plots
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

    def addEvents(self, events):
        """Add events to self"""
        if events == []:
            return # do nothing
        if len(events) == 1:
            # before blitting this single event to screen, grab current buffer, save as new background for quick restore if the next action is removal of this very same event
            self.background = self.copy_from_bbox(self.ax.bbox)
            self.quickRemovePlot = self.addEvent(events[0]) # add the single event, save reference to its plot
            print 'saved quick remove plot %r' % self.quickRemovePlot
        else:
            self.background = None
            for event in events: # add all events
                self.addEvent(event)
        self.blit(self.ax.bbox)

    def addEvent(self, event):
        """Put event in an available Plot, return the Plot"""
        t = event.t
        try:
            wave = event.wave # see if it's already been sliced
        except AttributeError:
            wave = event.load_wave(trange=(-DEFEVENTTW/2, DEFEVENTTW/2)) # this binds .wave to event
        wave = wave[t-self.tw/2 : t+self.tw/2] # slice it according to the width of this panel
        if len(self.available_plots) == 0: # if we've run out of plots for additional events
            self.init_plots() # init another batch of plots
        plot = self.available_plots.pop() # pop a Plot to assign this event to
        self.event_plots[event.id] = plot # push it to the event plot stack
        plot.update(wave, t)
        plot.show()
        plot.draw()
        return plot

    def removeEvents(self, events=None):
        """Remove event plots from self, events=None means remove all"""
        if events == []: # do nothing
            return
        if events == None:
            events = self.event_plots.keys()
        for event in events:
            # remove all specified events from .event_plots, use contents of
            # .event_plots to decide how to do the actual plot removal
            plot = self.removeEvent(event)
        # remove all events
        if self.event_plots.keys() == []:
            self.restore_region(self.reflines_background) # restore blank background with just the ref lines
        # remove the last added plot if a saved bg is available
        elif len(events) == 1 and plot == self.quickRemovePlot and self.background != None:
            print 'quick removing plot %r' % self.quickRemovePlot
            self.restore_region(self.background) # restore saved bg
        # remove more than one, but not all events
        else:
            self.restore_region(self.reflines_background) # restore blank background with just the ref lines
            for plot in self.event_plots.values():
                plot.draw() # redraw the remaining plots in .event_plots
        self.background = None # what was background is no longer useful for quick restoration on any other event removal
        self.quickRemovePlot = None # quickRemovePlot set in addEvents is no longer quickly removable
        self.blit(self.ax.bbox) # blit everything to screen

    def removeAllEvents(self):
        """Shortcut for removing all event plots"""
        self.removeEvents(events=None)

    def removeEvent(self, event):
        """Restore Plot holding event's data from used to available plot pool, return the Plot"""
        try:
            eventi = event.id # it's an Event object
        except AttributeError:
            eventi = event # it's just an int denoting the event id
        plot = self.event_plots.pop(eventi)
        plot.hide()
        self.available_plots.append(plot)
        return plot

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
        kwargs['tw'] = DEFSPIKESORTTW
        SortPanel.__init__(self, *args, **kwargs)
        self.gain = 1.5


class ChartSortPanel(SortPanel, ChartPanel):
    def __init__(self, *args, **kwargs):
        kwargs['tw'] = DEFCHARTSORTTW
        kwargs['cw'] = DEFSPIKESORTTW
        SortPanel.__init__(self, *args, **kwargs)
        self.gain = 1.5

    def _add_vref(self):
        """Override ChartPanel, use vrefs for tooltips"""
        SortPanel._add_vref(self)
