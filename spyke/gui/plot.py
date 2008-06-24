"""wx.Panels with embedded mpl figures based on FigureCanvasWxAgg.
Everything is plotted in units of uV and us

TODO: perhaps refactor, keep info about each channel together,
make a Channel object with .id, .pos, .colour, .line, .enabled properties,
and set_enable() method, and then stick them in a dict of chans indexed by id"""

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
#from spyke.gui.events import *

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
        colours = self.panel.colours
        #self.background = None
        for chan in self.chans:
            line = Line2D([0], # x and y data are just placeholders for now
                          [0], # TODO: will such a small amount of data before first .draw() cause problems for blitting?
                          linewidth=SPIKELINEWIDTH,
                          color=self.panel.colours[chan],
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
        self.tw = tw # temporal width of each channel, in plot units (us ostensibly)
        self.cw = cw # time width of caret, in plot units
        if stream != None: # either use stream kw here, or directly call init_probe_dependencies later from outside
            self.stream = stream
            self.init_probe_dependencies(stream.probe)
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

    def init_probe_dependencies(self, probe):
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
        self.siteloc = siteloc # bottom origin

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
        self.init_plots()
        #for ref in ['caret', 'vref', 'tref']: # add reference lines and caret in layered order
        #    self.add_ref(ref)
        #    self.spykeframe.ShowRef(ref) # also enforces menu item toggle state
        self.draw()
        self.background = self.copy_from_bbox(self.ax.bbox)

    def init_axes(self):
        """Init the axes and ref lines"""
        self.ax = self.figure.add_axes([0, 0, 1, 1], # lbwh relative to figure?
                                       axisbg=BACKGROUNDCOLOUR,
                                       frameon=False,
                                       alpha=1.)

    def init_plots(self):
        """Create Plots for this panel"""
        self.current_plot = Plot(chans=self.chans, panel=self) # just one for this base class
        self.current_plot.show()

    def add_ref(self, ref):
        if ref == 'tref':
            self._add_tref()
        elif ref == 'vref':
            self._add_vref()
        elif ref == 'caret':
            self._add_caret()

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

    def _update_tref(self):
        """Update position and size of vertical time reference line(s)"""
        cols = list(set([ xpos for chan, (xpos, ypos) in self.pos.iteritems() ]))
        ylims = self.ax.get_ylim()
        for col, vline in zip(cols, self.vlines):
            vline.set_data([col, col], ylims)

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

    def _update_vref(self):
        """Update position and size of horizontal voltage reference lines"""
        for (xpos, ypos), hline in zip(self.pos.itervalues(), self.hlines):
            hline.set_data([xpos-self.tw/2, xpos+self.tw/2], [ypos, ypos])

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

    def _update_caret_width(self):
        """Update caret"""
        # bottom left coord of rectangle
        self.caret.set_x(-self.cw/2)
        self.caret.set_width(self.cw)

    def update_background(self, plot=None):
        """Update background, exclude plot from it by temporarily setting its animated flag"""
        if plot != None: plot.set_animated(True)
        self.draw() # do a full draw of everything in self.ax
        self.background = self.copy_from_bbox(self.ax.bbox) # grab everything except plot
        if plot != None: plot.set_animated(False)

    def show_ref(self, ref, enable=True):
        if ref == 'tref':
            self._show_tref(enable)
        elif ref == 'vref':
            self._show_vref(enable)
        elif ref == 'caret':
            self._show_caret(enable)
        else:
            raise ValueError, 'invalid ref: %r' % ref
        self.update_background(self.current_plot) # update saved bg, exclude curret plot
        self.draw()

    def _show_tref(self, enable):
        try:
            self.vlines
        except AttributeError:
            self._add_tref()
        for vline in self.vlines:
            vline.set_visible(enable)

    def _show_vref(self, enable):
        try:
            self.hlines
        except AttributeError:
            self._add_vref()
        for hline in self.hlines:
            hline.set_visible(enable)

    def _show_caret(self, enable):
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
            line = self.current_plot.lines[chan]
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
        #self.update_background()
        self.restore_region(self.background)
        # update plots with new x and y vals
        self.current_plot.update(wave, tref)
        self.current_plot.draw()
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

    def OnKeyDown(self, event):
        """Let main spyke frame handle keypress events"""
        self.spykeframe.OnKeyDown(event)
        #key = event.GetKeyCode()
        #print 'in dataframe.onkeypress !!: ', key
        #event.guiEvent.Skip()

    def OnNavigation(self, event):
        """Navigation key press"""
        #print 'nagivation event:', event
        if event.GetDirection(): # forward
            direction = 1
        else: # backward
            direction = -1
        self.spykeframe.seek(self.current_plot.tref + direction*self.stream.tres)
        event.Skip() # allow left, right, pgup and pgdn to propagate to OnKeyDown handler

    def OnButtonPress(self, event):
        """Seek to timepoint as represented on chan closest to left mouse click,
        enable/disable specific chans on Ctrl+left click or right click, enable/disable
        all chans on Shift+left click"""
        button = event.guiEvent.GetButton()
        ctrl = event.guiEvent.ControlDown()
        shift = event.guiEvent.ShiftDown()
        #dclick = event.guiEvent.ButtonDClick(but=wx.MOUSE_BTN_LEFT)
        if button == wx.MOUSE_BTN_LEFT and not ctrl and not shift:
            # seek to timepoint
            chan = self.get_closestchans(event, n=1)
            xpos = self.pos[chan][0]
            t = event.xdata - xpos + self.current_plot.tref # undo position correction and convert from relative to absolute time
            self.spykeframe.seek(t) # call main spyke frame's seek method
        elif button == wx.MOUSE_BTN_LEFT and ctrl and not shift: # or button == wx.MOUSE_BTN_RIGHT and not ctrl and not shift:
            # enable/disable closest line
            chan = self.get_closestchans(event, n=1)
            line = self.current_plot.lines[chan]
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
            self.current_plot.lines[chan].set_visible(enable)
        self.draw(True)
    '''
    def OnPick(self, event):
        """Pop up a tooltip when mouse is within PICKTHRESH of a line"""
        tooltip = self.GetToolTip()
        if event.mouseevent.inaxes:
            line = event.artist # assume it's one of our SpykeLines, since those are the only ones with their .picker attrib enabled
            chan = line.chan
            xpos, ypos = self.pos[chan]
            t = event.mouseevent.xdata - xpos + self.current_plot.tref # undo position correction and convert from relative to absolute time
            v = (event.mouseevent.ydata - ypos) / self.gain
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
    def OnMotion(self, event):
        """Pop up a tooltip when figure mouse movement is over axes"""
        tooltip = self.GetToolTip()
        if event.inaxes:
            # or, maybe better to just post a pick event, and let the pointed to chan
            # (instead of clicked chan) stand up for itself
            #chan = self.get_closestchans(event, n=1)
            line = self.get_closestline(event)
            if line and line.get_visible():
                xpos, ypos = self.pos[line.chan]
                t = event.xdata - xpos + self.current_plot.tref
                v = (event.ydata - ypos) / self.gain
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

    def OnMouseWheel(self, event):
        """Zoom horizontally on CTRL+mouse wheel scroll"""
        if event.ControlDown():
            #lines = event.GetWheelRotation() / event.GetWheelDelta() # +ve or -ve num lines to scroll
            #x = 1.1**lines # transform -ve line to 0<x<1, and +ve line to 1<x<inf
            #self._zoomx(x)
            sign = np.sign(event.GetWheelRotation())
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

    def _show_caret(self, enable):
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

    def _show_vref(self, enable):
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

    def _show_vref(self, enable):
        """Override ChartPanel"""
        PlotPanel._show_vref(self, enable)

    def _update_caret_width(self):
        """Set optimal paint method"""
        PlotPanel._update_caret_width(self)
        self.draw(True)
        #self.Refresh() # possibly faster, but adds a lot of flicker


class SortPanel(PlotPanel):
    def __init__(self, *args, **kwargs):
        self.available_plots = [] # pool of available Plots
        self.used_plots = {} # Plots holding currently displayed event data, indexed by event id
        self.current_plot = None # current Plot being cycled between used and available
        PlotPanel.__init__(self, *args, **kwargs)
        self.spykeframe = self.GrandParent.GrandParent # sort pane, splitter window, sort frame, spyke frame

    def init_plots(self):
        """Create lines for multiple plots"""
        nplots = len(self.available_plots) + len(self.used_plots) # total number of existing plots
        for ploti in range(nplots, nplots+DEFNPLOTS):
            plot = Plot(chans=self.chans, panel=self)
            self.available_plots.append(plot)
        if self.current_plot == None:
            self.current_plot = self.available_plots[-1] # last one, top of stack
        # redraw the display
        #self.draw()

    def _add_vref(self):
        """Increase pick radius for vrefs from default zero, since we're
        relying on them for tooltips"""
        PlotPanel._add_vref(self)
        for hline in self.hlines:
            hline.set_pickradius(PICKRADIUS)

    # idea: have one background: black with ref lines. then, on each add(), you update current plot, draw the current plot's lines, and you blit the background _all_ the current used_plots to buffer in order. on remove(), you hide current plot, draw its lines?, then blit background and _all_ remaining used_plots to buffer in order. Might need to do a draw at very beginning (in init_lines?). no need to mess with animated flag!

    def add_event(self, event):
        """Add event to self, stick it in an available Plot"""
        print 'adding event %d' % event.id
        tref = event.t
        try:
            wave = event.wave # see if it's already been sliced
        except AttributeError:
            wave = event[tref-DEFEVENTTW/2 : tref+DEFEVENTTW/2] # slice it from the stream
            event.wave = wave # store it in the event, this will get pickled (hopefully)
        wave = event.wave[tref-self.tw/2 : tref+self.tw/2] # slice it according to the width of this panel
        if len(self.available_plots) == 0: # if we've run out of plots for additional events
            self.init_plots() # init another batch of plots
        plot = self.available_plots.pop() # pop a Plot to assign this event to
        self.used_plots[event.id] = plot # push it to the used plot stack
        if plot != self.current_plot: # if this isn't the plot that was last used
            print 'plot not same as last one'
            self.current_plot = plot # update current plot
            self.update_background(self.current_plot) # update bg, exclude current plot
        self.restore_region(self.background) # restore bg
        self.current_plot.update(wave, tref)
        self.current_plot.show()
        self.current_plot.draw()
        self.blit(self.ax.bbox)

    def remove_event(self, event):
        """Remove Plot holding event's data"""
        print 'removing event %d' % event.id
        plot = self.used_plots.pop(event.id)
        plot.hide()
        if plot == self.current_plot:
            self.restore_region(self.background) # restore saved bg
            self.blit(self.ax.bbox) # blit background to screen
        else: # do a full redraw
            self.current_plot = plot # update current plot
            #self.draw() # see below
            self.update_background(self.current_plot) # update bg, exclude current plot (which is invisible now anyway), does a full redraw
        # put it back in the available pool, at top of stack, ready to be popped
        self.available_plots.append(plot)

    def remove_all_events(self):
        """Remove all event Plots"""
        print 'removing all events'
        for ploti in self.used_plots.keys():
            plot = self.used_plots.pop(ploti)
            plot.hide()
            self.available_plots.append(plot)
        self.update_background(plot=None) # plot passed doesn't matter, since they're all hidden
        self.current_plot = self.available_plots[-1]

    def get_closestline(self, event):
        """Return line that's closest to mouse event coords
        Slightly modified from PlotPanel's version"""
        d2s = [] # sum squared distances
        hitlines = []
        closestchans = self.get_closestchans(event, n=NCLOSESTCHANSTOSEARCH)
        for chan in closestchans:
            line = self.hlines[chan] # consider all hlines, even if invisible
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

    def OnMotion(self, event):
        """Pop up a tooltip when figure mouse movement is over axes.
        Slightly modified from PlotPanel's version"""
        tooltip = self.GetToolTip()
        if event.inaxes:
            # or, maybe better to just post a pick event, and let the pointed to chan
            # (instead of clicked chan) stand up for itself
            #chan = self.get_closestchans(event, n=1)
            line = self.get_closestline(event) # get closest hline only
            if line:
                xpos, ypos = self.pos[line.chan]
                try:
                    tres = self.spykeframe.hpstream.tres
                except AttributeError: # no srf file loaded in spyke
                    tres = 1
                t = event.xdata - xpos # make it relative to the vertical tref line only, don't try to get absolute times
                v = (event.ydata - ypos) / self.gain
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














'''
class SpykeLine(Line2D):
    """Line2Ds that can be compared to each other for equality
    TODO: is subclassing Line2D really necessary?"""
    def __init__(self, *args, **kwargs):
        Line2D.__init__(self, *args, **kwargs)
        self.colour = 'none'

    def __hash__(self):
        """Hash the string representation of the y data"""
        return hash(self.colour + str(self._y))

    def __eq__(self, other):
        return hash(self) == hash(other)


class SortPanel(SpikePanel):
    """Sort panel. Presents a narrow temporal window of all channels
    layed out according to self.siteloc. Also allows overplotting and some
    user interaction"""
    def __init__(self, *args, **kwargs):
        SpikePanel.__init__(self, *args, **kwargs)
        self.colours = dict(zip(range(self.nchans), [YELLOW]*self.nchans))
        self.spikes = {}  # (spike, colour) -> [[SpykeLine], plotted]
        #self.x_vals = None
        self._initialized = False
        self.layers = {'g' : 0.5,
                       'y' :   1,
                       'r' : 0.7 }

        self.all_chans = self.nchans * [True]

    def _notVisible(self, spike, colour):
        lines, curr_visible = self.spikes[(spike, colour)]
        for line in lines:
            line._visible = False
        self.spikes[(spike, colour)][1] = False

    def _Visible(self, spike, colour, channels):
        lines, curr_visible = self.spikes[(spike, colour)]
        for line, chan in zip(lines, channels):
            line.chan_mask = chan
            line._visible = line.chan_mask
            line.zorder = self.layers[colour]
        self.spikes[(spike, colour)][1] = True

    def add(self, spike, colour, top=False, channels=None):
        """(Over)plot a given spike"""
        # initialize
        if not self._initialized:
            self.init_lines(spike)
            lines = []
            for num, channel in self.channels.iteritems():
                #xpos, ypos = self.pos[channel]
                channel._visible = channels[num]
                channel.chan_mask = channels[num]
                lines.append(channel)

            self.spikes[(spike, colour)] = [lines, True]
            self._initialized = True

        elif (spike, colour) in self.spikes:
            self.ax._visible = False
            self._Visible(spike, colour, channels)
            self.ax._visible = True

        elif (spike, colour) not in self.spikes:
            if channels is None:
                channels = self.all_chans
            self.ax._visible = False
            lines = []
            for chan in self.channels:
                xpos, ypos = self.pos[chan]
                line = SpykeLine(self.static_x_vals + xpos, # static_x_vals don't exist no more
                                 spike.data[chan] + ypos,
                                 linewidth=SPIKELINEWIDTH,
                                 color=self.colours[chan],
                                 antialiased=False)
                line.set_picker(PICKTHRESH)
                line._visible = False
                line.colour = colour
                line._visible = channels[chan]
                lines.append(line)
                line.zorder = self.layers[colour]
                self.ax.add_line(line)
            self.spikes[(spike, colour)] = [lines, True]
            self.ax._visible = True
        self.draw(True)

    def remove(self, spike, colour):
        """Remove the selected spike from the plot display"""
        #self._toggleVisible(spike, colour)
        self.ax._visible = False
        self._notVisible(spike, colour)
        #if colour in ('y'):
        #    lines, _ = self.spikes.pop((spike, colour))
        #    for line in lines:
        #        self.ax.lines.remove(line)
        self.ax._visible = True
        self.draw(True)


class ClickableSortPanel(SortPanel):
    def __init__(self, *args, **kwargs):
        SortPanel.__init__(self, *args, **kwargs)
        self.Bind(wx.EVT_LEFT_DCLICK, self.OnDoubleClick, self)
        self.mpl_connect('button_press_event', self.OnLeftDown)
        self.created = False

    def _createMaps(self):
        xpos = sorted(list(set([x for x, y in self.pos.itervalues()])))
        ypos = sorted(list(set([y for x, y in self.pos.itervalues()])))

        self.numcols = len(xpos)
        self.numrows = len(ypos)

        dist = lambda tup: (tup[1] - tup[0])
        self.x_width = dist(self.my_xlim) // self.numcols # my_xlim attrib has been deleted
        self.y_height = dist(self.my_ylim) // self.numrows # my_ylim attrib has been deleted

        self.intvalToChan = {}
        for chan, (xpos, ypos) in self.pos.iteritems():
            x_intval = xpos // self.x_width
            y_intval = ypos // self.y_height
            self.intvalToChan[(x_intval, y_intval)] = chan

    def _sendEvent(self, channels):
        event = ClickedChannelEvent(myEVT_CLICKED_CHANNEL, self.GetId())
        event.selected_channels = channels
        self.GetEventHandler().ProcessEvent(event)

    def OnDoubleClick(self, evt):
        channels = [True] * len(self.channels)
        self._sendEvent(channels)

    def pointToChannel(self, x, y):
        """Given a coordinate in the axes, find out what channel
        we're clicking on"""
        if not self.created:
            self._createMaps()
            self.created = True
        key = (int(x) // self.x_width, int(y) // self.y_height)
        if key in self.intvalToChan:
            return self.intvalToChan[key]
        return None

    def OnLeftDown(self, event):
        # event.inaxes
        channel = self.pointToChannel(event.xdata, event.ydata)
        if channel is not None:
            channels = [False] * len(self.channels)
            channels[channel] = True
            self._sendEvent(channels)
'''
