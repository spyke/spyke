"""wx.Panels with embedded mpl figures based on FigureCanvasWxAgg.
Everything is plotted in units of uV and us

NOTE: it now seems best to center on the desired timepoint,
instead of left justify. The center (midway between first and last phase)
of a spike is the interesting part, and should therefore be the timestamp.
To see equally well on either side of that peak, let's center all waveform
views on the timestamp, instead of left justifying as before. This means
we should treat probe siteloc coordinates as the centers of the channels,
instead of the leftmost point of each channel.

TODO: perhaps refactor, keep info about each channel together,
make a Channel object with .id, .pos, .colour, .line, .enabled properties,
and set_enable() method, and then stick them in a dict of chans indexed by id"""

from __future__ import division

__authors__ = 'Reza Lotun, Martin Spacek'

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

DEFAULTCHANCOLOUR = "#00FF00" # garish green
TREFCOLOUR = "#303030" # dark grey
VREFCOLOUR = "#303030" # dark grey
CARETCOLOUR = "#202020" # light black
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
GREY = '#7F7F7F'
WHITE = '#FFFFFF'
BROWN = '#AF5050'

COLOURS = [RED, ORANGE, YELLOW, GREEN, CYAN, LIGHTBLUE, VIOLET, MAGENTA, GREY, WHITE, BROWN]

NCLOSESTCHANSTOSEARCH = 10
PICKRADIUS = 15 # required for 'line.contains(event)' call
#PICKTHRESH = 2.0 # in pixels? has to be a float or it won't work?


class SpykeLine(Line2D):
    """Line2Ds that can be compared to each other for equality"""
    def __init__(self, *args, **kwargs):
        Line2D.__init__(self, *args, **kwargs)
        self.colour = 'none'
        self.chan_mask = True # is this channel-as-line displayed?

    def __hash__(self):
        """Hash the string representation of the y data"""
        return hash(self.colour + str(self._y))

    def __eq__(self, other):
        return hash(self) == hash(other)


class PlotPanel(FigureCanvasWxAgg):
    """A wx.Panel with an embedded mpl figure.
    Base class for specific types of plot panels"""

    # not necessarily constants
    uVperum = DEFUVPERUM
    usperum = DEFUSPERUM # decreasing this increases horizontal overlap between spike chans
                            # 17 gives roughly no horizontal overlap for self.tw == 1000 us

    def __init__(self, parent, id=-1, stream=None, tw=None, cw=None):
        FigureCanvasWxAgg.__init__(self, parent, id, Figure())
        self._ready = False
        self.stream = stream
        self.SiteLoc = stream.probe.SiteLoc # probe site locations with origin at center top
        self.tw = tw # temporal width of each channel, in plot units (us ostensibly)
        self.cw = cw # time width of caret, in plot units

        self.pos = {} # positions of line centers, in plot units (us, uV)
        self.chans = stream.probe.SiteLoc.keys()
        self.chans.sort() # a sorted list of chans, keeps us from having to do this over and over
        self.nchans = stream.probe.nchans
        self.figure.set_facecolor(BACKGROUNDCOLOUR)
        self.figure.set_edgecolor(BACKGROUNDCOLOUR) # should really just turn off the edge line altogether, but how?
        #self.figure.set_frameon(False) # not too sure what this does, causes painting problems
        self.SetBackgroundColour(WXBACKGROUNDCOLOUR)
        self.colours = dict(zip(range(self.nchans), [DEFAULTCHANCOLOUR]*self.nchans))

        # for plotting with mpl, convert probe SiteLoc to have center bottom origin instead of center top
        siteloc = copy(self.SiteLoc) # lowercase means bottom origin
        ys = [y for x, y in siteloc.values()]
        maxy = max(ys)
        for key, (x, y) in siteloc.items():
            y = maxy - y
            siteloc[key] = (x, y) # update
        self.siteloc = siteloc # bottom origin

        tooltip = wx.ToolTip('\n') # create a tooltip, stick a newline in there so subsequent ones are recognized
        tooltip.Enable(False) # leave disabled for now
        tooltip.SetDelay(0) # set popup delay in ms
        self.SetToolTip(tooltip) # connect it to self

        self.mpl_connect('button_press_event', self.OnButtonPress) # bind mouse click within figure
        #self.mpl_connect('key_press_event', self.OnKeyPress)
        # TODO: mpl is doing something weird that always catches arrow key presses
        #self.mpl_connect('pick_event', self.OnPick) # happens when an artist with a .picker attrib has a mouse event happen within epsilon distance of it
        self.mpl_connect('motion_notify_event', self.OnMotion) # mouse motion within figure
        #self.mpl_connect('scroll_event', self.OnMouseWheel) # doesn't seem to be implemented yet in mpl's wx backend
        self.Bind(wx.EVT_MOUSEWHEEL, self.OnMouseWheel) # use wx event directly, although this requires window focus
    '''
    def OnKeyPress(self, event):
        """Let main spyke frame handle keypress events"""
        #self.GrandParent.OnKeyDown(event.guiEvent)
        event.guiEvent.Skip()
    '''
    def init_plot(self, wave, tref):
        """Create the axes and its lines"""
        self.wave = wave
        self.tref = tref
        pos = [0, 0, 1, 1]
        self.ax = self.figure.add_axes(pos,
                                       axisbg=BACKGROUNDCOLOUR,
                                       frameon=False,
                                       alpha=1.)
        self.do_layout() # defined by subclasses, sets self.pos

        self.xy_um = self.get_xy_um()
        x = self.xy_um[0]
        self.colxs = np.asarray(list(set(x))) # unique x values that demarcate columns
        self.colxs.sort() # guarantee they're in order from left to right

        self.ax._visible = False
        self.ax.set_axis_off() # turn off the x and y axis

        for ref in ['caret', 'vref', 'tref']: # add reference lines and caret in layered order
            self.add_ref(ref)

        self.lines = {} # line to chan mapping
        self.ax._autoscaleon = False # TODO: not sure if this is necessary
        for chan, (xpos, ypos) in self.pos.iteritems():
            line = SpykeLine(wave.ts - tref + xpos,
                             wave[chan]*self.gain + ypos,
                             linewidth=SPIKELINEWIDTH,
                             color=self.colours[chan],
                             antialiased=True)
            line.chan = chan
            line.set_pickradius(PICKRADIUS)
            #line.set_picker(PICKTHRESH)
            self.lines[chan] = line
            self.ax.add_line(line)

        self.ax._visible = True
        self._ready = True
        # redraw the display
        self.draw(True)

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
            vline = SpykeLine([col, col],
                              ylims,
                              linewidth=TREFLINEWIDTH,
                              color=TREFCOLOUR,
                              antialiased=True,
                              visible=False)
            self.vlines.append(vline)
            self.ax.add_line(vline)

    def _update_tref(self):
        """Update positions of vertical time reference line(s)"""
        cols = list(set([ xpos for chan, (xpos, ypos) in self.pos.iteritems() ]))
        ylims = self.ax.get_ylim()
        for col, vline in zip(cols, self.vlines):
            vline.set_data([col, col], ylims)

    def _add_vref(self):
        """Add horizontal voltage reference lines"""
        self.hlines = []
        for (xpos, ypos) in self.pos.itervalues():
            hline = SpykeLine([xpos-self.tw/2, xpos+self.tw/2],
                              [ypos, ypos],
                              linewidth=VREFLINEWIDTH,
                              color=VREFCOLOUR,
                              antialiased=True,
                              visible=False)
            self.hlines.append(hline)
            self.ax.add_line(hline)

    def _update_vref(self):
        """Update positions of horizontal voltage reference lines"""
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
                               linewidth=0,
                               antialiased=False,
                               visible=False)
        self.ax.add_patch(self.caret)

    def _update_caret_width(self):
        """Update caret"""
        #ylim = self.ax.get_ylim()
        # bottom left coord of rectangle
        self.caret.set_x(-self.cw/2)
        #self.caret.set_y(ylim[0])
        self.caret.set_width(self.cw)
        #self.caret.set_height(ylim[1] - ylim[0])

    def show_ref(self, ref, enable=True):
        if ref == 'tref':
            self._show_tref(enable)
        elif ref == 'vref':
            self._show_vref(enable)
        elif ref == 'caret':
            self._show_caret(enable)
        self.draw(True)
        #self.Refresh() # possibly faster, but adds flicker

    def _show_tref(self, enable):
        for vline in self.vlines:
            vline.set_visible(enable)

    def _show_vref(self, enable):
        for hline in self.hlines:
            hline.set_visible(enable)

    def _show_caret(self, enable):
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
        in self.chans order. Not the same as siteloc for chart and lfp frames,
        which have only a single column"""
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
            line = self.lines[chan]
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
        self.wave = wave
        self.tref = tref
        if tref == None:
            tref = wave.ts[0] # use the first timestamp in the waveform as the reference time point
        # check if we've set up our axes yet
        if not self._ready: # TODO: does this really need to be checked on every single plot call?
            self.init_plot(wave, tref)
        # update plots with new x and y vals
        for chan, line in self.lines.iteritems():
            xpos, ypos = self.pos[chan]
            xdata = wave.ts - tref + xpos
            ydata = wave[chan]*self.gain + ypos
            line.set_data(xdata, ydata) # update the line's x and y data
            #line._visible = True # is this necessary? Never seem to set it false outside of SortPanel
        self.ax._visible = True
        self.draw(True)
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

    def OnButtonPress(self, event):
        """Seek to timepoint as represented on chan closest to mouse click"""
        chan = self.get_closestchans(event, n=1)
        xpos = self.pos[chan][0]
        t = event.xdata - xpos + self.tref # undo position correction and convert from relative to absolute time
        # call main spyke frame's seek method
        self.GrandParent.seek(t)
    '''
    def OnPick(self, event):
        """Pop up a tooltip when mouse is within PICKTHRESH of a line"""
        tooltip = self.GetToolTip()
        if event.mouseevent.inaxes:
            line = event.artist # assume it's one of our SpykeLines, since those are the only ones with their .picker attrib enabled
            chan = line.chan
            xpos, ypos = self.pos[chan]
            t = event.mouseevent.xdata - xpos + self.tref # undo position correction and convert from relative to absolute time
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
            if line:
                xpos, ypos = self.pos[line.chan]
                t = event.xdata - xpos + self.tref
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
        self.GrandParent.spiketw = self.tw
        self.GrandParent.frames['chart'].panel.cw = self.tw
        self.GrandParent.frames['chart'].panel._update_caret_width()
        self.GrandParent.plot(frametypes='spike') # replot

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
        self.GrandParent.charttw = self.tw
        self.GrandParent.frames['lfp'].panel.cw = self.tw
        self.GrandParent.frames['lfp'].panel._update_caret_width()
        self.GrandParent.plot(frametypes='chart') # replot

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
        self.GrandParent.lfptw = self.tw
        self.GrandParent.plot(frametypes='lfp') # replot

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
            self.init_plot(spike)
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
