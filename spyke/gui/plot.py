"""wx.Panels with embedded mpl figures based on FigureCanvasWxAgg"""

from __future__ import division

__author__ = 'Reza Lotun'

import itertools
from copy import copy
import random
import numpy
import wx

from matplotlib import rcParams
rcParams['lines.linestyle'] = '-'
rcParams['lines.marker'] = ''

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from spyke import surf
from spyke.gui.events import *

DEFAULTLINEWIDTH = 1 # mpl units - pixels? points? plot units (us)?
CHANHEIGHT = 100 # uV

DEFAULTCHANCOLOUR = "#00FF00" # garish green
CURSORCOLOUR = "#171717" # light black
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


class AxesWrapper(object):
    """A wrapper around an axes that delegates access to attributes to an
    actual axes object. Really meant to make axes hashable"""
    def __init__(self, axes):
        self._spyke_axes = axes

    def __getattr__(self, name):
        """Delegate access of attribs to wrapped axes"""
        return getattr(self._spyke_axes, name)

    def __hash__(self):
        """Base the hash function on the position rect"""
        return hash(str(self.get_position(original=True)))

    def __eq__(self, other):
        return hash(self) == hash(other)


class SpykeLine(Line2D):
    """Line2D's that can be compared to each other for equality"""
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
    def __init__(self, parent, id=-1, layout=None, tw=None, cw=None):
        FigureCanvasWxAgg.__init__(self, parent, id, Figure())
        self._ready = False
        self.layout = layout # layout with y coord origin at top
        self.tw = tw # temporal width of each channel, in plot units (us ostensibly)
        self.cw = cw # time width of cursor, in plot units

        self.pos = {} # position of lines
        self.channels = {} # plot y-data for each channel
        self.nchans = len(layout)
        self.figure.set_facecolor(BACKGROUNDCOLOUR)
        self.figure.set_edgecolor(BACKGROUNDCOLOUR) # should really just turn off the edge line altogether, but how?
        #self.figure.set_frameon(False) # not too sure what this does, causes painting problems
        self.SetBackgroundColour(WXBACKGROUNDCOLOUR)
        self.colours = dict(zip(range(self.nchans), [DEFAULTCHANCOLOUR]*self.nchans))
        self.linewidth = DEFAULTLINEWIDTH

        # for plotting with mpl, convert all y coords to have origin at bottom, not top
        bottomlayout = copy(self.layout)
        ys = [y for x, y in bottomlayout.values()]
        maxy = max(ys)
        for key, (x, y) in bottomlayout.items():
            y = maxy - y
            bottomlayout[key] = (x, y) # update
        self.bottomlayout = bottomlayout

    def init_plot(self, wave, tref):
        """Create the axes and its lines"""
        pos = [0, 0, 1, 1]
        self.ax = self.figure.add_axes(pos,
                                       axisbg=BACKGROUNDCOLOUR,
                                       frameon=False,
                                       alpha=1.)
        self.do_layout() # defined by subclasses

        self.ax._visible = False
        self.ax.set_axis_off() # turn off the x and y axis

        self.displayed_lines = {}
        self.ax._autoscaleon = False
        for chan, spacing in self.pos.iteritems():
            x_off, y_off = spacing
            line = SpykeLine(wave.ts - tref + x_off,
                             wave.data[chan] + y_off,
                             linewidth=self.linewidth,
                             color=self.colours[chan],
                             antialiased=True)
            self.displayed_lines[chan] = line
            self.ax.add_line(line)

            self.channels[chan] = line

        self.ax._visible = True
        self._ready = True
        # redraw the display
        self.draw(True)

    def get_spatialchans(self):
        """Return channels in spatial order, from bottom to top, left to right"""
        # first, sort x coords, then y: (secondary, then primary)
        xychans = [ (x, y, chan) for chan, (x, y) in self.bottomlayout.items() ] # list of (x, y, chan) 3-tuples
        xychans.sort() # stable sort in-place according to x values (first in tuple)
        yxchans = [ (y, x, chan) for (x, y, chan) in xychans ]
        yxchans.sort() # stable sort in-place according to y values (first in tuple)
        chans = [ chan for (y, x, chan) in yxchans ] # unload the chan indices, now sorted bottom to top, left to right
        return chans

    def plot(self, wave, tref=None):
        """Plot waveforms wrt a reference time point"""
        if tref == None:
            tref = wave.ts[0] # use the first timestamp in the waveform as the reference time point
        # check if we've set up our axes yet
        if not self._ready: # TODO: does this really need to be checked on every single plot call?
            self.init_plot(wave, tref)
        # update plots with new data
        line = self.displayed_lines.values()[0] # random line, first in the list
        updatexvals = (line.get_xdata()[0], line.get_xdata()[-1]) != (wave.ts[0]-tref, wave.ts[-1]-tref) # do endpoints differ?
        for chan, line in self.displayed_lines.iteritems():
            x_off, y_off = self.pos[chan]
            if updatexvals:
                line.set_xdata(wave.ts - tref + x_off) # update the line's x values (or really, the number of x values, their position shouldn't change in space)
                # should I also subtract self.tw/2 to make it truly centered for chartwin?
            line.set_ydata(wave.data[chan] + y_off) # update the line's y values
            line._visible = True # is this necessary? Never seem to set it false outside of SortPanel
        self.ax._visible = True
        self.draw(True)


class ChartPanel(PlotPanel):
    """Chart panel. Presents all channels layed out vertically according
    to site y coords in .layout"""

    def do_layout(self):
        self.ax.set_xlim(-self.tw/2, self.tw/2)
        self.ax.set_ylim(-CHANHEIGHT, self.nchans*CHANHEIGHT)
        self.add_cursor()
        chans = self.get_spatialchans()
        colourgen = itertools.cycle(iter(COLOURS))
        for chani, chan in enumerate(chans):
            self.pos[chan] = (0, chani*CHANHEIGHT)
            self.colours[chan] = colourgen.next() # now assign colours so that they cycle nicely in space

    def add_cursor(self):
        # add a shaded region to represent region shown in spike frame
        ylim = self.ax.get_ylim()
        xy = (0, ylim[0])
        width = self.cw
        height = ylim[1] - ylim[0]
        self.cursor = Rectangle(xy, width, height,
                                facecolor=CURSORCOLOUR, linewidth=0, antialiased=False)
        self.ax.add_patch(self.cursor)


class SpikePanel(PlotPanel):
    """Spike panel. Presents a narrow temporal window of all channels
    layed out according to self.layout"""

    def do_layout(self):
        """Map from polytrode layout given as (x, y) coordinates in um,
        into position information for the spike plots, which are stored
        as a list of four values [l, b, w, h].

        loc_i = (x, y) are the coordinates for the polytrode on channel i.
        We want to map these coordinates to the unit square.

           (0,1)                          (1,1)
              +------------------------------+
              |        +--(w)--+
              |<-(l)-> |       |
              |        |  x,y (h)
              |        |       |
              |        +-------+
              |            ^
              |            | (b)
              |            v
              +------------------------------+
             (0,0)                          (0,1)

        NOTE that unlike indicated above, actual .layout coords are:
            x: distance from center of polytrode
            y: distance down from top of polytrode border (slightly above top site)
        So, y locations need to be swapped vertically before being used - use .bottomlayout"""

        # project coordinates onto x and y axes respectively
        xs = [x for x, y in self.bottomlayout.values()]
        ys = [y for x, y in self.bottomlayout.values()]
        # get limits on coordinates
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        # define the width and height of the bounding boxes
        colwidth = self.tw - 100 # amount of horizontal screen space per column, slight overlap
        uniquexs = list(set(xs))
        ncols = len(uniquexs) # number of unique x site coords
        self.ax.set_xlim(0, ncols*colwidth)
        x_offsets = {}
        for i, x in enumerate(sorted(uniquexs)):
            x_offsets[x] = i * colwidth
        uniqueys = list(set(ys))
        nrows = len(uniqueys) # TODO: a 2 col staggered probe has nothing but unique y coords, but that doesn't mean they should all be spaced CHANHEIGHT apart vertically, only between those in adjacent rows of the same column
        self.ax.set_ylim(-CHANHEIGHT, nrows*CHANHEIGHT) # this doesn't seem right, see above
        y_offsets = {}
        for i, y in enumerate(sorted(uniqueys)):
            y_offsets[y] = i * CHANHEIGHT

        colourgen = itertools.cycle(iter(COLOURS))
        for chan in self.get_spatialchans():
            x, y = self.bottomlayout[chan]
            x_off = x_offsets[x]
            y_off = y_offsets[y]
            self.pos[chan] = (x_off, y_off)
            self.colours[chan] = colourgen.next() # now assign colours so that they cycle nicely in space


class SortPanel(SpikePanel):
    """Sort panel. Presents a narrow temporal window of all channels
    layed out according to self.layout. Also allows overplotting and some
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
                #x_off, y_off = self.pos[channel]
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

            #need to deal with removal of self.static_x_vals from base PlotPanel class, replace with:
            #xvals = wave.ts - wave.ts[0]
            #but that would require a waveform object, and this method only gets a "spike", whatever that is...

            self.ax._visible = False
            lines = []
            for chan in self.channels:
                x_off, y_off = self.pos[chan]
                line = SpykeLine(self.static_x_vals + x_off,
                                 spike.data[chan] + y_off,
                                 linewidth=self.linewidth,
                                 color=self.colours[chan],
                                 antialiased=False)
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
        self.xoff = sorted(list(set([x for x, y in self.pos.itervalues()])))
        self.yoff = sorted(list(set([y for x, y in self.pos.itervalues()])))

        self.numcols = len(self.xoff)
        self.numrows = len(self.yoff)

        dist = lambda tup: (tup[1] - tup[0])
        self.x_width = dist(self.my_xlim) // self.numcols # my_xlim attrib has been deleted
        self.y_height = dist(self.my_ylim) // self.numrows # my_ylim attrib has been deleted

        self.intvalToChan = {}
        for chan, offsets in self.pos.iteritems():
            x_off, y_off = offsets
            x_intval = x_off // self.x_width
            y_intval = y_off // self.y_height
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




######## Tests #########


class Opener(object):
    """Opens and parses the first available file in filenames"""
    FILENAMES = ['/data/ptc15/87 - track 7c spontaneous craziness.srf',
                 '/Documents and Settings/Reza Lotun/Desktop/Surfdata/87 - track 7c spontaneous craziness.srf',
                 '/media/windows/Documents and Settings/Reza ' \
                            'Lotun/Desktop/Surfdata/' \
                            '87 - track 7c spontaneous craziness.srf',
                 '/home/rlotun/data_spyke/'\
                            '87 - track 7c spontaneous craziness.srf',
                 '/data/87 - track 7c spontaneous craziness.srf',
                 '/Users/rlotun/work/spyke/data/smallSurf',
                 '/home/rlotun/spyke/data/smallSurf',
                ]

    def __init__(self):

        import spyke.detect
        import os

        for filename in self.FILENAMES:
            try:
                stat = os.stat(filename)
                break
            except:
                continue

        surf_file = surf.File(filename)
        self.surf_file = surf_file
        surf_file.parse()
        self.dstream = spyke.core.Stream(surf_file.highpassrecords)
        layout_name = surf_file.layoutrecords[0].electrode_name
        layout_name = layout_name.replace('\xb5', 'u') # replace 'micro' symbol with 'u'
        import spyke.probes
        self.layout = eval('spyke.probes.' + layout_name)() # yucky, UNTESTED
        self.curr = self.dstream.records[0].TimeStamp


class TestWindows(wx.App):
    def __init__(self, *args, **kwargs):
        kwargs['redirect'] = False
        wx.App.__init__(self, *args, **kwargs)

    def OnInit(self):
        op = Opener()
        self.events = panel = TestEventWin(None, -1, 'Data', op, size=(200,900))
        self.sort = panel3 = TestSortWin(None, -1, 'Events', op, size=(200,900))
        self.chart = panel4 = TestChartWin(None, -1, 'Chart', op, size=(500,600))
        self.SetTopWindow(self.events)
        self.events.Show(True)
        self.sort.Show(True)
        self.chart.Show(True)
        return True


class PlayWin(wx.Frame):
    def __init__(self, parent, id, title, op, **kwds):
        wx.Frame.__init__(self, parent, id, title, **kwds)
        self.stream = op.dstream
        self.layout = op.layout
        self.incr = 1000        # 1 ms
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.onTimerEvent, self.timer)

    def onTimerEvent(self, evt):
        pass

    def onEraseBackground(self, evt):
        """Prevent redraw flicker"""
        pass


class TestSortWin(PlayWin):
    """Reference implementation of a test Sort Window. An SpikePanel
    is simply embedded in a wx Frame, and a passed in data file is played"""
    def __init__(self, parent, id, title, op, **kwds):
        PlayWin.__init__(self, parent, id, title, op, **kwds)
        self.incr = 1000
        simp = spyke.detect.SimpleThreshold(self.stream,
                                            self.stream.records[0].TimeStamp)
        self.event_iter = iter(simp)
        self.plotPanel = SortPanel(self, self.layout.SiteLoc) # fast
        self.borderAxes = None
        self.timer.Start(200)

    def onTimerEvent(self, evt):
        wave = self.event_iter.next()
        self.plotPanel.plot(wave)


class TestSpikeWin(PlayWin):
    def __init__(self, parent, id, title, op, **kwds):
        PlayWin.__init__(self, parent, id, title, op, **kwds)
        self.plotPanel = SpikePanel(self, self.layout.SiteLoc) # fast
        self.data = None
        self.points = []
        self.selectionPoints = []
        self.borderAxes = None
        self.curr = op.curr
        self.timer.Start(200)

    def onTimerEvent(self, evt):
        wave = self.stream[self.curr:self.curr+self.incr]
        self.curr += self.incr
        self.plotPanel.plot(wave)


class TestChartWin(PlayWin):
    def __init__(self, parent, id, title, op, **kwds):
        PlayWin.__init__(self, parent, id, title, op, **kwds)
        self.plotPanel = ChartPanel(self, self.layout.SiteLoc)
        self.data = None
        self.points = []
        self.selectionPoints = []
        self.borderAxes = None
        self.curr = op.curr
        self.incr = 5000
        self.timer.Start(100)

    def onTimerEvent(self, evt):
        wave = self.stream[self.curr:self.curr+self.incr]
        self.curr += self.incr
        self.plotPanel.plot(wave)


if __name__ == '__main__':
    app = TestWindows()
    app.MainLoop()
