"""wx.Panels with embedded mpl figures based on FigureCanvasWxAgg.
Everything is plotted in units of uV and us

NOTE: it now seems best to center on the desired timepoint,
instead of left justify. The center (peak) of a spike is the interesting part,
and should therefore be the timestamp. To see equally well on either side of
that peak, let's center all waveform views on the timestamp, instead of left
justifying as before. This means we should treat probe layout coordinates as the
centers of the channels, instead of the leftmost point of each channel.

TODO: perhaps refactor, keep info about each channel together,
make a Channel object with .id, .pos, .colour, .line properties,
and then stick them in a dict of chans indexed by id"""

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
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from spyke import surf
from spyke.gui.events import *

DEFAULTLINEWIDTH = 1 # mpl units - pixels? points? plot units (us)?
CHANVSPACE = 75 # uV, vertical space between top and bottom chans and axes edge

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

MICROVOLTSPERMICRON = 2
MICROSECSPERMICRON = 17 # decreasing this increases horizontal overlap between spike chans

PICKTHRESH = 2.0 # in pixels? has to be a float or it won't work?

def um2uv(um):
    """Vertical conversion from um in channel layout
    space to uV in signal space"""
    return MICROVOLTSPERMICRON * um

def uv2um(uV):
    """Convert from uV to um"""
    return uV / MICROVOLTSPERMICRON

def um2us(um):
    """Horizontal conversion from um in channel layout
    space to us in signal space"""
    return MICROSECSPERMICRON * um

def us2um(us):
    """Convert from us to um"""
    return us / MICROSECSPERMICRON


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
    def __init__(self, parent, id=-1, layout=None, tw=None, cw=None):
        FigureCanvasWxAgg.__init__(self, parent, id, Figure())
        self._ready = False
        self.layout = layout # probe layout with origin at center top
        self.tw = tw # temporal width of each channel, in plot units (us ostensibly)
        if cw == None: # like for a SpikePanel
            cw = tw
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

        # for plotting with mpl, convert probe layout to have center bottom origin instead of center top
        bottomlayout = copy(self.layout)
        ys = [y for x, y in bottomlayout.values()]
        maxy = max(ys)
        for key, (x, y) in bottomlayout.items():
            y = maxy - y
            bottomlayout[key] = (x, y) # update
        self.bottomlayout = bottomlayout

        self.mpl_connect('button_press_event', self.OnButtonPress) # bind mouse click within figure
        #self.mpl_connect('pick_event', self.OnPick) # happens when an artist with a .picker attrib has a mouse event happen within epsilon distance of it

    def init_plot(self, wave, tref):
        """Create the axes and its lines"""
        self.wave = wave
        self.tref = tref
        pos = [0, 0, 1, 1]
        self.ax = self.figure.add_axes(pos,
                                       axisbg=BACKGROUNDCOLOUR,
                                       frameon=False,
                                       alpha=1.)
        self.do_layout() # defined by subclasses

        self.ax._visible = False
        self.ax.set_axis_off() # turn off the x and y axis

        self.lines = {}
        self.ax._autoscaleon = False
        for chan, (xpos, ypos) in self.pos.iteritems():
            line = SpykeLine(wave.ts - tref + xpos,
                             wave[chan] + ypos,
                             linewidth=self.linewidth,
                             color=self.colours[chan],
                             antialiased=True)
            line.chan = chan
            #line.set_picker(PICKTHRESH)
            self.lines[chan] = line
            self.ax.add_line(line)

            self.channels[chan] = line

        self.ax._visible = True
        self._ready = True
        # redraw the display
        self.draw(True)

    def get_spatialchans(self, order='vertical'):
        """Return channels in spatial order.
        order='vertical': sort from bottom to top, left to right
        order='horziontal': sort from left to right, bottom to top
        TODO: fix code duplication"""
        if order == 'vertical':
            # first, sort x coords, then y: (secondary, then primary)
            xychans = [ (x, y, chan) for chan, (x, y) in self.bottomlayout.items() ] # list of (x, y, chan) 3-tuples
            xychans.sort() # stable sort in-place according to x values (first in tuple)
            yxchans = [ (y, x, chan) for (x, y, chan) in xychans ]
            yxchans.sort() # stable sort in-place according to y values (first in tuple)
            chans = [ chan for (y, x, chan) in yxchans ] # unload the chan indices, now sorted bottom to top, left to right
        elif order == 'horizontal':
            # first, sort y coords, then x: (secondary, then primary)
            yxchans = [ (y, x, chan) for chan, (x, y) in self.bottomlayout.items() ] # list of (y, x, chan) 3-tuples
            yxchans.sort() # stable sort in-place according to y values (first in tuple)
            xychans = [ (x, y, chan) for (y, x, chan) in yxchans ] # list of (x, y, chan) 3-tuples
            xychans.sort() # stable sort in-place according to x values (first in tuple)
            chans = [ chan for (x, y, chan) in xychans ] # unload the chan indices, now sorted left to right, bottom to top
        else:
            raise ValueError
        return chans

    def get_closestchan(self, xdata, ydata):
        """Find channel that's closest to the (xdata, ydata) point.
        Convert x and y values to um, take sum(sqrd) distance between (xdata, ydata)
        and all site positions, find the argmin, and that's your nearest channel"""
        xdata_um = us2um(xdata)
        ydata_um = uv2um(ydata)
        xychans_um = [ (us2um(x), uv2um(y), chan) for chan, (x, y) in self.pos.items() ]
        # sum of squared distances, no need to bother sqare-rooting them
        d2 = np.asarray([ (x_um-xdata_um)**2 + (y_um-ydata_um)**2 for (x_um, y_um, chan) in xychans_um ])
        i = d2.argmin() # find index of smallest squared distance
        chan = xychans_um[i][2] # pull out the channel
        return chan

    def plot(self, wave, tref=None):
        """Plot waveforms wrt a reference time point"""
        self.wave = wave
        self.tref = tref
        if tref == None:
            tref = wave.ts[0] # use the first timestamp in the waveform as the reference time point
        # check if we've set up our axes yet
        if not self._ready: # TODO: does this really need to be checked on every single plot call?
            self.init_plot(wave, tref)
        # check if xvals have changed, this will normally only happen near extrema of .srf file
        chan, line = self.lines.iteritems().next() # random line
        xpos = self.pos[chan][0]
        xvals = line.get_xdata() - xpos # remove x position offset, so now we're in normal us units
        updatexvals = (xvals[0],  xvals[-1]) != (wave.ts[0]-tref, wave.ts[-1]-tref) # do xval endpoints differ?
        # update plots with new yvals, and possibly new xvals as well
        for chan, line in self.lines.iteritems():
            xpos, ypos = self.pos[chan]
            if updatexvals:
                # update the line's x values (or really, the number of x values, their position shouldn't change in space)
                line.set_xdata(wave.ts - tref + xpos)
            line.set_ydata(wave[chan] + ypos) # update the line's y values
            line._visible = True # is this necessary? Never seem to set it false outside of SortPanel
        self.ax._visible = True
        self.draw(True)

    def OnButtonPress(self, event):
        """Seek to timepoint as represented on chan closest to mouse click"""
        chan = self.get_closestchan(event.xdata, event.ydata)
        xpos = self.pos[chan][0]
        t = event.xdata - xpos + self.tref
        # call parent frame's seek method, which then calls main frame's seek method
        self.Parent.seek(t)
    '''
    def OnPick(self, event):
        """Good for figuring out which line has just been clicked on,
        within tolerance of PICKTHRESH in pixels"""
        line = event.artist # assume it's one of our SpykeLines, since those are the only ones with their .picker attrib enabled
        chan = line.chan
        xpos = self.pos[chan][0]
        t = event.mouseevent.xdata - xpos + self.tref # undo position correction and convert from relative to absolute time
        print chan, event.mouseevent.xdata, t
        self.Parent.seek(t)
    '''

class ChartPanel(PlotPanel):
    """Chart panel. Presents all channels layed out vertically according
    to the vertical order of their site coords in .layout

    TODO: layout vertical positions manually. Can't rely on site layout coords,
    since some of their y vals can be identical (like in a 3col polytrode), which
    then causes channels to be overplotted at the same y position"""

    def do_layout(self):
        chans = self.get_spatialchans('vertical') # ordered bottom to top, left to right
        #print 'vertical ordered chans in Chartpanel:\n%r' % chans
        self.ax.set_xlim(0 - self.tw/2, 0 + self.tw/2) # x origin at center
        self.ax.set_ylim(um2uv(self.bottomlayout[chans[0]][1]) - CHANVSPACE,
                         um2uv(self.bottomlayout[chans[-1]][1]) + CHANVSPACE)
        self.add_cursor()
        colourgen = itertools.cycle(iter(COLOURS))
        for chan in chans:
            self.pos[chan] = (0, um2uv(self.bottomlayout[chan][1])) # x=0 centers horizontally
            self.colours[chan] = colourgen.next() # assign colours so that they cycle nicely in space

    def add_cursor(self):
        """Add a shaded rectangle to represent the
        time window shown in the spike frame"""
        ylim = self.ax.get_ylim()
        xy = (-self.cw/2, ylim[0]) # bottom left coord of rectangle
        width = self.cw
        height = ylim[1] - ylim[0]
        self.cursor = Rectangle(xy, width, height,
                                facecolor=CURSORCOLOUR, linewidth=0, antialiased=False)
        self.ax.add_patch(self.cursor)


class SpikePanel(PlotPanel):
    """Spike panel. Presents a narrow temporal window of all channels
    layed out according to self.layout"""

    def do_layout(self):
        hchans = self.get_spatialchans('horizontal') # ordered left to right, bottom to top
        vchans = self.get_spatialchans('vertical') # ordered bottom to top, left to right
        #print 'horizontal ordered chans in Spikepanel:\n%r' % hchans
        self.ax.set_xlim(um2us(self.bottomlayout[hchans[0]][0]) - self.tw/2,
                         um2us(self.bottomlayout[hchans[-1]][0]) + self.tw/2) # x origin at center
        self.ax.set_ylim(um2uv(self.bottomlayout[vchans[0]][1]) - CHANVSPACE,
                         um2uv(self.bottomlayout[vchans[-1]][1]) + CHANVSPACE)
        colourgen = itertools.cycle(iter(COLOURS))
        for chan in vchans:
            # chan order doesn't matter for setting .pos, but it does for setting .colours
            self.pos[chan] = (um2us(self.bottomlayout[chan][0]),
                              um2uv(self.bottomlayout[chan][1]))
            self.colours[chan] = colourgen.next() # assign colours so that they cycle nicely in space


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
                                 linewidth=self.linewidth,
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
