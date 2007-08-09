from __future__ import division

"""
eventwin.py

A smattering of code demonstrating an event window for spyke
"""

__author__ = 'Reza Lotun'


import wx
import wxmpl
import matplotlib.numerix as nx
import matplotlib.patches

class Demo(wx.App):
    def OnInit(self):
        self.frame = panel = EventWin(None, -1, 'Events')
        self.frame.Show(True)
        return True


class EventWin(wx.Frame):
    def __init__(self, parent, id, title, **kwds):
        wx.Frame.__init__(self, parent, id, title, **kwds)

        self.data = None
        self.points = []
        self.selectionPoints = []
        self.plotPanel = wxmpl.PlotPanel(self, -1)
        self.borderAxes = None

        wxmpl.EVT_SELECTION(self, self.plotPanel.GetId(), self._on_selection)
        wxmpl.EVT_POINT(self, self.plotPanel.GetId(), self._on_point)

        #self._layout()
        self._replot()

    def _layout(self):
        btnSizer = wx.BoxSizer(wx.HORIZONTAL)
        btnSizer.Add((1, 1), 1, 0, 0)
        btnSizer.Add(self.regionButton, 0, wx.BOTTOM|wx.RIGHT, 5)
        btnSizer.Add(self.pointButton,  0, wx.BOTTOM|wx.RIGHT, 5)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.plotPanel, 0, wx.ALL, 5)
        sizer.Add(btnSizer, 0, wx.EXPAND, 0)

        self.SetSizer(sizer)
        self.Fit()

    def _genRand(self):
        dt = 0.1
        self.t = nx.arange(0.0, 10.0, dt)
        r = nx.exp(-self.t[:100]/0.05)               # impulse response
        if self.data is None:
            self.data = nx.randn(len(self.t))
        x = self.data
        s = nx.convolve(x,r,mode=2)[:len(x)]*dt  # colored noise
        return s

    def _replot(self):

        _gr = {}
        _ax = 0
        def _drawGraph(sp, axisbg, frameon):
            global _ax
            s = self._genRand()
            a = fig.add_axes(sp,
                             axisbg,
                             frameon=False)
            #_gr[_ax] = a
            #_ax += 1
            a.plot(self.t, s, 'g', antialiased=False)
            a.grid(True)
            a.set_xticks([])
            a.set_yticks([])

        ############
        #
        num = 54
        spacing = [0.3, 0.05, 0.01, 0.01]
        offset = 0.02
        overlap = 0.02
        #
        #############


        fig = self.plotPanel.get_figure()
        #fig.clear()
        horizMargin, vertMargin, hSep, vSep = spacing
        width = (1 - 2 * horizMargin - hSep) / 2
        n = num / 2
        height = (1 - 2 * vertMargin - (n - 1) * vSep) / n
        bot = vertMargin
        for i in range(num // 2):
            _drawGraph(sp=[horizMargin, bot - offset, width, height],
                     axisbg='y',
                     frameon=False)

            _drawGraph(sp=[horizMargin + width + hSep - overlap,
                                        bot, width, height],
                      axisbg='y', frameon=False)
            bot += height + vSep

        if self.borderAxes:
            a = fig.sca(self.borderAxes)
            a.set_frame_on(True)

        # redraw the disply
        self.plotPanel.draw()

    def _on_regionButton(self, evt):
        if self.regionButton.GetValue():
            self.plotPanel.set_zoom(False)
        else:
            self.plotPanel.set_zoom(True)

    def _on_selection(self, evt):
        self.plotPanel.set_zoom(True)
        #self.regionButton.SetValue(False)

        x1, y1 = evt.x1data, evt.y1data
        x2, y2 = evt.x2data, evt.y2data

        self.selectionPoints.append(((x1, y1), x2-x1, y2-y1))
        self._replot()

    def _on_point(self, evt):
        self.borderAxes = evt.axes
        self._replot()

app = Demo()
app.MainLoop()
