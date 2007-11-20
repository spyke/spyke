""" spyke.gui.events

Custom wx events used for communication between spyke gui components
"""
import wx

class PlotEvent(wx.PyCommandEvent):
    def __init__(self, evtType, id):
        wx.PyCommandEvent.__init__(self, evtType, id)
        self.plot = None
        self.remove = None
        self.colour = None

myEVT_PLOT = wx.NewEventType()
EVT_PLOT = wx.PyEventBinder(myEVT_PLOT, 1)

