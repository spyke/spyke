from __future__ import division
"""
spyke.plot - Plotting elements
"""

__author__ = 'Reza Lotun'

import random
import wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
import matplotlib.numerix as nx

import spyke.surf
import spyke.stream

class PlotPanel(FigureCanvasWxAgg):
    """ A generic set of spyke plots. Meant to be a superclass of specific
    implementations of a plot panel (e.g. ChartPanel, EventPanel, etc.)
    """
    def __init__(self, frame):
        FigureCanvasWxAgg.__init__(self, frame, -1, Figure())
        self.setColours()
        self.channels{}

    def setColours(self):
        self.figure.set_facecolor('black')
        self.SetBackgroundColour(wx.BLACK)

    def init_plot(self):
        pass


