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

def calc_spacings(layout):
    """ Map from polytrode locations given as (x, y) coordinates
    into position information for the spike plots, which are stored
    as a list of four values [l, b, w, h]. To illustrate this, consider
    loc_i = (x, y) are the coordinates for the polytrode on channel i.
    We want to map these coordinates to the unit square.
       (0,0)                          (0,1)
          +------------------------------+
          |        +--(w)--+
          |<-(l)-> |       |
          |        | loc_i (h)
          |        |       |
          |        +-------+
          |            ^ 
          |            | (b) 
          |            v 
          +------------------------------+
         (1,0)                          (1,1)
    """

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


