[spyke](http://spyke.github.io) is a Python application for visualizing, navigating, and spike
sorting high-density multichannel extracellular neuronal waveform data.

spyke currently works with [Swindale Lab](http://swindale.ecc.ubc.ca) `.srf` and `.tsf` files,
and Rodrigo Quian Quiroga's [simulated
data](http://www.vis.caltech.edu/~rodri/Wave_clus/Simulator.zip) MATLAB files. spyke can be
extended to work with other electrophysiology file formats. Some [sample
data](http://swindale.ecc.ubc.ca/spyke) is available. [Spike sorting for polytrodes: a divide
and conquer approach](http://dx.doi.org/10.3389/fnsys.2014.00006) is a paper describing the
overall approach.

Some functionality was inherited from Tim Blanche's Delphi program "SurfBawd". Some icons were
copied from Ubuntu's [Humanity](http://launchpad.net/humanity) icon theme.

Dependencies:

spyke requires recent versions of the following to be installed:

* [Python](http://python.org) (2.7.x, 3.x hasn't been tested)
* [IPython](http://ipython.org) 1.0
* [numpy](http://numpy.org)
* [scipy](http://scipy.org)
* [matplotlib](http://matplotlib.org)
* [PyQt4](http://www.riverbankcomputing.co.uk/software/pyqt)
  ([PySide](http://pyside.org) will probably work too, but is untested)
* [PyOpenGL](http://pyopengl.sourceforge.net/)
* [Cython](http://cython.org)

spyke is developed in Xubuntu 14.04. It should work in other Linux distributions. In
principle, it should also work in Windows and OSX.

A much older version is described in the paper [Python for large-scale electrophysiology]
(http://www.frontiersin.org/Neuroinformatics/10.3389/neuro.11.009.2008/abstract).

To run:
```
$ python main.py # in the spyke folder
```
To install for use as a library:
```
$ python setup.py install
```
See [TUTORIAL.md](TUTORIAL.md) for a fairly brief tutorial.

Many keyboard shortcuts are available. Tooltips give some hints. You can also discover them by
searching for `keyPressEvent` methods in the code.
