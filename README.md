[spyke](http://spyke.github.io) is a Python application for visualizing, navigating,
and spike sorting neuronal extracellular waveform data.

spyke currently works with [Swindale Lab](http://swindale.ecc.ubc.ca) `.srf` files,
but may be extended to work with other electrophysiology file formats. Some
[sample data](http://swindale.ecc.ubc.ca/spyke) is available.

Some functionality was inherited from Tim Blanche's Delphi program "SurfBawd".
Some icons were copied from Ubuntu's [Humanity](http://launchpad.net/humanity)
icon theme.

Dependencies:

spyke requires recent versions of the following to be installed:

* [Python](http://python.org) (2.7.x, 3.x hasn't been tested)
* [IPython](http://ipython.org) 1.0.dev
* [numpy](http://numpy.org)
* [scipy](http://scipy.org)
* [matplotlib](http://matplotlib.org)
* [PyQt4](http://www.riverbankcomputing.co.uk/software/pyqt)
  ([PySide](http://pyside.org) will probably work too, but is untested)
* [PyOpenGL](http://pyopengl.sourceforge.net/)
* [Cython](http://cython.org)

spyke is developed in Xubuntu 12.10. It should work in other Linux distributions.
In principle, it should also work in Windows and OSX.

An older version is described in the paper [Python for large-scale electrophysiology]
(http://www.frontiersin.org/Neuroinformatics/10.3389/neuro.11.009.2008/abstract).

To run:
```
$ python main.py # in the spyke folder
```
To install for use as a library:
```
$ python setup.py install
```
Many keyboard shortcuts are available. Tooltips give some hints. You can also
discover them by searching for `keyPressEvent` methods in the code.
