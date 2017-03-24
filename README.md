[spyke](http://spyke.github.io) is a Python application for visualizing, navigating, and spike
sorting high-density multichannel extracellular neuronal waveform data.

spyke currently works with `.dat` files, Blackrock `.nsx` files, [Swindale
Lab](http://swindale.ecc.ubc.ca) `.srf` and `.tsf` files, and Rodrigo Quian Quiroga's
[simulated data](http://www.vis.caltech.edu/~rodri/Wave_clus/Simulator.zip) .mat files.
spyke can be extended to work with other electrophysiology file formats. Some [sample
data](http://swindale.ecc.ubc.ca/spyke) is available. [Spike sorting for polytrodes: a divide
and conquer approach](http://dx.doi.org/10.3389/fnsys.2014.00006) is a paper describing the
overall approach. spyke is described in greater detail in Chapter 3 and Appendix C.2 of this
[thesis](http://mspacek.github.io/mspacek_thesis.pdf).

Some functionality was inherited from Tim Blanche's Delphi program "SurfBawd". Some icons were
copied from Ubuntu's [Humanity](http://launchpad.net/humanity) icon theme.

Dependencies:

spyke requires recent versions of the following to be installed:

* [Python](http://python.org) (2.7.x, 3.x hasn't been tested)
* [IPython](http://ipython.org)
* [numpy](http://numpy.org)
* [scipy](http://scipy.org)
* [matplotlib](http://matplotlib.org)
* [PyQt4](http://www.riverbankcomputing.co.uk/software/pyqt)
  ([PySide](http://pyside.org) will probably work too, but is untested)
* [PyOpenGL](http://pyopengl.sourceforge.net/)
* [Cython](http://cython.org)
* [MDP](http://mdp-toolkit.sourceforge.net/)
* [scikit-learn](http://scikit-learn.org)
* [PyWavelets](http://www.pybytes.com/pywavelets/)

spyke is developed in Xubuntu 16.04. It should work in other Linux distributions. In
principle, it should also work in Windows and OSX.

A much older version is described in the paper [Python for large-scale electrophysiology]
(http://www.frontiersin.org/Neuroinformatics/10.3389/neuro.11.009.2008/abstract).

To run:
```
$ python main.py # in the spyke folder
```
To install for use as a library (TODO: doesn't currently work):
```
$ python setup.py install
```

For the NVIDIA 340 driver in Linux, it seems the "Allow Flipping" option must be disabled in
the OpenGL settings in the `nvidia-settings` app, otherwise the 3D cluster plot will not
update properly.

Plotting of spike waveforms in the Sort window can be slow and generate flicker. This seems to
be a problem with matplotlib, and can be fixed by applying the following diff to matplotlib:

```
diff --git a/lib/matplotlib/backends/backend_qt5agg.py b/lib/matplotlib/backends/backend_qt5agg.py
index 5b8e111..55fdb4f 100644
@@ -165,3 +165,3 @@ class FigureCanvasQTAggBase(object):
         t = b + h
-        self.repaint(l, self.renderer.height-t, w, h)
+        self.update(l, self.renderer.height-t, w, h)

```

See [TUTORIAL.md](TUTORIAL.md) for a fairly brief tutorial.

Many keyboard shortcuts are available. Tooltips give some hints. You can also discover them by
searching for `keyPressEvent` methods in the code.
