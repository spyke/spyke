### spyke

[spyke](http://spyke.github.io) is a Python application for visualizing, navigating, and spike
sorting high-density multichannel extracellular neuronal waveform data.

spyke currently works with `.dat` files, Blackrock `.nsx` files, [Swindale
Lab](http://swindale.ecc.ubc.ca) `.srf` and `.tsf` files, and Rodrigo Quian Quiroga's
[simulated data](http://www.vis.caltech.edu/~rodri/Wave_clus/Simulator.zip) `.mat` files.
spyke can be extended to work with other electrophysiology file formats. Some [sample
data](http://swindale.ecc.ubc.ca/spyke) is available. [Spike sorting for polytrodes: a divide
and conquer approach](http://dx.doi.org/10.3389/fnsys.2014.00006) is a paper describing the
overall approach. spyke is described in greater detail in Chapter 3 and Appendix C.2 of this
[thesis](http://mspacek.github.io/mspacek_thesis.pdf).

A much older version is described in the paper
[Python for large-scale electrophysiology](http://www.frontiersin.org/Neuroinformatics/10.3389/neuro.11.009.2008/abstract).

Some functionality was inherited from Tim Blanche's Delphi program "SurfBawd". Some icons were
copied from Ubuntu's [Humanity](http://launchpad.net/humanity) icon theme.

### Dependencies

spyke requires recent versions of the following to be installed:

* [Python](http://python.org) (>=3.6)
* [PyQt5](http://www.riverbankcomputing.co.uk/software/pyqt)
  ([PySide](http://pyside.org) will probably work too, but is untested)
* [PyOpenGL](http://pyopengl.sourceforge.net)
* [IPython](http://ipython.org)
* [numpy](http://numpy.org)
* [scipy](http://scipy.org)
* [matplotlib](http://matplotlib.org)
* [Cython](http://cython.org)
* [MDP](http://mdp-toolkit.sourceforge.net)
* [scikit-learn](http://scikit-learn.org)
* [PyWavelets](http://www.pybytes.com/pywavelets)
* [jsonpickle](https://github.com/jsonpickle/jsonpickle)
* [simplejson](https://github.com/simplejson/simplejson)

spyke is developed in [Xubuntu](http://xubuntu.org) 20.04. It should work in other Linux
distributions, and is known to work in OSX. In principle, it should also work in Windows.

spyke is a Qt5 application. To make it look like a normal GTK application on a GTK-based
desktop like (X)ubuntu, make sure to install the `qt5-style-plugins` package in (X)ubuntu, and
then launch the `qt5ct` config app to set your Qt style to `gtk2`.

### Installation

Most often, you'll want to do a "developer" install, which lets you work on or otherwise
update spyke in-place, in whatever folder you cloned it into with git, while still being able
to call `import spyke` and use it as a library system-wide. This creates an egg-link in your
system site-packages or dist-packages folder to the source code:

```
$ sudo python setup.py develop
```

or the equivalent using pip:

```
$ sudo pip3 install -e .
```

This will also install a bash script on your system so that you can simply type `spyke` at the
command line to launch it from anywhere.

```
$ spyke
```

Alternatively, you can launch spyke with:

```
$ python -m spyke.main
```

which gives you some more flexibility, such as allowing you to specify what version of Python
you want to use.

Instead of development mode, you can also install by copying the code to your system Python
installation:

```
$ sudo python setup.py install
```

However, unlike in developer mode, every time you update from git, you'll have to re-run the
above installation command.

### Documentation

See [TUTORIAL.md](TUTORIAL.md) for a fairly brief tutorial.

Many keyboard shortcuts are available. Tooltips give some hints. You can also discover them by
searching for `keyPressEvent` methods in the code.
