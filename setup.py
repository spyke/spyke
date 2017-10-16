"""spyke installation script

A "developer" install lets you work on or otherwise update spyke in-place, in whatever
folder you cloned it into with git, while still being able to call `import spyke` and use it
as a library system-wide. This creates an egg-link in your system site-packages or
dist-packages folder to the source code:

>> sudo python setup.py develop

This will also install a bash script on your system so that you can simply type `spyke` at the
command line to launch it from anywhere.

Other setup.py commands:

For a normal installation (copies files to system site-packages or dist-packages folder):
>>> sudo python setup.py install

Build extensions in-place for development:
>>> python setup.py build_ext --inplace

Create a source distribution and force tar.gz file:
>>> python setup.py sdist --formats=gztar

Create a binary distribution:
>>> python setup.py bdist_wininst

NOTE: Make sure there's a MANIFEST.in that includes all the files you want to place
in the tarball. See http://wiki.python.org/moin/DistUtilsTutorial
"""

## TODO: for automatic dependency resolution, add `install_requires` kwarg to setup()

from setuptools import setup
from spyke.__version__ import __version__

# list of extra (non .py) files required by the spyke package, relative to its path:
spyke_files = ["res/*.png"]

setup(name='spyke',
      version=__version__,
      license='BSD',
      description='Visualization, navigation, and spike sorting of extracellular '
                  'waveform data',
      author='Martin Spacek, Reza Lotun',
      author_email='git at mspacek mm st',
      url='http://spyke.github.io',
      # include subfolders with code as additional packages
      packages=['spyke'],
      package_data={'spyke' : spyke_files},
      scripts=['bin/spyke'],
      )
