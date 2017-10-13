"""spyke installation script

to do a "developer" install, such that you can work on the code where it is on your system,
while still being able to call `import spyke` and use it as a library. This creates an
egg-link in your system site-packages or dist-packages folder:
>> sudo python setup.py develop

to do a normal installation:
>> python setup.py install

to build extensions in-place for development:
>>> python setup.py build_ext --inplace

to create source distribution and force tar.gz file:
>>> python setup.py sdist --formats=gztar

to create binary distribution:
>>> python setup.py bdist_wininst

NOTE: Make sure there's a MANIFEST.in that includes all the files you want to place
in the tarball. See http://wiki.python.org/moin/DistUtilsTutorial
"""

from setuptools import setup # setuptools adds develop option that distutils lacks?
from spyke.__init__ import __version__

spyke_files = ["res/*.png"] # list of extra (non .py) files required by the spyke package, relative to its path

setup(name='spyke',
      version=__version__,
      license='BSD',
      description='Visualization, navigation, and spike sorting of extracellular '
                  'waveform data',
      author='Martin Spacek, Reza Lotun',
      author_email='git at mspacek mm st',
      url='http://spyke.github.io',
      # have to explicitly include subfolders with code as additional packages
      packages=['spyke'],
      package_data={'spyke' : spyke_files},
      #cmdclass={'build_ext': build_ext},
      #ext_modules=[#simple_detect_cy,
      #             detect_cy,
      #             cython_test,
      #             cy_thread_test
      #             ],
      )
